"""Staged OGSC-32 workflow; teacher training and historic calibration are absent."""

import argparse
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

import torch
from .assets import resolve_clean_model, verify_carrier
from .carrier_bridge import Sources
from .evaluate import score_outputs
from .generation import generate_one, generate_split
from .paths import default_paths, validate_run_inputs
from .progress import stage_progress
from .quality import tune_gamma
from .state import read_json, sha256, write_json


def _parser():
    parser = argparse.ArgumentParser(prog='python -m ogsc.run')
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'smoke', 'tune-gamma', 'generate-one', 'generate-teacher', 'evaluate-teacher',
                 'train-student', 'evaluate-student', 'full'):
        cmd = sub.add_parser(name)
        for arg in ('weight32-repo', 'oracle-repo', 'source-run', 'carrier-dir', 'teacher-clean',
                    'model-cache', 'output-run', 'student-runner', 'student-source-run'):
            cmd.add_argument('--' + arg)
        cmd.add_argument('--offload', action='store_true')
        if name == 'generate-teacher':
            cmd.add_argument('--split', choices=('train', 'validation', 'test'), action='append')
        if name == 'generate-one':
            cmd.add_argument('--prompt', required=True)
            cmd.add_argument('--seed', required=True, type=int)
            cmd.add_argument('--output-image')
    return parser


def _prepare(paths):
    inputs = validate_run_inputs(paths)
    carrier, hashes = verify_carrier(paths)
    output = Path(paths['output_run'])
    output.mkdir(parents=True, exist_ok=True)
    provenance_path = output / 'provenance.json'
    prior = read_json(provenance_path) if provenance_path.is_file() else None
    if prior is not None:
        if (prior.get('source_files_sha256') != inputs['source_sha256']
                or prior.get('source_run_sha256') != inputs['source_run_sha256']
                or prior.get('carrier_artifact_sha256') != hashes):
            raise RuntimeError('Existing output run belongs to different source/carrier inputs; use a new --output-run')
        # Reuse the exact pinned snapshot path recorded by the first prepare.
        paths['teacher_clean'] = prior.get('resolved_model_path')
    model, model_identity = resolve_clean_model(paths, inputs['assets'])
    paths['teacher_clean'] = str(model)
    for name in ('config.json', 'prompts.json'):
        write_json(output / name, inputs['config'] if name == 'config.json' else inputs['prompts'])
    implementation_sha256 = {p.relative_to(Path(__file__).resolve().parents[1]).as_posix(): sha256(p)
                              for p in sorted(Path(__file__).resolve().parent.glob('*.py'))}
    if prior is not None:
        provenance = prior
        provenance['model'] = model_identity
        provenance['resolved_model_path'] = str(model)
        versions = provenance.setdefault('resume_implementation_sha256', [])
        if implementation_sha256 != provenance.get('implementation_sha256') and implementation_sha256 not in versions:
            versions.append(implementation_sha256)
    else:
        provenance = {
        'protocol': 'OGSC-32 soft-min final-latent one-step',
        'code_git_commit': _git_commit(), 'source_files_sha256': inputs['source_sha256'],
        'implementation_sha256': implementation_sha256,
        'source_run_sha256': inputs['source_run_sha256'], 'carrier_artifact_sha256': hashes,
        'model': model_identity, 'resolved_model_path': str(model),
        'source_run_path': paths['source_run'], 'carrier_path': paths['carrier_dir'],
        'prompts_sha256': sha256(Path(paths['source_run']) / 'prompts.json'),
        'random_seed': inputs['config']['seed'], 'gamma': None,
        'sampling': {'scheduler': 'source SD2.1 PNDM', 'steps': 50, 'cfg': 7.5,
                     'resolution': [512, 512], 'generation_precision': 'float16',
                     'semantic_precision': 'float32'},
        'packages': inputs['packages'], 'carrier_coordinates': 'original normalized_coordinates; no median subtraction',
        'created_by': 'ogsc.run prepare',
        }
    write_json(provenance_path, provenance)
    stages_path = output / 'state/stages.json'
    stages = read_json(stages_path) if stages_path.is_file() else {}
    stages['prepare'] = {'status': 'complete', 'model': model_identity, 'carrier_sha256': hashes}
    write_json(stages_path, stages)
    print(json.dumps({'status': 'ready', 'model': model_identity, 'carrier': hashes,
                      'output_run': str(output)}, indent=2))
    return inputs, carrier


def _git_commit():
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=Path(__file__).resolve().parents[1],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return os.environ.get('GIT_COMMIT', 'unavailable: workspace has no Git metadata')


def _sources(paths):
    identity_path = Path(paths['output_run']) / 'provenance.json'
    if not identity_path.is_file():
        raise RuntimeError('Run `python -m ogsc.run prepare` first')
    identity = read_json(identity_path)
    paths['teacher_clean'] = identity['resolved_model_path']
    if not Path(paths['teacher_clean']).is_dir():
        raise RuntimeError(f"Prepared clean model snapshot is missing: {paths['teacher_clean']}")
    inputs = validate_run_inputs(paths)
    if inputs['source_sha256'] != identity['source_files_sha256']:
        raise RuntimeError('Weight32/oracle source files changed since prepare; create a new output run')
    if inputs['source_run_sha256'] != identity['source_run_sha256']:
        raise RuntimeError('Source config/prompts/assets changed since prepare; create a new output run')
    _, carrier_hashes = verify_carrier(paths)
    if carrier_hashes != identity['carrier_artifact_sha256']:
        raise RuntimeError('Original carrier changed since prepare; create a new output run')
    return Sources(paths)


def _update_stage(paths, name, result):
    path = Path(paths['output_run']) / 'state/stages.json'
    state = read_json(path) if path.exists() else {}
    state[name] = {'status': 'complete', 'result': result}
    write_json(path, state)


def _gamma_is_reusable(paths):
    output = Path(paths['output_run'])
    gamma_path = output / 'calibration/gamma.json'
    if not gamma_path.is_file():
        image_root = output / 'images'
        started = any(image_root.glob('teacher_clean_*/sample_*.png')) if image_root.exists() else False
        if started:
            raise RuntimeError('OGSC images exist but gamma artifact is missing; refusing to retune mid-run')
        return None
    gamma = read_json(gamma_path)
    if gamma.get('status') != 'selected' or gamma.get('gamma') is None or float(gamma['gamma']) <= 0:
        raise RuntimeError('Existing gamma artifact has no selected positive gamma; inspect it or use a new output run')
    expected_prompts = sha256(Path(paths['source_run']) / 'prompts.json')
    if gamma.get('validation_prompts_sha256') != expected_prompts:
        raise RuntimeError('Existing gamma artifact validation prompts changed; use a new output run')
    return gamma


def _record_gamma(paths, gamma):
    provenance_path = Path(paths['output_run']) / 'provenance.json'
    provenance = read_json(provenance_path)
    provenance['gamma'] = gamma['gamma']
    provenance['gamma_sha256'] = sha256(Path(paths['output_run']) / 'calibration/gamma.json')
    write_json(provenance_path, provenance)


def _smoke(paths):
    from .control import one_step, soft_min_loss
    torch.manual_seed(31)
    model = torch.nn.Linear(12, 32, bias=False).eval().requires_grad_(False)
    latent = torch.randn(1, 12)
    marked, info = one_step(latent, model, 1e-4)
    if not soft_min_loss(model(marked)).item() < info['loss_before']:
        raise RuntimeError('Smoke objective did not improve at small gamma')
    _update_stage(paths, 'smoke', {'device': 'cpu', 'vjp': 1, 'gamma_zero_supported': True})
    print('OGSC CPU smoke passed (toy model only; no SD model/GPU loaded).')


def _student(paths, evaluate=False):
    runner = paths.get('student_runner')
    source = Path(paths['student_source_run'])
    required = [source / 'complete.json', source / 'teacher_choice.json',
                source / 'result/threshold.json', source / 'result/threshold_student.json',
                source / 'result/threshold_teacher.json', source / 'validation_cache',
                source / 'images/teacher_wm_train/image_hashes.json',
                source / 'images/teacher_wm_validation/image_hashes.json',
                source / 'images/teacher_wm_test/image_hashes.json',
                source / 'teacher_fp/refined/adapter_model.safetensors']
    missing = [str(path) for path in required if not path.exists()]
    if not runner or not Path(runner).is_file():
        detail = 'Exact student bundle is not present.'
        status = 'STUDENT_TRAINER_MISSING'
    elif missing:
        detail = ('Exact 12,500-step trainer is available at ' + str(runner) +
                  ', but its compatible Structure teacher source artifacts are absent. Missing: ' +
                  ', '.join(missing))
        status = 'STUDENT_SOURCE_ARTIFACTS_MISSING'
    else:
        detail = ('Exact trainer and source artifacts exist, but OGSC image overlay is not staged; '
                  'refusing to train on the original non-OGSC teacher images.')
        status = 'STUDENT_OGSC_DATASET_NOT_STAGED'
    result = {'status': status, 'detail': detail,
              'trainer': str(runner) if runner else None,
              'student_source_run': str(source), 'training_invoked': False}
    _update_stage(paths, 'evaluate_student' if evaluate else 'train_student', result)
    print(json.dumps(result, indent=2))
    return result


def _write_report(paths, student_result):
    output = Path(paths['output_run'])
    metrics = read_json(output / 'result/teacher_key_metrics.json')
    gamma = read_json(output / 'calibration/gamma.json')
    lines = ['# OGSC-32 run summary', '',
             'Teacher images were generated from clean SD2.1 with one final-latent soft-min control step.',
             'This report records measured code outputs only; it is not a claim of research success or student transfer.',
             '', f"- Selected validation gamma: `{gamma.get('gamma')}`",
             '- Gamma selection used paired validation quality only; test FKR was not used.',
             f"- Student status: `{student_result['status']}`",
             f"- Student detail: {student_result['detail']}", '',
             '## Teacher full-key recovery', '',
             '| Split | Model | Group size | Bit accuracy | FKR | Groups |',
             '|---|---|---:|---:|---:|---:|']
    for label, record in metrics['results'].items():
        split = label.rsplit('_', 1)[-1]
        model = 'OGSC SD2.1' if '_ogsc_' in label else 'Clean SD2.1'
        for row in record['official_key_metrics']['rows']:
            lines.append(f"| {split} | {model} | {row['images_per_group']} | {row['bit_accuracy']:.6f} | {row['full_key_recovery_rate']:.6f} | {row['groups_per_partition']} |")
    lines += ['', '## Validation quality trials', '',
              '| Gamma | Mean LPIPS | Median LPIPS | CLIP drop | Mean pixel MSE | Pass |',
              '|---:|---:|---:|---:|---:|---|']
    for trial in gamma.get('trials', []):
        lines.append(f"| {trial['gamma']:.9f} | {trial['mean_paired_lpips']:.6f} | {trial['median_paired_lpips']:.6f} | {trial['mean_clip_drop']:.6f} | {trial['mean_pixel_mse']:.8f} | {trial['passes']} |")
    lines += ['', 'See `provenance.json`, `result/teacher_quality.json`, and `result/teacher_key_metrics.json` for source identity and full per-image reports.']
    path = output / 'result/summary.md'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return {'summary': str(path), 'sha256': sha256(path), 'student_status': student_result['status']}


def main(argv=None):
    args = _parser().parse_args(argv)
    paths = default_paths(args)
    try:
        if args.command == 'prepare':
            with stage_progress('prepare inputs, hashes, and pinned model'):
                _prepare(paths)
            return 0
        if args.command == 'smoke':
            with stage_progress('CPU logic smoke check'):
                _smoke(paths)
            return 0
        if args.command == 'full':
            with stage_progress('prepare inputs, hashes, and pinned model'):
                _prepare(paths)
            with stage_progress('CPU logic smoke check'):
                _smoke(paths)
            with stage_progress('validate frozen sources and carrier'):
                sources = _sources(paths)
            with stage_progress('check reusable validation gamma'):
                gamma = _gamma_is_reusable(paths)
            if gamma is None:
                with stage_progress('validation gamma tuning'):
                    gamma = tune_gamma(paths, sources)
                _update_stage(paths, 'tune_gamma', gamma)
            else:
                _update_stage(paths, 'tune_gamma', {'gamma': gamma['gamma'], 'reused': True})
                print(f"OGSC resume: reusing frozen gamma={gamma['gamma']:.9g}; skipping tune-gamma.", flush=True)
            _record_gamma(paths, gamma)
            for split in ('train', 'validation', 'test'):
                with stage_progress(f'generate clean and OGSC {split} images'):
                    result = generate_split(paths, sources, split)
                _update_stage(paths, f'generate_{split}', result)
            with stage_progress('evaluate teacher PNGs and key recovery'):
                result = score_outputs(paths, sources)
            _update_stage(paths, 'evaluate_teacher', result)
            with stage_progress('check student evaluation prerequisites'):
                student = _student(paths)
            with stage_progress('write final run report'):
                result = _write_report(paths, student)
            _update_stage(paths, 'report', result)
            return 0
        if args.command == 'train-student':
            _student(paths); return 0
        if args.command == 'evaluate-student':
            _student(paths, evaluate=True); return 0
        with stage_progress('validate frozen sources and carrier'):
            sources = _sources(paths)
        output = Path(paths['output_run'])
        if args.command == 'generate-one':
            with stage_progress('generate one production OGSC image'):
                result = generate_one(paths, sources, args.prompt, args.seed, args.output_image)
        elif args.command == 'tune-gamma':
            with stage_progress('validation gamma tuning'):
                result = tune_gamma(paths, sources)
        elif args.command == 'generate-teacher':
            splits = args.split or ['train', 'validation', 'test']
            result = {}
            for split in splits:
                with stage_progress(f'generate clean and OGSC {split} images'):
                    result[split] = generate_split(paths, sources, split)
        elif args.command == 'evaluate-teacher':
            with stage_progress('evaluate teacher PNGs and key recovery'):
                result = score_outputs(paths, sources)
        else:
            raise AssertionError(args.command)
        _update_stage(paths, args.command.replace('-', '_'), result)
        print(json.dumps(result, indent=2, default=str))
        return 0
    except Exception as error:
        print(f'OGSC stage {args.command} failed: {error}', file=sys.stderr)
        if getattr(args, 'debug', False):
            traceback.print_exc()
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
