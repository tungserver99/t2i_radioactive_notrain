"""Paired clean and one-step OGSC teacher image generation."""

import json
import math
import shutil
import sys
import time
from pathlib import Path

import torch
from .control import OBJECTIVE, one_step
from .progress import progress_line
from .state import atomic_path, read_json, sha256, write_json


def image_name(index):
    return f'sample_{index:04d}.png'


def validate_marked_record(record, row, seed, gamma):
    if (record.get('image_id') != row['image_id'] or record.get('text') != row['text']
            or record.get('seed') != seed):
        raise RuntimeError('Cached OGSC latent identity mismatch')
    if 'gamma' in record and not math.isclose(float(record['gamma']), float(gamma), rel_tol=0, abs_tol=1e-12):
        raise RuntimeError('Cached OGSC latent gamma mismatch')
    latent = record.get('latent')
    if not isinstance(latent, torch.Tensor) or not torch.isfinite(latent).all():
        raise RuntimeError('Cached OGSC latent is missing, invalid, or nonfinite')
    return record


def generation_progress_message(phase, completed, total, elapsed_seconds,
                               work_completed=None, work_total=None):
    total = max(int(total), 1)
    completed = min(max(int(completed), 0), total)
    work_total = total if work_total is None else max(int(work_total), 0)
    work_completed = completed if work_completed is None else min(max(int(work_completed), 0), work_total)
    if work_total == 0:
        eta = 'ETA 0 min (all cached)'
    elif work_completed < min(10, work_total):
        eta = f'ETA warming up ({work_completed}/{min(10, work_total)} timed items)'
    else:
        seconds = elapsed_seconds * (work_total - work_completed) / work_completed
        eta = f'ETA {seconds / 60:.1f} min'
    return (f'OGSC {phase}: {completed}/{total} ({completed / total:.1%}) | '
            f'{work_completed}/{work_total} this run | {elapsed_seconds / 60:.1f} min elapsed | {eta}')


def write_image_metadata(folder, rows):
    """Write prompt-to-image mapping without rereading generated PNGs."""
    with (folder / 'metadata.jsonl').open('w', encoding='utf-8') as stream:
        for index, row in enumerate(rows):
            stream.write(json.dumps({'file_name': image_name(index), 'text': row['text']}) + '\n')


def clean_record_path(records, calibration_records, split, index):
    """Find the shared clean latent or the validation latent made during gamma tuning."""
    path = records / f'{index:04d}.pt'
    if path.is_file():
        return path
    if split == 'validation':
        tuned_path = calibration_records / f'{index:04d}.pt'
        if tuned_path.is_file():
            return tuned_path
    return path


def _has_png_signature(path):
    try:
        with path.open('rb') as stream:
            return stream.read(8) == b'\x89PNG\r\n\x1a\n'
    except OSError:
        return False


def reuse_selected_validation_images(output, destination, rows, gamma):
    """Reuse the already decoded selected-gamma validation trial PNGs."""
    trial_dir = output / 'calibration/gamma_trials' / f'{gamma:.10f}'
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / 'selected_gamma_reuse.json'
    marker_data = {'gamma': gamma, 'count': len(rows), 'source_dir': str(trial_dir.resolve())}
    expected_names = [image_name(index) for index in range(len(rows))]
    sources = {path.name: path for path in trial_dir.glob('sample_*.png') if path.is_file()}
    targets = {path.name: path for path in destination.glob('sample_*.png') if path.is_file()}
    if marker.is_file() and read_json(marker) == marker_data:
        if all(name in sources and name in targets for name in expected_names):
            print(f'OGSC validation: all {len(rows)} selected-gamma trial images already present.', flush=True)
            return 0

    scan_started = time.perf_counter()
    pending = []
    for index, name in enumerate(expected_names, start=1):
        source = sources.get(name)
        target = targets.get(name)
        if source is None:
            pending.append(index)
        elif (target is None or target.stat().st_size != source.stat().st_size
              or not _has_png_signature(target)):
            pending.append(index)
        if index == 1 or index % 100 == 0 or index == len(rows):
            print(progress_line('validation trial reuse scan', index, len(rows),
                                time.perf_counter() - scan_started), flush=True)
    if any(name not in sources for name in expected_names):
        for name in expected_names:
            target = targets.get(name)
            if target is not None and not _has_png_signature(target):
                target.unlink()
        print(f'OGSC validation trial cache incomplete at {trial_dir}; will compute missing controls.', flush=True)
        return None
    pending = [index - 1 for index in pending]
    if not pending:
        print(f'OGSC validation: all {len(rows)} selected-gamma trial images already present.', flush=True)
        write_json(marker, marker_data)
        return 0
    started = time.perf_counter()
    print(f'OGSC validation: reusing selected-gamma trial PNGs; copying {len(pending)} files, no VJP/decode.', flush=True)
    for work_i, index in enumerate(pending, start=1):
        source = sources[image_name(index)]
        target = destination / image_name(index)
        with atomic_path(target) as temporary:
            shutil.copyfile(source, temporary)
            if (temporary.stat().st_size != source.stat().st_size
                    or not _has_png_signature(temporary)):
                raise IOError(f'Incomplete validation trial copy: {source} -> {target}')
        if work_i == 1 or work_i % 25 == 0 or work_i == len(pending):
            print(generation_progress_message(
                'validation trial image reuse', work_i, len(pending), time.perf_counter() - started,
                work_completed=work_i, work_total=len(pending)), flush=True)
    write_json(marker, marker_data)
    return len(pending)


def _save_png(image, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_path(path) as tmp:
        image.save(tmp, format='PNG')


def _semantic_teacher(paths):
    harness = Path(paths['weight32_repo']) / 'BEST_TEACHER_32bits_refine/harness'
    sys.path.insert(0, str(harness))
    import semantic_teacher
    return semantic_teacher


def _decode(pipe, latent):
    with torch.inference_mode():
        decoded = pipe.vae.decode(latent / pipe.vae.config.scaling_factor).sample
        return pipe.image_processor.postprocess(decoded, output_type='pil')


def _rows(paths, split):
    rows = read_json(Path(paths['source_run']) / 'prompts.json')[split]
    if split == 'validation':
        rows = rows[:read_json(Path(paths['source_run']) / 'config.json')['validation_generation_n']]
    return rows


def infer_one(pipe, sources, scorer, carrier, prompt, clean_latent, gamma):
    """Production path: control one clean final latent and decode only OGSC output."""
    marked_latent, diagnostics = one_step(
        clean_latent, sources.signed_view(scorer, carrier, prompt), gamma)
    image = _decode(pipe, marked_latent)[0]
    return image, marked_latent, diagnostics


def generate_one(paths, sources, prompt, seed, output_image=None):
    """Generate one OGSC PNG directly from a prompt/seed, without clean PNG output."""
    if not torch.cuda.is_available():
        raise RuntimeError('Production OGSC inference needs CUDA; local checks use CPU smoke')
    output = Path(paths['output_run'])
    selected = read_json(output / 'calibration/gamma.json')
    if (selected.get('status') != 'selected' or selected.get('gamma') is None
            or selected.get('objective') != OBJECTIVE):
        raise RuntimeError('Run tune-gamma and select a positive validation gamma before inference')
    gamma = float(selected['gamma'])
    if output_image is None:
        output_image = output / 'images/production' / f'ogsc_seed_{int(seed)}.png'
    image_path = Path(output_image).expanduser().resolve()
    teacher = _semantic_teacher(paths)
    pipe = scorer = carrier = None
    started = time.perf_counter()
    try:
        # Both roles are resident for low-latency inference. The experiment
        # generator remains phase-separated to reduce peak VRAM use.
        pipe = teacher.load_native(Path(paths['teacher_clean']))
        scorer = sources.extractor(Path(paths['teacher_clean']), device='cuda')
        carrier = sources.carrier(Path(paths['carrier_dir']), 'cuda')
        sources.verify_frozen(scorer)
        generator = torch.Generator('cuda').manual_seed(int(seed))
        with torch.inference_mode():
            clean_latent = pipe([prompt], generator=[generator], num_inference_steps=50,
                                guidance_scale=7.5, height=512, width=512,
                                output_type='latent').images.detach()
        image, marked_latent, diagnostics = infer_one(
            pipe, sources, scorer, carrier, prompt, clean_latent, gamma)
        _save_png(image, image_path)
        record = {'protocol': 'OGSC-32 direct one-sample inference', 'prompt': prompt,
                  'seed': int(seed), 'gamma': gamma, 'steps': 50, 'cfg': 7.5,
                  'resolution': [512, 512], 'dtype': 'float16 generation / float32 semantic control',
                  'total_seconds': time.perf_counter() - started,
                  'loss_before': diagnostics['loss_before'], 'gradient_rms': diagnostics['gradient_rms'],
                  'direction_l2': diagnostics['direction_l2'],
                  'clean_png_written': False, 'gamma_selection_sha256': sha256(output / 'calibration/gamma.json')}
        write_json(image_path.with_suffix('.json'), record)
        return {'image': str(image_path), 'metadata': str(image_path.with_suffix('.json')),
                'gamma': gamma, 'total_seconds': record['total_seconds']}
    finally:
        del pipe, scorer, carrier
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def generate_split(paths, sources, split):
    if not torch.cuda.is_available():
        raise RuntimeError('Teacher generation needs CUDA; local CPU checks use smoke instead')
    source, output = Path(paths['source_run']), Path(paths['output_run'])
    config = read_json(source / 'config.json')
    rows = _rows(paths, split)
    seeds = [int(config['seed']) * 1_000_000 + int(r['image_id']) for r in rows]
    gamma = float(read_json(output / 'calibration/gamma.json')['gamma'])
    clean_dir, wm_dir = output / 'images' / f'teacher_clean_{split}', output / 'images' / f'teacher_ogsc_{split}'
    clean_dir.mkdir(parents=True, exist_ok=True); wm_dir.mkdir(parents=True, exist_ok=True)
    generation = {'split': split, 'gamma': gamma, 'objective': OBJECTIVE,
                  'steps': 50, 'cfg': 7.5, 'resolution': [512, 512],
                  'dtype': 'float16', 'scheduler': 'original SD2.1 PNDM', 'seed_rule': 'config.seed*1000000+image_id',
                  'prompts_sha256': sha256(source / 'prompts.json')}
    prior_generation = wm_dir / 'generation.json'
    if prior_generation.is_file():
        prior = read_json(prior_generation)
        if (prior.get('split') != split or prior.get('objective') != OBJECTIVE
                or prior.get('prompts_sha256') != generation['prompts_sha256']
                or not math.isclose(float(prior.get('gamma', float('nan'))), gamma,
                                    rel_tol=0, abs_tol=1e-12)):
            raise RuntimeError(f'Existing {split} OGSC outputs use a different split, prompt set, or gamma')
    for folder in (clean_dir, wm_dir):
        write_json(folder / 'generation.json', generation)
    teacher = _semantic_teacher(paths)
    records = output / 'state/latents' / split
    records.mkdir(parents=True, exist_ok=True)
    calibration_records = output / 'calibration/validation/latents'
    clean_indices = [i for i in range(len(rows)) if not (
        (clean_dir / image_name(i)).exists()
        and clean_record_path(records, calibration_records, split, i).is_file())]
    clean_todo = len(clean_indices)
    print(f'OGSC clean {split}: {len(rows) - clean_todo} cached; {clean_todo} remaining.', flush=True)
    # Clean sampling uses the original FP16 pipeline only.
    if clean_indices:
        pipe = teacher.load_native(Path(paths['teacher_clean']))
        clean_started = time.perf_counter()
        try:
            for work_i, i in enumerate(clean_indices, start=1):
                row, seed = rows[i], seeds[i]
                clean_png, latent_path = clean_dir / image_name(i), records / f'{i:04d}.pt'
                tuned_path = clean_record_path(records, calibration_records, split, i)
                if split == 'validation' and tuned_path != latent_path and tuned_path.is_file():
                    tuned_record = torch.load(tuned_path, map_location='cpu', weights_only=True)
                    if (tuned_record.get('image_id') != row['image_id']
                            or tuned_record.get('text') != row['text']
                            or tuned_record.get('seed') != seed):
                        raise RuntimeError(f'Cached validation latent identity mismatch for {i}')
                    latent = tuned_record['latent'].to('cuda')
                    if not clean_png.is_file():
                        _save_png(_decode(pipe, latent)[0], clean_png)
                    with atomic_path(latent_path) as tmp:
                        torch.save(tuned_record, tmp)
                else:
                    gen = torch.Generator('cuda').manual_seed(seed)
                    with torch.inference_mode():
                        latent = pipe([row['text']], generator=[gen], num_inference_steps=50,
                                      guidance_scale=7.5, height=512, width=512, output_type='latent').images
                    latent = latent.detach()
                    _save_png(_decode(pipe, latent)[0], clean_png)
                    with atomic_path(latent_path) as tmp:
                        torch.save({'latent': latent.cpu(), 'image_id': row['image_id'], 'text': row['text'], 'seed': seed}, tmp)
                if work_i == 1 or work_i % 25 == 0 or work_i == clean_todo:
                    print(generation_progress_message(
                        f'clean {split}', len(rows) - clean_todo + work_i, len(rows),
                        time.perf_counter() - clean_started,
                        work_completed=work_i, work_total=clean_todo), flush=True)
        finally:
            del pipe
            torch.cuda.empty_cache()

    # Gamma tuning already computed and decoded the selected candidate for every
    # validation prompt. Reuse those exact PNGs instead of repeating the VJP.
    if split == 'validation':
        selected_gamma = read_json(output / 'calibration/gamma.json')
        if (selected_gamma.get('status') == 'selected'
                and math.isclose(float(selected_gamma.get('gamma', -1)), gamma, rel_tol=0, abs_tol=1e-12)
                and selected_gamma.get('validation_prompts_sha256') == sha256(source / 'prompts.json')):
            reuse_selected_validation_images(output, wm_dir, rows, gamma)

    # Only samples still missing their marked PNG need a control latent.
    decode_indices = [i for i in range(len(rows)) if not (wm_dir / image_name(i)).is_file()]
    control_indices = []
    for i in decode_indices:
        row = rows[i]
        marked_latent_path = records / f'{i:04d}_wm.pt'
        if marked_latent_path.is_file():
            cached = torch.load(marked_latent_path, map_location='cpu', weights_only=True)
            validate_marked_record(cached, row, seeds[i], gamma)
        else:
            control_indices.append(i)
    control_todo = len(control_indices)
    print(f'OGSC control {split}: {len(rows) - control_todo} samples need no new control; {control_todo} remaining.', flush=True)
    if control_indices:
        scorer = sources.extractor(Path(paths['teacher_clean']), device='cuda')
        carrier = sources.carrier(source / 'carrier', 'cuda')
        sources.verify_frozen(scorer)
        control_started = time.perf_counter()
        try:
            for work_i, i in enumerate(control_indices, start=1):
                row, seed = rows[i], seeds[i]
                record_path = clean_record_path(records, calibration_records, split, i)
                record = torch.load(record_path, map_location='cpu', weights_only=True)
                if record['image_id'] != row['image_id'] or record['text'] != row['text'] or record['seed'] != seed:
                    raise RuntimeError(f'Clean latent identity mismatch for {i}')
                z = record['latent'].to('cuda')
                started = time.perf_counter()
                marked, diagnostics = one_step(z, sources.signed_view(scorer, carrier, row['text']), gamma)
                control_time = time.perf_counter() - started
                wm_latent = records / f'{i:04d}_wm.pt'
                with atomic_path(wm_latent) as tmp:
                    torch.save({'latent': marked.cpu(), 'image_id': row['image_id'], 'text': row['text'],
                                'seed': seed, 'gamma': gamma, 'loss_before': diagnostics['loss_before'],
                                'control_seconds': control_time}, tmp)
                if work_i == 1 or work_i % 25 == 0 or work_i == control_todo:
                    print(generation_progress_message(
                        f'control {split}', work_i, control_todo,
                        time.perf_counter() - control_started,
                        work_completed=work_i, work_total=control_todo), flush=True)
        finally:
            del scorer, carrier
            torch.cuda.empty_cache()

    # Decode through the same native FP16 VAE path used for clean images.
    decode_todo = len(decode_indices)
    print(f'OGSC marked decode {split}: {len(rows) - decode_todo} cached; {decode_todo} remaining.', flush=True)
    if decode_indices:
        pipe = teacher.load_native(Path(paths['teacher_clean']))
        decode_started = time.perf_counter()
        try:
            for work_i, i in enumerate(decode_indices, start=1):
                path = wm_dir / image_name(i)
                record = torch.load(records / f'{i:04d}_wm.pt', map_location='cpu', weights_only=True)
                _save_png(_decode(pipe, record['latent'].to('cuda'))[0], path)
                if work_i == 1 or work_i % 25 == 0 or work_i == decode_todo:
                    print(generation_progress_message(
                        split, len(rows) - decode_todo + work_i, len(rows),
                        time.perf_counter() - decode_started,
                        work_completed=work_i, work_total=decode_todo), flush=True)
        finally:
            del pipe
            torch.cuda.empty_cache()

    for folder in (clean_dir, wm_dir):
        write_image_metadata(folder, rows)
    return {'split': split, 'count': len(rows), 'gamma': gamma,
            'metadata': 'metadata.jsonl (per-image PNG hashes disabled)'}
