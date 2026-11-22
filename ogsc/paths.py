"""Paths for OGSC; historical calibration files are deliberately not inputs."""

import importlib.metadata
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHT = 'BEST_TEACHER_32bits_refine'
METRICS = 'server21/files/var/lib/structure32_student12500/key_metrics.py'


def resolved(value):
    return Path(value).expanduser().resolve()


def find_weight_root(root):
    """Find the existing weight32 checkout above a standalone OGSC checkout."""
    root = resolved(root)
    for candidate in (root, *root.parents):
        if (candidate / WEIGHT).is_dir():
            return candidate
    return root


def find_oracle_root(root):
    """Find the existing oracle checkout beside the working OGSC repository."""
    root = resolved(root)
    seen = set()
    for ancestor in (root, *root.parents):
        for candidate in (ancestor / 'Hmark_inmodel', ancestor):
            candidate = candidate.resolve()
            if candidate in seen:
                continue
            seen.add(candidate)
            try:
                oracle_directory(candidate)
            except RuntimeError:
                continue
            return candidate
    return root.parent.parent / 'Hmark_inmodel'


def default_paths(args):
    def get(name, fallback=None):
        return getattr(args, name, None) or os.environ.get(f'OGSC_{name.upper()}') or fallback
    weight = resolved(get('weight32_repo', find_weight_root(ROOT)))
    source = resolved(get('source_run', weight / WEIGHT / 'server21/files/var/lib/structure32_student12500/runs/structure32'))
    teacher = get('teacher_clean')
    oracle = resolved(get('oracle_repo', find_oracle_root(ROOT)))
    return {
        'weight32_repo': str(weight), 'source_run': str(source),
        'carrier_dir': str(resolved(get('carrier_dir', source / 'carrier'))),
        'teacher_clean': str(resolved(teacher)) if teacher else None,
        'model_cache': str(resolved(get('model_cache', weight / 'model_cache'))),
        'oracle_repo': str(oracle),
        'output_run': str(resolved(get('output_run', weight / 'runs/ogsc32'))),
        'student_runner': str(resolved(get('student_runner', ROOT / 'structure_student12500/campaign.py'))),
        'student_source_run': str(resolved(get('student_source_run', ROOT / 'structure_student12500/data/threshold_quality_20261022/structure_margin_20261026/structure_margin_native'))),
        'baseline_run': str(resolved(get('baseline_run', weight / 'runs/ogsc32'))),
        'offload': bool(getattr(args, 'offload', False)),
    }


def oracle_directory(root):
    for base in (root / 'Malicious_quantization/bitmark_lora', root / 'bitmark_lora', root):
        if (base / 'semantic_oracle_math.py').is_file() and (base / 'semantic_sign_train.py').is_file():
            return base
    raise RuntimeError(f'Oracle helpers were not found under {root}')


def package_versions(names=('torch', 'diffusers', 'transformers', 'huggingface_hub', 'numpy', 'Pillow', 'lpips')):
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def source_paths(paths):
    weight = Path(paths['weight32_repo']) / WEIGHT
    harness = weight / 'harness'
    oracle = oracle_directory(Path(paths['oracle_repo']))
    metrics = weight / METRICS
    needed = [harness / n for n in ('semantic_carrier.py', 'semantic_teacher.py', 'semantic_sign_train.py')]
    needed += [oracle / n for n in ('semantic_oracle_math.py', 'semantic_sign_train.py')]
    needed.append(metrics)
    return {'weight_root': weight, 'harness': harness, 'oracle': oracle, 'metrics': metrics, 'required': needed}


def validate_run_inputs(paths):
    source = Path(paths['source_run'])
    carrier = Path(paths['carrier_dir'])
    required = [source / n for n in ('assets.json', 'config.json', 'prompts.json')]
    required += source_paths(paths)['required']
    required += [carrier / n for n in ('complete.json', 'key.json', 'pca.pt', 'stability.json', 'provenance.json')]
    missing = [p for p in required if not p.is_file()]
    if missing:
        raise RuntimeError('Required OGSC inputs are missing:\n' + '\n'.join(f'Missing {p}' for p in missing))
    from .state import read_json, sha256
    assets, config, prompts = (read_json(source / n) for n in ('assets.json', 'config.json', 'prompts.json'))
    if config.get('num_bits') != 32 or config.get('inference_steps') != 50:
        raise RuntimeError('Expected original 32-bit, 50-step source run configuration')
    if not assets.get('teacher_clean_hashes') or not assets.get('teacher_source', {}).get('repository') or not assets.get('teacher_source', {}).get('revision'):
        raise RuntimeError('assets.json must pin clean model repository, revision, and file hashes')
    if any(not prompts.get(split) for split in ('train', 'validation', 'test')):
        raise RuntimeError('prompts.json must contain nonempty train/validation/test splits')
    return {'assets': assets, 'config': config, 'prompts': prompts,
            'source_sha256': {str(p.resolve()): sha256(p) for p in source_paths(paths)['required']},
            'source_run_sha256': {n: sha256(source / n) for n in ('assets.json', 'config.json', 'prompts.json')},
            'carrier_sha256': {n: sha256(carrier / n) for n in ('complete.json', 'key.json', 'pca.pt', 'stability.json', 'provenance.json')},
            'packages': package_versions()}
