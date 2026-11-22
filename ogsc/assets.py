"""Resolve the pinned clean model and the original 32-bit carrier."""

import importlib.util
import os
import sys
from pathlib import Path

from .paths import source_paths
from .state import read_json, sha256


def _load_carrier_module(paths):
    layout = source_paths(paths)
    sys.path.insert(0, str(layout['harness']))
    path = layout['harness'] / 'semantic_carrier.py'
    spec = importlib.util.spec_from_file_location('_ogsc_original_semantic_carrier', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_carrier(paths):
    """Load and validate the byte-verified, frozen source carrier."""
    carrier_module = _load_carrier_module(paths)
    carrier_dir = Path(paths['carrier_dir'])
    carrier = carrier_module.SemanticCarrier.load(carrier_dir, device='cpu')
    key = read_json(carrier_dir / 'key.json')
    if (len(carrier.signs) != 32 or carrier.signs.shape != (32,)
            or not ((carrier.signs == 1) | (carrier.signs == -1)).all()
            or carrier.carrier_version is not None or carrier.medians is not None
            or key.get('num_bits') != 32 or key.get('carrier_version') is not None
            or key.get('medians') is not None):
        raise RuntimeError('OGSC requires the original uncentered 32-bit threshold carrier')
    return carrier, {
        name: sha256(carrier_dir / name)
        for name in ('complete.json', 'key.json', 'pca.pt', 'stability.json', 'provenance.json')
    }


def verify_model_snapshot(root, expected_hashes):
    root = Path(root).expanduser().resolve()
    missing = []
    mismatched = []
    for relative, expected in expected_hashes.items():
        path = root / relative
        if not path.is_file():
            missing.append(str(path))
        elif sha256(path) != expected:
            mismatched.append(f'{path}: expected {expected}, actual {sha256(path)}')
    if mismatched:
        raise RuntimeError('Pinned SD2.1 model hash mismatch:\n' + '\n'.join(mismatched))
    if missing:
        return False, missing
    return True, []


def resolve_clean_model(paths, assets):
    source = assets['teacher_source']
    repo_id = source['repository']
    revision = source['revision']
    expected = assets['teacher_clean_hashes']
    configured = paths.get('teacher_clean')
    metadata_path = assets.get('teacher_clean')

    weight_root = Path(paths['weight32_repo']) / 'BEST_TEACHER_32bits_refine'
    cache_candidates = []
    hf_cache = Path(os.environ.get('HF_HUB_CACHE', Path.home() / '.cache/huggingface/hub'))
    cache_candidates.append(hf_cache / f'models--{repo_id.replace("/", "--")}' / 'snapshots' / revision)
    for candidate in (configured, metadata_path, weight_root / 'models/sd2.1-pinned', *cache_candidates):
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if not path.is_dir():
            continue
        valid, missing = verify_model_snapshot(path, expected)
        if valid:
            return path.resolve(), {'repository': repo_id, 'revision': revision,
                                    'resolved_from': 'existing_local_snapshot',
                                    'model_sha256': expected}
        if configured and Path(configured).expanduser().resolve() == path.resolve():
            raise RuntimeError('Configured clean model is incomplete; missing: ' + ', '.join(missing))

    # Reuse an exact HF cache revision before fetching. snapshot_download itself
    # is cache aware and only contacts HF if this immutable revision is absent.
    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError('huggingface_hub is required to fetch the pinned clean SD2.1 model') from error
    cache_dir = Path(paths['model_cache']).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    downloaded = Path(snapshot_download(
        repo_id=repo_id,
        revision=revision,
        cache_dir=str(cache_dir),
        allow_patterns=list(expected),
    )).resolve()
    valid, missing = verify_model_snapshot(downloaded, expected)
    if not valid:
        raise RuntimeError('Downloaded pinned SD2.1 snapshot is incomplete; missing: ' + ', '.join(missing))
    return downloaded, {'repository': repo_id, 'revision': revision,
                        'resolved_from': 'huggingface_hub_snapshot_download',
                        'model_sha256': expected}
