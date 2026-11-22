"""Score saved RGB8 PNGs with the original weight32 carrier and FKR metric."""

from pathlib import Path
import json
import time

import numpy as np
import torch

from .generation import _rows, image_name
from .progress import progress_line, stage_progress
from .state import read_json, sha256, write_json


def load_image_metadata(folder, expected_rows):
    metadata_path = folder / 'metadata.jsonl'
    with metadata_path.open(encoding='utf-8') as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if len(rows) != len(expected_rows):
        raise RuntimeError(f'PNG metadata count mismatch: {metadata_path}')
    for index, (row, expected) in enumerate(zip(rows, expected_rows)):
        if row.get('file_name') != image_name(index) or row.get('text') != expected['text']:
            raise RuntimeError(f'PNG metadata identity mismatch: {metadata_path} at row {index}')
        if not (folder / row['file_name']).is_file():
            raise RuntimeError(f'Missing generated PNG: {folder / row["file_name"]}')
    return rows


def valid_score_rows(score_rows, metadata):
    if not isinstance(score_rows, list) or len(score_rows) != len(metadata):
        return False
    for score, image in zip(score_rows, metadata):
        if not isinstance(score, dict):
            return False
        coordinates = score.get('coordinates')
        if (score.get('file') != image['file_name'] or not isinstance(coordinates, list)
                or len(coordinates) != 32):
            return False
        if not np.isfinite(np.asarray(coordinates, dtype=np.float64)).all():
            return False
    return True


def load_score_checkpoint(path, identity, metadata):
    if not path.is_file():
        return []
    try:
        checkpoint = read_json(path)
        rows = checkpoint.get('rows', [])
        if checkpoint.get('identity') != identity or not isinstance(rows, list):
            return []
        if not valid_score_rows(rows, metadata[:len(rows)]):
            return []
        return rows
    except (OSError, ValueError, TypeError):
        return []


def scoring_identity(folder, label, metadata):
    files = []
    for row in metadata:
        stat = (folder / row['file_name']).stat()
        files.append([row['file_name'], stat.st_size, stat.st_mtime_ns])
    generation_path = folder / 'generation.json'
    generation = read_json(generation_path) if generation_path.is_file() else None
    return {'label': label, 'metadata': metadata, 'generation': generation, 'files': files}


def full_key_metrics(coordinates, signs, aggregate):
    """Delegate FKR definition to the unmodified original key_metrics module."""
    return aggregate(coordinates, signs, sizes=(1, 8, 32, 64, 256), repeats=100, seed=20261029)


def score_outputs(paths, sources):
    if not torch.cuda.is_available():
        raise RuntimeError('Official PNG scoring needs server CUDA; no local model execution is performed')
    source, output = Path(paths['source_run']), Path(paths['output_run'])
    teacher = sources.semantic_teacher
    carrier = sources.carrier(source / 'carrier', 'cuda')
    extractor = sources.extractor(Path(paths['teacher_clean']), device='cuda')
    sources.verify_frozen(extractor)
    results, score_hashes = {}, {}
    try:
        for split in ('train', 'validation', 'test'):
            for kind in ('clean', 'ogsc'):
                label = f'teacher_{kind}_{split}'
                folder = output / 'images' / label
                expected_rows = _rows(paths, split)
                rows = load_image_metadata(folder, expected_rows)
                score_path = output / 'scores' / f'{label}.json'
                identity = scoring_identity(folder, label, rows)
                identity_path = score_path.with_suffix('.identity.json')
                checkpoint_path = output / 'scores/checkpoints' / f'{label}.json'
                score_rows = None
                if score_path.is_file():
                    try:
                        cached_rows = read_json(score_path)
                        identity_matches = (not identity_path.is_file()
                                            or read_json(identity_path) == identity)
                        if identity_matches and valid_score_rows(cached_rows, rows):
                            score_rows = cached_rows
                            if not identity_path.is_file():
                                write_json(identity_path, identity)
                            print(f'OGSC {label}: reusing completed score file ({len(rows)} images).', flush=True)
                    except (OSError, ValueError, TypeError):
                        score_rows = None
                if score_rows is None:
                    score_rows = load_score_checkpoint(checkpoint_path, identity, rows)
                    if score_rows:
                        print(f'OGSC {label} PNG scoring: resuming from {len(score_rows)}/{len(rows)} checkpointed images.', flush=True)
                    else:
                        score_rows = []
                        print(f'OGSC {label} PNG scoring: {len(rows)} images in batches of 4.', flush=True)
                    score_started = time.perf_counter()
                    last_checkpoint = len(score_rows)
                    for start in range(len(score_rows), len(rows), 4):
                        batch = rows[start:start + 4]
                        images = teacher.pixels([folder / row['file_name'] for row in batch], device='cuda')
                        prompts = [row['text'] for row in batch]
                        with torch.no_grad():
                            q = carrier.coordinates(extractor.encode_image(images, prompts))
                        q_numpy = q.detach().cpu().double().numpy()
                        score_rows.extend({'file': row['file_name'], 'coordinates': coords.tolist()}
                                           for row, coords in zip(batch, q_numpy))
                        completed = len(score_rows)
                        if completed - last_checkpoint >= 100 or completed == len(rows):
                            write_json(checkpoint_path, {'identity': identity, 'rows': score_rows})
                            last_checkpoint = completed
                        batch_i = start // 4 + 1
                        if batch_i == 1 or batch_i % 25 == 0 or completed == len(rows):
                            print(progress_line(f'{label} PNG scoring', completed, len(rows),
                                                time.perf_counter() - score_started), flush=True)
                    write_json(score_path, score_rows)
                    write_json(identity_path, identity)
                    if checkpoint_path.is_file():
                        checkpoint_path.unlink()
                q_values = [row['coordinates'] for row in score_rows]
                coordinates = np.asarray(q_values, dtype=np.float64)
                signs = carrier.signs.detach().cpu().numpy()
                with stage_progress(f'{label} official FKR aggregation'):
                    metrics = full_key_metrics(coordinates, signs, sources.key_metrics.aggregate_metrics)
                results[label] = {'n': len(rows), 'official_key_metrics': metrics,
                                  'scores_sha256': sha256(score_path),
                                  'coordinate_source': 'saved RGB8 PNGs scored with original semantic carrier'}
                score_hashes[label] = sha256(score_path)
                print(f'OGSC official PNG scoring: {label} complete ({len(rows)} images).', flush=True)
    finally:
        del carrier, extractor
        torch.cuda.empty_cache()
    payload = {'metric_source': str(Path(paths['weight32_repo']) / 'BEST_TEACHER_32bits_refine' /
                                    'server21/files/var/lib/structure32_student12500/key_metrics.py'),
               'metric_source_sha256': sha256(Path(paths['weight32_repo']) / 'BEST_TEACHER_32bits_refine' /
                                    'server21/files/var/lib/structure32_student12500/key_metrics.py'),
               'group_sizes': [1, 8, 32, 64, 256], 'repeats': 100, 'seed': 20261029,
               'results': results}
    write_json(output / 'result/teacher_key_metrics.json', payload)
    quality_file = output / 'calibration/gamma.json'
    if quality_file.is_file():
        quality = read_json(quality_file)
        write_json(output / 'result/teacher_quality.json', {
            'selected_gamma': quality.get('gamma'), 'limits': quality.get('limits'),
            'validation_trials': quality.get('trials'), 'quality_source': 'paired validation PNGs',
            'test_used_for_selection': False, 'fkr_used_for_selection': False})
    utility_status = 'torch-fidelity unavailable; FID/KID skipped'
    try:
        import torch_fidelity  # noqa: F401
        harness = Path(paths['weight32_repo']) / 'BEST_TEACHER_32bits_refine/harness'
        import sys
        sys.path.insert(0, str(harness))
        import utility
        with stage_progress('validation CLIP/FID/KID utility'):
            utility.main(output, 'validation', labels=['teacher_ogsc_validation'])
        with stage_progress('test CLIP/FID/KID utility'):
            utility.main(output, 'test', labels=['teacher_ogsc_test'])
        utility_status = 'original weight32 utility computed paired CLIP/FID/KID'
    except ImportError:
        pass
    return {'labels': sorted(results), 'score_hashes': score_hashes,
            'metrics_sha256': sha256(output / 'result/teacher_key_metrics.json'),
            'quality_utility': utility_status}
