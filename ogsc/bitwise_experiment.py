"""Fixed-gamma validation comparison for the independent bitwise objective."""

import json
import math
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .control import OBJECTIVE, one_step
from .evaluate import full_key_metrics, load_score_checkpoint, valid_score_rows
from .generation import (_decode, _rows, _save_png, generation_progress_message,
                         image_name, write_image_metadata)
from .progress import progress_line, stage_progress
from .state import atomic_path, read_json, write_json

FIXED_GAMMA = 0.156398773
BASELINE_LABELS = ('teacher_clean_validation', 'teacher_ogsc_validation')
NEW_LABEL = 'teacher_bitwise_validation'


def _require_pngs(folder, rows):
    folder = Path(folder)
    metadata = folder / 'metadata.jsonl'
    if metadata.is_file():
        with metadata.open(encoding='utf-8') as stream:
            image_rows = [json.loads(line) for line in stream if line.strip()]
        if len(image_rows) != len(rows):
            raise RuntimeError(f'PNG metadata count mismatch: {metadata}')
    else:
        image_rows = [{'file_name': image_name(i), 'text': row['text']}
                      for i, row in enumerate(rows)]
    started = time.perf_counter()
    for i, (image_row, expected) in enumerate(zip(image_rows, rows), start=1):
        if (image_row.get('file_name') != image_name(i - 1)
                or image_row.get('text') != expected['text']):
            raise RuntimeError(f'Validation PNG identity mismatch: {folder} at sample {i - 1}')
        if not (folder / image_row['file_name']).is_file():
            raise RuntimeError(f'Missing validation PNG: {folder / image_row["file_name"]}')
        if i == 1 or i % 100 == 0 or i == len(rows):
            print(progress_line(f'check {folder.name} PNGs', i, len(rows),
                                time.perf_counter() - started), flush=True)
    return image_rows


def _scoring_identity(folder, label, metadata):
    files = []
    started = time.perf_counter()
    for i, row in enumerate(metadata, start=1):
        stat = (folder / row['file_name']).stat()
        files.append([row['file_name'], stat.st_size, stat.st_mtime_ns])
        if i == 1 or i % 100 == 0 or i == len(metadata):
            print(progress_line(f'{label} score-cache identity', i, len(metadata),
                                time.perf_counter() - started), flush=True)
    generation_path = folder / 'generation.json'
    generation = read_json(generation_path) if generation_path.is_file() else None
    return {'label': label, 'metadata': metadata, 'generation': generation, 'files': files}


def _read_clean_latents(baseline, output, rows, seeds, paths):
    candidate_dirs = [
        baseline / 'state/latents/validation',
        baseline / 'calibration/validation/latents',
        output / 'state/bitwise_validation/clean_latents',
    ]
    records = [None] * len(rows)
    missing = []
    started = time.perf_counter()
    print(f'OGSC clean validation latent cache: checking {len(rows)} samples.', flush=True)
    for i, (row, seed) in enumerate(zip(rows, seeds)):
        record_path = next((folder / f'{i:04d}.pt' for folder in candidate_dirs
                            if (folder / f'{i:04d}.pt').is_file()), None)
        if record_path is None:
            missing.append(i)
        else:
            record = torch.load(record_path, map_location='cpu', weights_only=True)
            latent = record.get('latent')
            if (record.get('image_id') != row['image_id'] or record.get('text') != row['text']
                    or record.get('seed') != seed or not isinstance(latent, torch.Tensor)
                    or not torch.isfinite(latent).all()):
                raise RuntimeError(f'Clean validation latent identity mismatch at sample {i}')
            records[i] = latent
        completed = i + 1
        if completed == 1 or completed % 100 == 0 or completed == len(rows):
            print(progress_line('clean validation latent cache', completed, len(rows),
                                time.perf_counter() - started), flush=True)

    if missing:
        print(f'OGSC clean validation latent cache: {len(rows) - len(missing)} cached; '
              f'{len(missing)} missing; sampling only missing clean latents.', flush=True)
        teacher = _semantic_teacher(paths)
        pipe = teacher.load_native(Path(paths['teacher_clean']))
        cache_dir = output / 'state/bitwise_validation/clean_latents'
        sampling_started = time.perf_counter()
        try:
            for work_i, i in enumerate(missing, start=1):
                row, seed = rows[i], seeds[i]
                generator = torch.Generator('cuda').manual_seed(seed)
                with torch.inference_mode():
                    latent = pipe([row['text']], generator=[generator], num_inference_steps=50,
                                  guidance_scale=7.5, height=512, width=512,
                                  output_type='latent').images.detach()
                record = {'latent': latent.cpu(), 'image_id': row['image_id'],
                          'text': row['text'], 'seed': seed}
                with atomic_path(cache_dir / f'{i:04d}.pt') as temporary:
                    torch.save(record, temporary)
                records[i] = record['latent']
                if work_i == 1 or work_i % 25 == 0 or work_i == len(missing):
                    print(generation_progress_message(
                        'clean validation latent recovery', len(rows) - len(missing) + work_i,
                        len(rows), time.perf_counter() - sampling_started,
                        work_completed=work_i, work_total=len(missing)), flush=True)
        finally:
            del pipe
            torch.cuda.empty_cache()
    else:
        print('OGSC clean validation latent cache: all clean latents reused; no clean generation.', flush=True)
    return records


def _semantic_teacher(paths):
    import sys

    harness = Path(paths['weight32_repo']) / 'BEST_TEACHER_32bits_refine/harness'
    sys.path.insert(0, str(harness))
    import semantic_teacher
    return semantic_teacher


def _validate_bitwise_record(record, row, seed):
    latent = record.get('latent')
    if (record.get('objective') != OBJECTIVE or record.get('image_id') != row['image_id']
            or record.get('text') != row['text'] or record.get('seed') != seed
            or not math.isclose(float(record.get('gamma', float('nan'))), FIXED_GAMMA,
                                rel_tol=0, abs_tol=1e-12)
            or not isinstance(latent, torch.Tensor) or not torch.isfinite(latent).all()):
        raise RuntimeError('Cached bitwise validation latent identity/objective mismatch')
    return record


def _generate_bitwise(paths, sources, baseline, rows, seeds, clean_latents):
    output = Path(paths['output_run'])
    if output.resolve() == baseline.resolve():
        raise RuntimeError('Bitwise validation must use a separate --output-run so the old run stays unchanged')
    folder = output / 'images' / NEW_LABEL
    folder.mkdir(parents=True, exist_ok=True)
    clean_folder = baseline / 'images/teacher_clean_validation'
    old_folder = baseline / 'images/teacher_ogsc_validation'
    _require_pngs(clean_folder, rows)
    _require_pngs(old_folder, rows)

    source_hash = __import__('hashlib').sha256(
        (Path(paths['source_run']) / 'prompts.json').read_bytes()).hexdigest()
    for label, reference_folder in (('clean', clean_folder), ('soft-min', old_folder)):
        reference_generation = reference_folder / 'generation.json'
        if not reference_generation.is_file():
            raise RuntimeError(f'Missing {label} validation generation metadata: {reference_generation}')
        reference = read_json(reference_generation)
        if (reference.get('split') != 'validation'
                or reference.get('prompts_sha256') != source_hash):
            raise RuntimeError(f'Existing {label} validation PNGs do not match the current validation prompts')

    old_gamma_path = baseline / 'calibration/gamma.json'
    if not old_gamma_path.is_file():
        raise RuntimeError(f'Missing prior soft-min gamma record: {old_gamma_path}')
    old_gamma = read_json(old_gamma_path)
    if (old_gamma.get('status') != 'selected' or old_gamma.get('gamma') is None
            or not math.isclose(float(old_gamma['gamma']), FIXED_GAMMA, rel_tol=0, abs_tol=1e-9)):
        raise RuntimeError('Old soft-min validation images were generated with a different gamma; '
                           'cannot make the requested controlled comparison')
    baseline_provenance = baseline / 'provenance.json'
    protocol = read_json(baseline_provenance).get('protocol', '').lower() if baseline_provenance.is_file() else ''
    if 'soft-min' not in protocol:
        raise RuntimeError('The baseline run is not identified as the prior soft-min experiment')
    old_generation = read_json(old_folder / 'generation.json')
    if not math.isclose(float(old_generation.get('gamma', float('nan'))), FIXED_GAMMA,
                        rel_tol=0, abs_tol=1e-9):
        raise RuntimeError('The old soft-min PNG metadata uses a different gamma')

    generation = {'split': 'validation', 'objective': OBJECTIVE, 'gamma': FIXED_GAMMA,
                  'steps': 50, 'cfg': 7.5, 'resolution': [512, 512], 'dtype': 'float16',
                  'scheduler': 'original SD2.1 PNDM', 'seed_rule': 'config.seed*1000000+image_id',
                  'prompts_sha256': source_hash, 'baseline_run': str(baseline)}
    generation_path = folder / 'generation.json'
    if generation_path.is_file():
        prior = read_json(generation_path)
        for key in ('split', 'objective', 'gamma', 'prompts_sha256'):
            if prior.get(key) != generation[key]:
                raise RuntimeError(f'Existing bitwise validation output has incompatible {key}; '
                                   'choose a new --output-run')
    write_json(generation_path, generation)

    record_dir = output / 'state/bitwise_validation/marked_latents'
    record_dir.mkdir(parents=True, exist_ok=True)
    records = {}
    missing_control, missing_decode = [], []
    scan_started = time.perf_counter()
    for i, (row, seed) in enumerate(zip(rows, seeds)):
        record_path = record_dir / f'{i:04d}.pt'
        image_path = folder / image_name(i)
        if record_path.is_file():
            records[i] = _validate_bitwise_record(
                torch.load(record_path, map_location='cpu', weights_only=True), row, seed)
        elif image_path.is_file():
            raise RuntimeError(f'Bitwise validation PNG exists without its resume record: {image_path}')
        else:
            missing_control.append(i)
        if not image_path.is_file():
            missing_decode.append(i)
        completed = i + 1
        if completed == 1 or completed % 100 == 0 or completed == len(rows):
            print(progress_line('bitwise validation resume scan', completed, len(rows),
                                time.perf_counter() - scan_started), flush=True)

    print(f'OGSC bitwise validation control: {len(rows) - len(missing_control)} cached; '
          f'{len(missing_control)} remaining; one VJP per uncached image.', flush=True)
    if missing_control:
        scorer = sources.extractor(Path(paths['teacher_clean']), device='cuda')
        carrier = sources.carrier(Path(paths['source_run']) / 'carrier', 'cuda')
        sources.verify_frozen(scorer)
        started = time.perf_counter()
        try:
            for work_i, i in enumerate(missing_control, start=1):
                row, seed = rows[i], seeds[i]
                latent = clean_latents[i].to('cuda')
                torch.cuda.synchronize()
                control_started = time.perf_counter()
                marked, diagnostics = one_step(
                    latent, sources.signed_view(scorer, carrier, row['text']), FIXED_GAMMA)
                torch.cuda.synchronize()
                control_seconds = time.perf_counter() - control_started
                record = {'objective': OBJECTIVE, 'gamma': FIXED_GAMMA, 'latent': marked.cpu(),
                          'image_id': row['image_id'], 'text': row['text'], 'seed': seed,
                          'control_seconds': control_seconds,
                          'loss_before': diagnostics['loss_before']}
                with atomic_path(record_dir / f'{i:04d}.pt') as temporary:
                    torch.save(record, temporary)
                records[i] = record
                if work_i == 1 or work_i % 25 == 0 or work_i == len(missing_control):
                    print(generation_progress_message(
                        'bitwise validation control', len(rows) - len(missing_control) + work_i,
                        len(rows), time.perf_counter() - started,
                        work_completed=work_i, work_total=len(missing_control)), flush=True)
        finally:
            del scorer, carrier
            torch.cuda.empty_cache()

    print(f'OGSC bitwise validation PNG decode: {len(rows) - len(missing_decode)} cached; '
          f'{len(missing_decode)} remaining.', flush=True)
    if missing_decode:
        teacher = _semantic_teacher(paths)
        pipe = teacher.load_native(Path(paths['teacher_clean']))
        started = time.perf_counter()
        try:
            for work_i, i in enumerate(missing_decode, start=1):
                record = records.get(i)
                if record is None:
                    record = _validate_bitwise_record(
                        torch.load(record_dir / f'{i:04d}.pt', map_location='cpu', weights_only=True),
                        rows[i], seeds[i])
                _save_png(_decode(pipe, record['latent'].to('cuda'))[0],
                          folder / image_name(i))
                if work_i == 1 or work_i % 25 == 0 or work_i == len(missing_decode):
                    print(generation_progress_message(
                        'bitwise validation PNG decode', len(rows) - len(missing_decode) + work_i,
                        len(rows), time.perf_counter() - started,
                        work_completed=work_i, work_total=len(missing_decode)), flush=True)
        finally:
            del pipe
            torch.cuda.empty_cache()
    print(f'OGSC bitwise validation metadata: writing prompt mapping for {len(rows)} PNGs.', flush=True)
    write_image_metadata(folder, rows)
    return {'clean_folder': clean_folder, 'softmin_folder': old_folder,
            'bitwise_folder': folder, 'records': records, 'generation': generation}


def _score_validation_variants(paths, sources, folders, rows):
    baseline = Path(paths['baseline_run'])
    output = Path(paths['output_run'])
    prepared = {}
    pending = []
    for label, folder in folders.items():
        metadata = _require_pngs(folder, rows)
        identity = _scoring_identity(folder, label, metadata)
        output_score = output / 'scores' / f'{label}.json'
        output_identity = output_score.with_suffix('.identity.json')
        candidate_paths = [(baseline / 'scores' / f'{label}.json',
                            (baseline / 'scores' / f'{label}.identity.json')),
                           (output_score, output_identity)]
        cached = None
        for score_path, identity_path in candidate_paths:
            if not score_path.is_file() or not identity_path.is_file():
                continue
            try:
                score_rows = read_json(score_path)
                if read_json(identity_path) == identity and valid_score_rows(score_rows, metadata):
                    cached = score_rows
                    prepared_score_path = score_path
                    print(f'OGSC {label}: reusing completed PNG score ({len(rows)} images).', flush=True)
                    break
            except (OSError, ValueError, TypeError):
                pass
        prepared[label] = {'folder': folder, 'metadata': metadata, 'identity': identity,
                           'score_path': (prepared_score_path if cached is not None else output_score),
                           'score_rows': cached}
        if cached is None:
            checkpoint = output / 'scores/checkpoints' / f'{label}.json'
            partial = load_score_checkpoint(checkpoint, identity, metadata)
            prepared[label]['score_rows'] = partial
            prepared[label]['checkpoint'] = checkpoint
            pending.append(label)

    if pending:
        teacher = sources.semantic_teacher
        extractor = sources.extractor(Path(paths['teacher_clean']), device='cuda')
        carrier = sources.carrier(Path(paths['source_run']) / 'carrier', 'cuda')
        sources.verify_frozen(extractor)
        try:
            for label in pending:
                entry = prepared[label]
                rows_done = entry['score_rows']
                folder, metadata, identity = entry['folder'], entry['metadata'], entry['identity']
                checkpoint, score_path = entry['checkpoint'], entry['score_path']
                start_time = time.perf_counter()
                print(f'OGSC {label} PNG scoring: {len(rows_done)}/{len(rows)} cached; '
                      f'batches of 4.', flush=True)
                last_checkpoint = len(rows_done)
                for start in range(len(rows_done), len(rows), 4):
                    batch = metadata[start:start + 4]
                    images = teacher.pixels([folder / row['file_name'] for row in batch], device='cuda')
                    prompts = [row['text'] for row in batch]
                    with torch.no_grad():
                        coordinates = carrier.coordinates(extractor.encode_image(images, prompts))
                    values = coordinates.detach().cpu().double().numpy()
                    rows_done.extend({'file': row['file_name'], 'coordinates': value.tolist()}
                                     for row, value in zip(batch, values))
                    completed = len(rows_done)
                    if completed - last_checkpoint >= 100 or completed == len(rows):
                        write_json(checkpoint, {'identity': identity, 'rows': rows_done})
                        last_checkpoint = completed
                    batch_number = start // 4 + 1
                    if batch_number == 1 or batch_number % 25 == 0 or completed == len(rows):
                        print(progress_line(f'{label} PNG scoring', completed, len(rows),
                                            time.perf_counter() - start_time), flush=True)
                write_json(score_path, rows_done)
                write_json(score_path.with_suffix('.identity.json'), identity)
                if checkpoint.is_file():
                    checkpoint.unlink()
                entry['score_rows'] = rows_done
        finally:
            del extractor, carrier
            torch.cuda.empty_cache()

    results = {}
    signs = None
    for label, entry in prepared.items():
        coordinates = np.asarray([row['coordinates'] for row in entry['score_rows']], dtype=np.float64)
        if signs is None:
            signs = sources.carrier(Path(paths['source_run']) / 'carrier', 'cpu').signs.numpy()
        with stage_progress(f'{label} official exact-32 FKR aggregation'):
            metrics = full_key_metrics(coordinates, signs, sources.key_metrics.aggregate_metrics)
        signed = coordinates * signs.reshape(1, 32)
        mean_margins = signed.mean(axis=0)
        results[label] = {
            'n': len(entry['score_rows']), 'official_key_metrics': metrics,
            'mean_signed_margins': mean_margins.tolist(),
            'per_bit_accuracy': (signed > 0).mean(axis=0).tolist(),
            'negative_mean_margin_bits_1_indexed': [int(i + 1) for i in np.where(mean_margins < 0)[0]],
            'score_file': str(entry['score_path']),
        }
    return results


def _score_independent_clean_controls(paths, sources):
    """Score the held-out clean test PNGs to measure exact-match FPR independently."""
    baseline = Path(paths['baseline_run'])
    output = Path(paths['output_run'])
    folder = baseline / 'images/teacher_clean_test'
    rows = _rows(paths, 'test')
    if len(rows) != 1000:
        raise RuntimeError(f'Independent clean-control FPR expects 1,000 test prompts; found {len(rows)}')
    metadata = _require_pngs(folder, rows)
    label = 'teacher_clean_test'
    identity = _scoring_identity(folder, label, metadata)
    baseline_score = baseline / 'scores' / f'{label}.json'
    baseline_identity = baseline_score.with_suffix('.identity.json')
    if baseline_score.is_file() and baseline_identity.is_file():
        try:
            score_rows = read_json(baseline_score)
            if read_json(baseline_identity) == identity and valid_score_rows(score_rows, metadata):
                print(f'OGSC {label}: reusing completed independent clean-control scores '
                      f'({len(rows)} images).', flush=True)
                source_score = baseline_score
            else:
                score_rows = None
        except (OSError, ValueError, TypeError):
            score_rows = None
    else:
        score_rows = None
    if score_rows is None:
        score_path = output / 'scores' / f'{label}.json'
        identity_path = score_path.with_suffix('.identity.json')
        if score_path.is_file() and identity_path.is_file():
            try:
                cached = read_json(score_path)
                if read_json(identity_path) == identity and valid_score_rows(cached, metadata):
                    score_rows = cached
                    source_score = score_path
                    print(f'OGSC {label}: reusing resumed independent clean-control scores '
                          f'({len(rows)} images).', flush=True)
            except (OSError, ValueError, TypeError):
                score_rows = None
        if score_rows is None:
            checkpoint = output / 'scores/checkpoints' / f'{label}.json'
            score_rows = load_score_checkpoint(checkpoint, identity, metadata)
            print(f'OGSC {label} PNG scoring: {len(score_rows)}/{len(rows)} cached; '
                  'scoring independent clean controls.', flush=True)
            teacher = sources.semantic_teacher
            extractor = sources.extractor(Path(paths['teacher_clean']), device='cuda')
            carrier = sources.carrier(Path(paths['source_run']) / 'carrier', 'cuda')
            sources.verify_frozen(extractor)
            started = time.perf_counter()
            last_reported = len(score_rows)
            last_checkpoint = len(score_rows)
            try:
                for start in range(len(score_rows), len(rows), 4):
                    batch = metadata[start:start + 4]
                    images = teacher.pixels([folder / row['file_name'] for row in batch], device='cuda')
                    with torch.no_grad():
                        coordinates = carrier.coordinates(
                            extractor.encode_image(images, [row['text'] for row in batch]))
                    values = coordinates.detach().cpu().double().numpy()
                    score_rows.extend({'file': row['file_name'], 'coordinates': value.tolist()}
                                     for row, value in zip(batch, values))
                    completed = len(score_rows)
                    if completed - last_checkpoint >= 100 or completed == len(rows):
                        write_json(checkpoint, {'identity': identity, 'rows': score_rows})
                        last_checkpoint = completed
                    if completed - last_reported >= 100 or completed == len(rows):
                        print(progress_line(f'{label} PNG scoring', completed, len(rows),
                                            time.perf_counter() - started), flush=True)
                        last_reported = completed
                write_json(score_path, score_rows)
                write_json(identity_path, identity)
                if checkpoint.is_file():
                    checkpoint.unlink()
                source_score = score_path
            finally:
                del extractor, carrier
                torch.cuda.empty_cache()
    coordinates = np.asarray([row['coordinates'] for row in score_rows], dtype=np.float64)
    signs = sources.carrier(Path(paths['source_run']) / 'carrier', 'cpu').signs.numpy()
    with stage_progress('independent clean test exact-32 FKR aggregation'):
        metrics = full_key_metrics(coordinates, signs, sources.key_metrics.aggregate_metrics)
    return {'n': len(rows), 'official_key_metrics': metrics,
            'score_file': str(source_score), 'split': 'test'}


def _quality_identity(folders, rows, gamma):
    identity = {'objective': OBJECTIVE, 'gamma': gamma, 'n': len(rows), 'files': {}}
    started = time.perf_counter()
    total = len(rows) * len(folders)
    completed = 0
    for label, folder in folders.items():
        identity['files'][label] = []
        for i in range(len(rows)):
            path = folder / image_name(i)
            stat = path.stat()
            identity['files'][label].append([path.name, stat.st_size, stat.st_mtime_ns])
            completed += 1
            if completed == 1 or completed % 100 == 0 or completed == total:
                print(progress_line('validation quality-cache identity', completed, total,
                                    time.perf_counter() - started), flush=True)
    return identity


def _paired_quality(paths, folders, rows, teacher, gamma):
    output = Path(paths['output_run'])
    result_path = output / 'result/bitwise_validation_quality.json'
    identity = _quality_identity(folders, rows, gamma)
    if result_path.is_file():
        existing = read_json(result_path)
        if existing.get('identity') == identity:
            print(f'OGSC validation CLIP/LPIPS: reusing completed quality file ({len(rows)} pairs).', flush=True)
            return existing

    partial_path = result_path.with_suffix('.partial.json')
    partial = None
    if partial_path.is_file():
        try:
            candidate = read_json(partial_path)
            if candidate.get('identity') == identity:
                candidate_clean = candidate.get('clip_clean', [])
                candidate_values = candidate.get('variants', {})
                if (isinstance(candidate_clean, list)
                        and all(len(candidate_values.get(label, {}).get(key, [])) == len(candidate_clean)
                                for label in ('teacher_ogsc_validation', NEW_LABEL)
                                for key in ('clip', 'lpips'))
                        and len(candidate_clean) <= len(rows)):
                    partial = candidate
        except (OSError, ValueError, TypeError):
            partial = None

    clips_clean = partial.get('clip_clean', []) if partial else []
    values = partial.get('variants', {}) if partial else {}
    values = {label: {'clip': values.get(label, {}).get('clip', []),
                      'lpips': values.get(label, {}).get('lpips', [])}
              for label in ('teacher_ogsc_validation', NEW_LABEL)}
    resumed = len(clips_clean)
    if resumed:
        print(f'OGSC validation CLIP/LPIPS: resuming from {resumed}/{len(rows)} completed pairs.', flush=True)

    harness = Path(paths['weight32_repo']) / 'BEST_TEACHER_32bits_refine/harness'
    import sys
    sys.path.insert(0, str(harness))
    from utility import CLIP_REV
    from transformers import CLIPModel, CLIPProcessor
    import lpips

    device = torch.device('cpu' if paths.get('offload') else 'cuda')
    labels = ('teacher_ogsc_validation', NEW_LABEL)
    clean_files = [folders['teacher_clean_validation'] / image_name(i) for i in range(len(rows))]
    started = time.perf_counter()
    clip = processor = perceptual = None
    if resumed < len(rows):
        with stage_progress('load CLIP and LPIPS quality models'):
            clip = CLIPModel.from_pretrained('openai/clip-vit-base-patch32', revision=CLIP_REV).to(device).eval().requires_grad_(False)
            processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32', revision=CLIP_REV)
            perceptual = lpips.LPIPS(net='alex', version='0.1').to(device).eval().requires_grad_(False)
    try:
        batch_size = 8
        last_reported = resumed
        last_checkpoint = resumed
        for start in range(resumed, len(rows), batch_size):
            end = min(start + batch_size, len(rows))
            batch_rows = rows[start:end]
            clean_batch = clean_files[start:end]
            clean_images = []
            for path in clean_batch:
                with Image.open(path) as image:
                    clean_images.append(image.convert('RGB'))
            clean_pixels = teacher.pixels(clean_batch, device=str(device))
            clean_inputs = processor(text=[row['text'] for row in batch_rows], images=clean_images,
                                     return_tensors='pt', padding=True, truncation=True).to(device)
            with torch.inference_mode():
                clean_output = clip(**clean_inputs)
                clips_clean.extend((clean_output.image_embeds * clean_output.text_embeds)
                                   .sum(-1).clamp_min(0).mul(2.5).cpu().tolist())
            for label in labels:
                paths_batch = [folders[label] / image_name(i) for i in range(start, end)]
                images = []
                for path in paths_batch:
                    with Image.open(path) as image:
                        images.append(image.convert('RGB'))
                inputs = processor(text=[row['text'] for row in batch_rows], images=images,
                                   return_tensors='pt', padding=True, truncation=True).to(device)
                marked_pixels = teacher.pixels(paths_batch, device=str(device))
                with torch.inference_mode():
                    clip_output = clip(**inputs)
                    clip_scores = (clip_output.image_embeds * clip_output.text_embeds)
                    clip_scores = clip_scores.sum(-1).clamp_min(0).mul(2.5)
                    distances = perceptual(clean_pixels, marked_pixels).flatten()
                values[label]['clip'].extend(clip_scores.cpu().tolist())
                values[label]['lpips'].extend(distances.cpu().tolist())
            completed = end
            if completed == len(rows) or completed - last_reported >= 100:
                print(progress_line('validation paired CLIP/LPIPS', completed, len(rows),
                                    time.perf_counter() - started), flush=True)
                last_reported = completed
            if completed == len(rows) or completed - last_checkpoint >= 100:
                write_json(partial_path, {'identity': identity, 'clip_clean': clips_clean,
                                          'variants': values})
                last_checkpoint = completed
    finally:
        del clip, processor, perceptual
        torch.cuda.empty_cache()

    lpips_limit = .15
    clip_drop_limit = .01
    report = {'identity': identity, 'n': len(rows), 'clip_revision': CLIP_REV,
              'clip_definition': 'mean 2.5 * max(cos(image,text),0), original utility convention',
              'lpips': 'alex, version=0.1', 'limits': {'mean_paired_lpips': lpips_limit,
                                                       'mean_clip_drop': clip_drop_limit},
              'clean_clip': float(np.mean(clips_clean)), 'variants': {}}
    for label in labels:
        clip_values = values[label]['clip']
        lpips_values = values[label]['lpips']
        clip_score = float(np.mean(clip_values))
        mean_lpips = float(np.mean(lpips_values))
        report['variants'][label] = {
            'clip_score': clip_score,
            'clip_drop': report['clean_clip'] - clip_score,
            'mean_paired_lpips': mean_lpips,
            'median_paired_lpips': float(np.median(lpips_values)),
            'passes_existing_quality_limits': (mean_lpips <= lpips_limit
                                               and report['clean_clip'] - clip_score <= clip_drop_limit),
            'clip_per_image': clip_values,
            'lpips_per_image': lpips_values,
        }
    write_json(result_path, report)
    if partial_path.is_file():
        partial_path.unlink()
    return report


def _group_rows(result):
    return result['official_key_metrics']['rows']


def _report(paths, metrics, independent_controls, quality, records, gamma):
    output = Path(paths['output_run'])
    clean = metrics['teacher_clean_validation']
    softmin = metrics['teacher_ogsc_validation']
    bitwise = metrics[NEW_LABEL]
    clean_fpr = next(row['full_key_recovery_rate'] for row in _group_rows(independent_controls)
                     if row['images_per_group'] == 1)
    latencies = [float(record['control_seconds']) for record in records.values()
                 if record.get('control_seconds') is not None]
    latency = None
    if latencies:
        latency = {'n_measured': len(latencies), 'mean_seconds': statistics.mean(latencies),
                   'median_seconds': statistics.median(latencies),
                   'p95_seconds': float(np.percentile(latencies, 95))}
    comparison = {}
    for label, record in metrics.items():
        q = quality['variants'].get(label)
        comparison[label] = {
            'official_key_metrics': record['official_key_metrics'],
            'mean_signed_margins': record['mean_signed_margins'],
            'per_bit_accuracy': record['per_bit_accuracy'],
            'negative_mean_margin_bits_1_indexed': record['negative_mean_margin_bits_1_indexed'],
            'quality': q,
        }
    payload = {
        'protocol': 'OGSC-32 controlled validation comparison', 'split': 'validation',
        'n': bitwise['n'], 'objective': OBJECTIVE, 'gamma': gamma,
        'full_key_rule': 'exact 32/32 signs; original key_metrics.aggregate_metrics',
        'independent_clean_controls': independent_controls,
        'clean_empirical_fpr_exact_32_of_32': clean_fpr,
        'inference_control_overhead_seconds': latency,
        'comparison': comparison,
        'quality_limits': quality['limits'],
        'student_training_run': False,
    }
    write_json(output / 'result/bitwise_validation_comparison.json', payload)

    lines = ['# OGSC-32: soft-min vs. bitwise softplus', '',
             f'- Validation images: {bitwise["n"]}', f'- Bitwise gamma: `{gamma}`',
             f'- Bitwise objective: `{OBJECTIVE}`',
             '- The clean latents and original clean/soft-min PNGs are reused from the prior run.',
             '- Full-key recovery uses the original exact 32/32 evaluator; no 24/32 detector is used.',
             '- Empirical FPR uses the separate clean test split (not used for gamma selection).',
             f'- Clean test FPR (exact 32/32, group size 1): `{clean_fpr:.6f}`.', '',
             '| Variant | Bit accuracy @1 | FKR @1 | FKR @8 | FKR @32 | FKR @64 | FKR @256 | Mean LPIPS | CLIP drop |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    labels = [('Clean validation', clean), ('Old soft-min', softmin), ('Bitwise softplus', bitwise)]
    for name, entry in labels:
        rows = _group_rows(entry)
        by_size = {row['images_per_group']: row for row in rows}
        lpips_value = quality['variants'].get('teacher_ogsc_validation' if name == 'Old soft-min'
                                               else NEW_LABEL, {}).get('mean_paired_lpips')
        clip_drop = quality['variants'].get('teacher_ogsc_validation' if name == 'Old soft-min'
                                             else NEW_LABEL, {}).get('clip_drop')
        if name == 'Clean validation':
            lpips_value, clip_drop = None, None
        render = lambda value: '—' if value is None else f'{value:.5f}'
        lines.append(f"| {name} | {by_size[1]['bit_accuracy']:.5f} | "
                     f"{by_size[1]['full_key_recovery_rate']:.5f} | "
                     f"{by_size[8]['full_key_recovery_rate']:.5f} | "
                     f"{by_size[32]['full_key_recovery_rate']:.5f} | "
                     f"{by_size[64]['full_key_recovery_rate']:.5f} | "
                     f"{by_size[256]['full_key_recovery_rate']:.5f} | "
                     f"{render(lpips_value)} | {render(clip_drop)} |")
    lines += ['', '## Bitwise mean signed margins', '',
              '| Bit | Clean | Old soft-min | Bitwise softplus | Bitwise bit accuracy |',
              '|---:|---:|---:|---:|---:|']
    for bit in range(32):
        lines.append(f"| {bit + 1} | {clean['mean_signed_margins'][bit]:.6f} | "
                     f"{softmin['mean_signed_margins'][bit]:.6f} | "
                     f"{bitwise['mean_signed_margins'][bit]:.6f} | "
                     f"{bitwise['per_bit_accuracy'][bit]:.5f} |")
    bad = bitwise['negative_mean_margin_bits_1_indexed']
    lines += ['', f'- Bitwise bits with negative mean signed margin: `{bad}`.',
              f'- Measured bitwise semantic-control latency: `{latency}` seconds summary.',
              f'- Bitwise image quality within existing mean limits: '
              f"`{quality['variants'][NEW_LABEL]['passes_existing_quality_limits']}`.",
              '- This command evaluates validation only and does not start student distillation.', '']
    summary_path = output / 'result/bitwise_validation_comparison.md'
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f'OGSC bitwise validation report: {summary_path}', flush=True)
    return payload


def run_bitwise_validation(paths, sources, gamma=FIXED_GAMMA):
    if not torch.cuda.is_available():
        raise RuntimeError('Bitwise validation needs server CUDA; local checks use CPU/tiny model only')
    if not math.isclose(float(gamma), FIXED_GAMMA, rel_tol=0, abs_tol=1e-12):
        raise ValueError(f'The controlled first comparison uses fixed gamma={FIXED_GAMMA}')
    source = Path(paths['source_run'])
    baseline = Path(paths['baseline_run'])
    output = Path(paths['output_run'])
    rows = _rows(paths, 'validation')
    if len(rows) != 1000:
        raise RuntimeError(f'The requested controlled experiment expects 1,000 validation prompts; found {len(rows)}')
    config = read_json(source / 'config.json')
    seeds = [int(config['seed']) * 1_000_000 + int(row['image_id']) for row in rows]

    with stage_progress('prepare isolated bitwise validation output'):
        output.mkdir(parents=True, exist_ok=True)
        (output / 'result').mkdir(parents=True, exist_ok=True)
        (output / 'state').mkdir(parents=True, exist_ok=True)
        write_json(output / 'calibration/bitwise_validation.json', {
            'status': 'running', 'objective': OBJECTIVE, 'gamma': FIXED_GAMMA,
            'split': 'validation', 'count': len(rows), 'baseline_run': str(baseline),
        })

    with stage_progress('reuse clean latents and validate clean/soft-min reference PNGs'):
        clean_latents = _read_clean_latents(baseline, output, rows, seeds, paths)

    with stage_progress('generate bitwise softplus validation PNGs'):
        folders = _generate_bitwise(paths, sources, baseline, rows, seeds, clean_latents)

    with stage_progress('score clean, old soft-min, and bitwise PNGs with original FKR evaluator'):
        variants = {
            'teacher_clean_validation': folders['clean_folder'],
            'teacher_ogsc_validation': folders['softmin_folder'],
            NEW_LABEL: folders['bitwise_folder'],
        }
        metrics = _score_validation_variants(paths, sources, variants, rows)

    with stage_progress('score independent clean test controls for empirical FPR'):
        independent_controls = _score_independent_clean_controls(paths, sources)

    with stage_progress('evaluate paired validation CLIP and LPIPS'):
        quality = _paired_quality(paths, variants, rows, sources.semantic_teacher,
                                  FIXED_GAMMA)

    with stage_progress('write bitwise comparison report'):
        result = _report(paths, metrics, independent_controls, quality,
                         folders['records'], FIXED_GAMMA)
    calibration = read_json(output / 'calibration/bitwise_validation.json')
    calibration.update({'status': 'complete', 'result': str(output / 'result/bitwise_validation_comparison.md')})
    write_json(output / 'calibration/bitwise_validation.json', calibration)
    return {
        'status': 'complete',
        'summary': str(output / 'result/bitwise_validation_comparison.md'),
        'json': str(output / 'result/bitwise_validation_comparison.json'),
        'n': len(rows), 'objective': OBJECTIVE, 'gamma': FIXED_GAMMA,
        'clean_empirical_fpr_exact_32_of_32': result['clean_empirical_fpr_exact_32_of_32'],
        'inference_control_overhead_seconds': result['inference_control_overhead_seconds'],
        'bitwise_fkr': [row for row in metrics[NEW_LABEL]['official_key_metrics']['rows']],
    }
