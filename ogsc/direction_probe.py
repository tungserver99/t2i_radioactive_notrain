"""Small-sample diagnostic for the OGSC bitwise direction issue."""

import json
import math
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from .bitwise_experiment import (FIXED_GAMMA, NEW_LABEL, _scoring_identity,
                                 _validate_bitwise_record, _semantic_teacher)
from .control import OBJECTIVE, one_step
from .evaluate import valid_score_rows
from .generation import (_decode, _rows, _save_png, generation_progress_message,
                         image_name)
from .progress import progress_line
from .state import atomic_path, read_json, write_json

GAMMA_OLD = FIXED_GAMMA
GAMMA_SMALL = GAMMA_OLD / 16.0
PROBE_COUNT = 64
FAILED_BITS = (3, 5, 13, 14, 17, 31, 32)


def evenly_spaced_indices(total, count=PROBE_COUNT):
    """Choose `count` unique indices spanning the complete prompt split."""
    total, count = int(total), int(count)
    if total <= 0 or count <= 0 or count > total:
        raise ValueError('Expected 0 < probe count <= number of prompts')
    if count == 1:
        return [0]
    denominator = count - 1
    return [(position * (total - 1) + denominator // 2) // denominator
            for position in range(count)]


def interpolate_small_latent(clean, marked, gamma_old=GAMMA_OLD, gamma_small=GAMMA_SMALL):
    """Scale the cached FP16 old-gamma displacement down, preserving storage dtype."""
    if not isinstance(clean, torch.Tensor) or not isinstance(marked, torch.Tensor):
        raise TypeError('Clean and marked latents must be torch tensors')
    if clean.shape != marked.shape or clean.dtype not in (torch.float16, torch.float32, torch.bfloat16):
        raise ValueError('Clean and marked latents must have equal shapes and floating dtypes')
    if not math.isfinite(float(gamma_old)) or gamma_old <= 0:
        raise ValueError('gamma_old must be finite and positive')
    if not math.isfinite(float(gamma_small)) or gamma_small < 0:
        raise ValueError('gamma_small must be finite and nonnegative')
    direction = (marked.float() - clean.float()) / float(gamma_old)
    return (clean.float() + float(gamma_small) * direction).to(dtype=clean.dtype)


def classify_direction_evidence(failed_bits, surrogate_shifts, small_png_shifts, old_png_shifts):
    """Classify only when all listed failed bits match one spec-defined pattern."""
    bits = [int(bit) for bit in failed_bits]
    mismatch = [bit for bit in bits if surrogate_shifts[bit] > 0 and small_png_shifts[bit] < 0]
    conflict = [bit for bit in bits if surrogate_shifts[bit] <= 0 and small_png_shifts[bit] <= 0]
    overshoot = [bit for bit in bits if small_png_shifts[bit] > 0 and old_png_shifts[bit] < 0]
    patterns = {
        'surrogate_mismatch': mismatch,
        'gradient_conflict': conflict,
        'overshoot': overshoot,
    }
    complete = [name for name, supported in patterns.items() if bits and len(supported) == len(bits)]
    conclusion = complete[0] if len(complete) == 1 else 'inconclusive'
    support = len(patterns.get(conclusion, [])) if conclusion != 'inconclusive' else max(
        (len(values) for values in patterns.values()), default=0)
    return {
        'conclusion': conclusion,
        'supporting_bits': support,
        'failed_bits': bits,
        'evidence_counts': {name: len(values) for name, values in patterns.items()},
        'evidence_bits': patterns,
        'rule': ('A named cause is reported only if the same pattern holds for every currently failed bit; '
                 'otherwise the outcome is inconclusive.'),
    }


def _clean_latent(baseline, output, index, row, expected_seed):
    candidates = (
        baseline / 'state/latents/validation' / f'{index:04d}.pt',
        baseline / 'calibration/validation/latents' / f'{index:04d}.pt',
        output / 'state/bitwise_validation/clean_latents' / f'{index:04d}.pt',
    )
    for path in candidates:
        if not path.is_file():
            continue
        record = torch.load(path, map_location='cpu', weights_only=True)
        latent = record.get('latent')
        if (record.get('image_id') != row['image_id'] or record.get('text') != row['text']
                or record.get('seed') != expected_seed or not isinstance(latent, torch.Tensor)
                or not torch.isfinite(latent).all()):
            raise RuntimeError(f'Clean validation latent identity mismatch at sample {index}: {path}')
        return latent
    raise RuntimeError(f'No cached clean latent for selected validation sample {index}; '
                       'the direction probe will not resample diffusion latents')


def _read_full_scores(path, rows, label, image_folder):
    if not path.is_file():
        raise RuntimeError(f'Missing reusable {label} coordinate score file: {path}')
    score_rows = read_json(path)
    metadata = [{'file_name': image_name(i), 'text': row['text']} for i, row in enumerate(rows)]
    if not valid_score_rows(score_rows, metadata):
        raise RuntimeError(f'Reusable {label} score file is incomplete or does not match validation prompts: {path}')
    identity_path = path.with_suffix('.identity.json')
    if not identity_path.is_file():
        raise RuntimeError(f'Missing {label} score identity sidecar: {identity_path}')
    expected_identity = _scoring_identity(image_folder, label, metadata)
    if not _score_identity_matches(read_json(identity_path), expected_identity):
        raise RuntimeError(f'Reusable {label} score identity does not match its PNGs: {identity_path}')
    print(f'OGSC direction probe: reusing {label} coordinates for {len(rows)} prompts.', flush=True)
    return score_rows


def _score_identity_matches(stored, current):
    """Compare existing score identities without hashing image contents.

    Drive mounts can round mtime nanoseconds during synchronization, so allow
    one quantized-second difference (less than two seconds in exact time) while
    keeping names, sizes, prompts, labels, and generation metadata exact.
    """
    if any(stored.get(key) != current.get(key) for key in ('label', 'metadata', 'generation')):
        return False
    old_files, new_files = stored.get('files'), current.get('files')
    if not isinstance(old_files, list) or not isinstance(new_files, list) or len(old_files) != len(new_files):
        return False
    for old, new in zip(old_files, new_files):
        if not isinstance(old, list) or not isinstance(new, list) or len(old) != 3 or len(new) != 3:
            return False
        if old[:2] != new[:2]:
            return False
        try:
            if abs(int(old[2]) // 1_000_000_000 - int(new[2]) // 1_000_000_000) > 1:
                return False
        except (TypeError, ValueError, OverflowError):
            return False
    return True


def _failed_bits_from_baseline(output):
    path = Path(output) / 'result/bitwise_validation_comparison.json'
    if not path.is_file():
        raise RuntimeError(f'Missing completed bitwise comparison report: {path}')
    report = read_json(path)
    try:
        values = report['comparison'][NEW_LABEL]['negative_mean_margin_bits_1_indexed']
        bits = tuple(sorted({int(value) for value in values}))
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f'Invalid failed-bit list in baseline comparison report: {path}') from error
    if any(bit < 1 or bit > 32 for bit in bits):
        raise RuntimeError(f'Failed-bit list must use bit numbers 1..32: {path}')
    return bits


def _load_marked(output, index, row, seed):
    candidates = (
        output / 'state/bitwise_validation/marked_latents' / f'{index:04d}.pt',
        output / 'state/direction_probe/marked_latents' / f'{index:04d}.pt',
    )
    for path in candidates:
        if path.is_file():
            return _validate_bitwise_record(
                torch.load(path, map_location='cpu', weights_only=True), row, seed)
    return None


def _probe_cache_path(output, index):
    return output / 'state/direction_probe/samples' / f'{index:04d}.pt'


def _read_probe_cache(path, row, seed):
    if not path.is_file():
        return None
    record = torch.load(path, map_location='cpu', weights_only=True)
    latent = record.get('small_latent')
    if (record.get('objective') != OBJECTIVE or record.get('image_id') != row['image_id']
            or record.get('text') != row['text'] or record.get('seed') != seed
            or record.get('gamma_old') != GAMMA_OLD or record.get('gamma_small') != GAMMA_SMALL
            or not isinstance(latent, torch.Tensor) or not torch.isfinite(latent).all()):
        raise RuntimeError(f'Direction-probe latent cache identity mismatch: {path}')
    return record


def _load_small_score_cache(path, indices, rows):
    if not path.is_file():
        return {}
    try:
        payload = read_json(path)
        if (payload.get('indices') != indices or payload.get('gamma_old') != GAMMA_OLD
                or payload.get('gamma_small') != GAMMA_SMALL):
            return {}
        result = {}
        for entry in payload.get('rows', []):
            index = int(entry['index'])
            coordinates = entry.get('coordinates')
            if (index not in indices or entry.get('file') != image_name(index)
                    or not isinstance(coordinates, list) or len(coordinates) != 32
                    or not np.isfinite(np.asarray(coordinates, dtype=np.float64)).all()):
                continue
            result[index] = entry
        return result
    except (OSError, ValueError, TypeError, KeyError):
        return {}


def _signed_scores(score_rows, indices, signs):
    values = np.asarray([score_rows[index]['coordinates'] for index in indices], dtype=np.float64)
    return values * np.asarray(signs, dtype=np.float64).reshape(1, 32)


def _softplus_loss(margins):
    return float(np.logaddexp(0.0, -np.asarray(margins, dtype=np.float64)).mean())


def _accuracy_summary(coordinates, signs):
    coordinates = np.asarray(coordinates, dtype=np.float64)
    signs = np.asarray(signs, dtype=np.int8)
    if coordinates.ndim != 2 or coordinates.shape[1] != signs.size:
        raise ValueError('Coordinate and key dimensions do not match')
    decoded = np.where(coordinates >= 0, 1, -1)
    per_image = (decoded == signs.reshape(1, -1)).sum(axis=1)
    return {'mean_correct_bits_per_image': float(per_image.mean()),
            'median_correct_bits_per_image': float(np.median(per_image)),
            'mean_bit_accuracy': float(per_image.mean() / signs.size)}


def _fkr_summary(coordinates, signs, aggregate):
    return aggregate(np.asarray(coordinates, dtype=np.float64), np.asarray(signs, dtype=np.float64),
                     sizes=(1, 8, 32, 64), repeats=100, seed=20261029)


def _build_report(indices, clean, old, surrogate_clean, surrogate_small,
                  small_scores, signs, aggregate, timings, failed_bits=FAILED_BITS):
    clean_signed = _signed_scores(clean, indices, signs)
    old_signed = _signed_scores(old, indices, signs)
    small_signed = _signed_scores(small_scores, indices, signs)
    surrogate_clean = np.asarray(surrogate_clean, dtype=np.float64)
    surrogate_small = np.asarray(surrogate_small, dtype=np.float64)
    surrogate_delta = surrogate_small - surrogate_clean
    small_delta = small_signed - clean_signed
    old_delta = old_signed - clean_signed
    bits = []
    for offset in range(32):
        bit = offset + 1
        small_shift = float(small_delta[:, offset].mean())
        surrogate_shift = float(surrogate_delta[:, offset].mean())
        bits.append({
            'bit': bit,
            'clean_margin': float(clean_signed[:, offset].mean()),
            'small_gamma_shift': small_shift,
            'old_gamma_shift': float(old_delta[:, offset].mean()),
            'positive_fraction': float((small_delta[:, offset] > 0).mean()),
            'surrogate_shift': surrogate_shift,
            'png_shift': small_shift,
            'png_minus_surrogate_residual': small_shift - surrogate_shift,
            'surrogate_clean_margin': float(surrogate_clean[:, offset].mean()),
            'surrogate_small_margin': float(surrogate_small[:, offset].mean()),
            'problem_bit': bit in failed_bits,
        })
    surrogate_by_bit = {item['bit']: item['surrogate_shift'] for item in bits}
    small_by_bit = {item['bit']: item['small_gamma_shift'] for item in bits}
    old_by_bit = {item['bit']: item['old_gamma_shift'] for item in bits}
    diagnosis = classify_direction_evidence(failed_bits, surrogate_by_bit, small_by_bit, old_by_bit)
    coordinates = {
        'clean': np.asarray([clean[i]['coordinates'] for i in indices], dtype=np.float64),
        'bitwise_old_gamma': np.asarray([old[i]['coordinates'] for i in indices], dtype=np.float64),
        'bitwise_small_gamma': np.asarray([small_scores[i]['coordinates'] for i in indices], dtype=np.float64),
    }
    margins = {'clean': clean_signed, 'bitwise_old_gamma': old_signed,
               'bitwise_small_gamma': small_signed}
    accuracy = {name: _accuracy_summary(value, signs) for name, value in coordinates.items()}
    fkr = {name: _fkr_summary(value, signs, aggregate) for name, value in coordinates.items()}
    png_losses = {name: _softplus_loss(value) for name, value in margins.items()}
    surrogate_losses = {'clean': _softplus_loss(surrogate_clean),
                        'small_gamma': _softplus_loss(surrogate_small)}
    return {
        'status': 'complete', 'protocol': 'OGSC-32 cached-direction small-gamma diagnostic',
        'objective': OBJECTIVE, 'n': len(indices), 'indices': indices,
        'gamma_old': GAMMA_OLD, 'gamma_small': GAMMA_SMALL,
        'gamma_small_definition': 'gamma_old / 16; diagnostic only, not a tuned hyperparameter',
        'direction_estimate': '(cached_bitwise_latent - cached_clean_latent) / gamma_old; cached latents are FP16',
        'full_key_rule': 'original exact 32/32 evaluator; FKR reported only at group sizes 1/8/32/64',
        'small_sample_warning': 'FKR uses only 64 evenly spaced validation images; do not treat as a final population estimate.',
        'per_bit': bits,
        'accuracy': accuracy,
        'fkr': fkr,
        'softplus_loss': {'surrogate_fp32': surrogate_losses, 'scored_png': png_losses},
        'timings_seconds': timings,
        'diagnosis': diagnosis,
    }


def _write_markdown(path, report):
    failed = set(report['diagnosis']['failed_bits'])
    lines = [
        '# OGSC-32 direction probe', '',
        f"- Status: `{report['status']}`; evenly spaced validation samples: `{report['n']}`.",
        f"- Gamma old: `{report['gamma_old']:.12g}`; gamma small: `{report['gamma_small']:.12g}` (diagnostic only).",
        '- No diffusion sampling, training, full validation scoring, FID/KID, or full-set CLIP was run.',
        '- New output PNGs were decoded from cached clean/bitwise latents with native SD2.1 VAE; clean and old bitwise scores were reused.',
        '- FKR uses exact 32/32 and the original evaluator. This 64-image diagnostic omits FKR@256; small-sample FKR is not a final estimate.',
        '- “Small-gamma shift” and “PNG shift” are both the measured PNG `Δr_small`; the duplicate requested columns are explicitly the same value.', '',
        '| Bit | Clean margin | Small-gamma shift | Old-gamma shift | Positive fraction | Surrogate shift | PNG shift |',
        '|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for item in report['per_bit']:
        marker = '**' if item['bit'] in failed else ''
        values = [item['bit'], item['clean_margin'], item['small_gamma_shift'],
                  item['old_gamma_shift'], item['positive_fraction'],
                  item['surrogate_shift'], item['png_shift']]
        lines.append('| ' + ' | '.join(
            f'{marker}{value}{marker}' if column == 0 else f'{marker}{value:.6f}{marker}'
            for column, value in enumerate(values)) + ' |')
    lines += ['', '## Correct bits per image', '', '| Variant | Mean | Median | Mean accuracy |', '|---|---:|---:|---:|']
    for name, label in (('clean', 'Clean PNG'), ('bitwise_old_gamma', 'Bitwise PNG, old gamma'),
                        ('bitwise_small_gamma', 'Bitwise PNG, small gamma')):
        item = report['accuracy'][name]
        lines.append(f"| {label} | {item['mean_correct_bits_per_image']:.3f} | "
                     f"{item['median_correct_bits_per_image']:.1f} | {item['mean_bit_accuracy']:.5f} |")
    lines += ['', '## Exact full-key recovery', '',
              '| Variant | Group size | Exact 32/32 FKR |', '|---|---:|---:|']
    for name, label in (('clean', 'Clean PNG'), ('bitwise_old_gamma', 'Bitwise PNG, old gamma'),
                        ('bitwise_small_gamma', 'Bitwise PNG, small gamma')):
        for row in report['fkr'][name]['rows']:
            lines.append(f"| {label} | {row['images_per_group']} | {row['full_key_recovery_rate']:.6f} |")
    lines += ['', '## Softplus loss', '', '| Measurement path | Before/clean | After/small gamma | Old-gamma PNG |', '|---|---:|---:|---:|',
              f"| Surrogate FP32 | {report['softplus_loss']['surrogate_fp32']['clean']:.6f} | {report['softplus_loss']['surrogate_fp32']['small_gamma']:.6f} | — |",
              f"| Scored PNG | {report['softplus_loss']['scored_png']['clean']:.6f} | {report['softplus_loss']['scored_png']['bitwise_small_gamma']:.6f} | {report['softplus_loss']['scored_png']['bitwise_old_gamma']:.6f} |",
              '', '## Diagnosis', '',
              f"- Conclusion: **{report['diagnosis']['conclusion']}**.",
              f"- Failed-bit support counts: `{json.dumps(report['diagnosis']['evidence_counts'], sort_keys=True)}`.",
              f"- Rule: {report['diagnosis']['rule']}", '',
              '## Runtime', '', '| Stage | Seconds |', '|---|---:|']
    for name, seconds in report['timings_seconds'].items():
        lines.append(f'| {name.replace("_", " ")} | {seconds:.2f} |')
    lines += ['', 'The JSON also records the surrogate-to-PNG residual for each bit and the exact selected validation indices.', '']
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines), encoding='utf-8')


def run_direction_probe(paths, sources):
    if not torch.cuda.is_available():
        raise RuntimeError('Direction probe requires server CUDA; no local model execution is performed')
    overall_started = time.perf_counter()
    baseline = Path(paths['baseline_run'])
    output = Path(paths['output_run'])
    rows = _rows(paths, 'validation')
    if len(rows) != 1000:
        raise RuntimeError(f'Direction probe expects the existing 1,000-prompt validation split; found {len(rows)}')
    indices = evenly_spaced_indices(len(rows), PROBE_COUNT)
    config = read_json(Path(paths['source_run']) / 'config.json')
    seeds = [int(config['seed']) * 1_000_000 + int(row['image_id']) for row in rows]
    result_dir = output / 'result'
    state_dir = output / 'state/direction_probe'
    sample_cache_dir = state_dir / 'samples'
    marked_cache_dir = state_dir / 'marked_latents'
    image_dir = output / 'images/direction_probe_small'
    score_dir = output / 'scores'
    for directory in (result_dir, sample_cache_dir, marked_cache_dir, image_dir, score_dir):
        directory.mkdir(parents=True, exist_ok=True)

    clean_scores_path = baseline / 'scores/teacher_clean_validation.json'
    old_scores_path = output / 'scores/teacher_bitwise_validation.json'
    clean_scores = _read_full_scores(
        clean_scores_path, rows, 'teacher_clean_validation',
        baseline / 'images/teacher_clean_validation')
    old_scores = _read_full_scores(
        old_scores_path, rows, 'teacher_bitwise_validation',
        output / 'images/teacher_bitwise_validation')
    failed_bits = _failed_bits_from_baseline(output)

    cache_records = {}
    clean_latents = {}
    missing_marked = []
    started = time.perf_counter()
    print(f'OGSC direction probe cache scan: checking {len(indices)} selected samples.', flush=True)
    for position, index in enumerate(indices, start=1):
        row = rows[index]
        seed = seeds[index]
        clean_latents[index] = _clean_latent(baseline, output, index, row, seed)
        cache_path = _probe_cache_path(output, index)
        cache_records[index] = _read_probe_cache(cache_path, row, seed)
        if cache_records[index] is None and _load_marked(output, index, row, seed) is None:
            missing_marked.append(index)
        if position == 1 or position % 8 == 0 or position == len(indices):
            print(progress_line('direction probe latent cache scan', position, len(indices),
                                time.perf_counter() - started), flush=True)

    one_step_seconds = 0.0
    if missing_marked:
        print(f'OGSC direction probe: recomputing one-step controls for {len(missing_marked)} selected samples; '
              'no diffusion sampling.', flush=True)
        scorer = sources.extractor(Path(paths['teacher_clean']), device='cuda')
        carrier = sources.carrier(Path(paths['source_run']) / 'carrier', 'cuda')
        sources.verify_frozen(scorer)
        started = time.perf_counter()
        try:
            for work_i, index in enumerate(missing_marked, start=1):
                row, seed = rows[index], seeds[index]
                torch.cuda.synchronize()
                marked, diagnostics = one_step(
                    clean_latents[index].to('cuda'),
                    sources.signed_view(scorer, carrier, row['text']), GAMMA_OLD)
                torch.cuda.synchronize()
                record = {'objective': OBJECTIVE, 'gamma': GAMMA_OLD, 'latent': marked.cpu(),
                          'image_id': row['image_id'], 'text': row['text'], 'seed': seed,
                          'loss_before': diagnostics['loss_before']}
                path = marked_cache_dir / f'{index:04d}.pt'
                with atomic_path(path) as temporary:
                    torch.save(record, temporary)
                if work_i == 1 or work_i % 8 == 0 or work_i == len(missing_marked):
                    print(generation_progress_message(
                        'direction probe one-step recovery', len(indices) - len(missing_marked) + work_i,
                        len(indices), time.perf_counter() - started,
                        work_completed=work_i, work_total=len(missing_marked)), flush=True)
        finally:
            del scorer, carrier
            torch.cuda.empty_cache()
        one_step_seconds = time.perf_counter() - started

    started = time.perf_counter()
    missing_small = []
    for index in indices:
        if cache_records[index] is None:
            row, seed = rows[index], seeds[index]
            marked = _load_marked(output, index, row, seed)
            if marked is None:
                marked_path = marked_cache_dir / f'{index:04d}.pt'
                marked = _validate_bitwise_record(
                    torch.load(marked_path, map_location='cpu', weights_only=True), row, seed)
            small = interpolate_small_latent(clean_latents[index], marked['latent'])
            cache_records[index] = {
                'objective': OBJECTIVE, 'image_id': row['image_id'], 'text': row['text'], 'seed': seed,
                'gamma_old': GAMMA_OLD, 'gamma_small': GAMMA_SMALL, 'small_latent': small,
            }
            path = _probe_cache_path(output, index)
            with atomic_path(path) as temporary:
                torch.save(cache_records[index], temporary)
        if 'surrogate_clean_signed' not in cache_records[index] or 'surrogate_small_signed' not in cache_records[index]:
            missing_small.append(index)
    print(f'OGSC direction probe latent preparation: {len(indices) - len(missing_small)} cached; '
          f'{len(missing_small)} need surrogate scoring.', flush=True)
    latent_prep_seconds = time.perf_counter() - started

    missing_decode = [index for index in indices if not (image_dir / image_name(index)).is_file()]
    decode_seconds = 0.0
    if missing_decode:
        started = time.perf_counter()
        teacher = _semantic_teacher(paths)
        pipe = teacher.load_native(Path(paths['teacher_clean']))
        try:
            for work_i, index in enumerate(missing_decode, start=1):
                record = cache_records[index]
                image = _decode(pipe, record['small_latent'].to('cuda'))[0]
                _save_png(image, image_dir / image_name(index))
                torch.cuda.synchronize()
                if work_i == 1 or work_i % 8 == 0 or work_i == len(missing_decode):
                    print(generation_progress_message(
                        'direction probe small-gamma PNG decode', len(indices) - len(missing_decode) + work_i,
                        len(indices), time.perf_counter() - started,
                        work_completed=work_i, work_total=len(missing_decode)), flush=True)
        finally:
            del pipe
            torch.cuda.empty_cache()
        decode_seconds = time.perf_counter() - started
    else:
        print(f'OGSC direction probe small-gamma PNG decode: {len(indices)} cached; 0 remaining.', flush=True)

    metadata_path = image_dir / 'metadata.jsonl'
    with metadata_path.open('w', encoding='utf-8') as stream:
        for index in indices:
            stream.write(json.dumps({'index': index, 'file_name': image_name(index),
                                     'text': rows[index]['text']}) + '\n')

    score_path = score_dir / 'direction_probe_small.json'
    score_checkpoint = score_dir / 'direction_probe_small.partial.json'
    small_score_rows = _load_small_score_cache(score_path, indices, rows)
    if len(small_score_rows) != len(indices):
        partial_rows = _load_small_score_cache(score_checkpoint, indices, rows)
        small_score_rows.update(partial_rows)
    missing_score = [index for index in indices if index not in small_score_rows]
    scoring_seconds = 0.0
    if missing_score or missing_small:
        started = time.perf_counter()
        scorer = sources.extractor(Path(paths['teacher_clean']), device='cuda')
        carrier = sources.carrier(Path(paths['source_run']) / 'carrier', 'cuda')
        sources.verify_frozen(scorer)
        last_checkpoint = 0
        try:
            for work_i, index in enumerate(indices, start=1):
                row = rows[index]
                record_path = _probe_cache_path(output, index)
                record = cache_records[index]
                if 'surrogate_clean_signed' not in record or 'surrogate_small_signed' not in record:
                    signed_view = sources.signed_view(scorer, carrier, row['text'])
                    with torch.no_grad():
                        clean_margin = signed_view(clean_latents[index].to('cuda').float())
                        small_margin = signed_view(record['small_latent'].to('cuda').float())
                    record['surrogate_clean_signed'] = clean_margin.detach().cpu().flatten().tolist()
                    record['surrogate_small_signed'] = small_margin.detach().cpu().flatten().tolist()
                    cache_records[index] = record
                    with atomic_path(record_path) as temporary:
                        torch.save(record, temporary)
                if index not in small_score_rows:
                    png_path = image_dir / image_name(index)
                    pixels = sources.semantic_teacher.pixels([png_path], device='cuda')
                    with torch.no_grad():
                        coordinates = carrier.coordinates(scorer.encode_image(pixels, [row['text']]))
                    values = coordinates.detach().cpu().double().flatten().numpy().tolist()
                    small_score_rows[index] = {'index': index, 'file': image_name(index),
                                               'coordinates': values}
                if work_i == 1 or work_i % 8 == 0 or work_i == len(indices):
                    print(progress_line('direction probe surrogate and PNG scoring', work_i,
                                        len(indices), time.perf_counter() - started), flush=True)
                if work_i - last_checkpoint >= 4 or work_i == len(indices):
                    payload = {'indices': indices, 'gamma_old': GAMMA_OLD, 'gamma_small': GAMMA_SMALL,
                               'rows': [small_score_rows[i] for i in indices if i in small_score_rows]}
                    write_json(score_checkpoint, payload)
                    last_checkpoint = work_i
        finally:
            del scorer, carrier
            torch.cuda.empty_cache()
        scoring_seconds = time.perf_counter() - started
    if len(small_score_rows) != len(indices):
        raise RuntimeError(f'Direction probe only scored {len(small_score_rows)}/{len(indices)} PNGs')
    write_json(score_path, {'indices': indices, 'gamma_old': GAMMA_OLD, 'gamma_small': GAMMA_SMALL,
                            'rows': [small_score_rows[i] for i in indices]})
    if score_checkpoint.is_file():
        score_checkpoint.unlink()

    signs = sources.carrier(Path(paths['source_run']) / 'carrier', 'cpu').signs.numpy()
    surrogate_clean = [cache_records[i]['surrogate_clean_signed'] for i in indices]
    surrogate_small = [cache_records[i]['surrogate_small_signed'] for i in indices]
    timings = {'one_step_recovery': one_step_seconds,
               'latent_cache_and_interpolation': latent_prep_seconds,
               'native_vae_png_decode': decode_seconds, 'surrogate_and_png_scoring': scoring_seconds}
    report = _build_report(indices, clean_scores, old_scores, surrogate_clean,
                           surrogate_small, small_score_rows, signs,
                           sources.key_metrics.aggregate_metrics, timings, failed_bits)
    report['timings_seconds']['total_this_run'] = time.perf_counter() - overall_started
    report['counts'] = {'validation_total': len(rows), 'probe_samples': len(indices),
                        'cached_clean_latents_reused': len(indices),
                        'cached_marked_latents_used_or_recovered': len(indices),
                        'small_gamma_pngs': len(indices)}
    report['artifacts'] = {'small_gamma_png_dir': str(image_dir),
                           'small_gamma_score_file': str(score_path)}
    json_path = result_dir / 'direction_probe.json'
    markdown_path = result_dir / 'direction_probe.md'
    write_json(json_path, report)
    _write_markdown(markdown_path, report)
    print(f'OGSC direction probe report: {markdown_path}', flush=True)
    return {'status': 'complete', 'n': len(indices), 'indices': indices,
            'conclusion': report['diagnosis']['conclusion'],
            'json': str(json_path), 'markdown': str(markdown_path),
            'small_gamma_score_file': str(score_path)}
