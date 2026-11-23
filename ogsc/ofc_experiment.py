"""One resumable, paired 256-image validation of OFC-32."""

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .control import OBJECTIVE
from .direction_probe import (_clean_latent, _read_full_scores, _score_identity_matches,
                              evenly_spaced_indices)
from .evaluate import full_key_metrics, valid_score_rows
from .generation import (_decode, _rows, _save_png, generation_progress_message,
                         image_name)
from .ofc_control import ofc_control
from .progress import progress_line, stage_progress
from .state import atomic_path, read_json, write_json

OFC_OBJECTIVE = 'ofc32_online_clean_anchored_feedback_v1'
OFC_LABEL = 'teacher_ofc_validation'
REFERENCE_LABELS = ('teacher_clean_validation', 'teacher_ogsc_validation',
                    'teacher_bitwise_validation')
QUALITY_LABELS = ('teacher_ogsc_validation', 'teacher_bitwise_validation', OFC_LABEL)
QUALITY_LIMITS = {'mean_paired_lpips': 0.15, 'mean_clip_drop': 0.01}
FKR_SEED = 20261029


def _coordinates(score_rows, indices):
    return np.asarray([score_rows[i]['coordinates'] for i in indices], dtype=np.float64)


def _variant_metrics(coordinates, signs, aggregate):
    coordinates = np.asarray(coordinates, dtype=np.float64)
    signs = np.asarray(signs, dtype=np.int8).reshape(32)
    signed = coordinates * signs.reshape(1, 32)
    decoded = np.where(coordinates >= 0, 1, -1)
    correct = (decoded == signs.reshape(1, 32)).sum(axis=1)
    mean_margins = signed.mean(axis=0)
    groups = full_key_metrics(coordinates, signs.astype(np.float64), aggregate)
    return {
        'n': int(signed.shape[0]),
        'bit_accuracy_at_1': float((decoded == signs.reshape(1, 32)).mean()),
        'mean_correct_bits_per_image': float(correct.mean()),
        'best_correct_bits_per_image': int(correct.max()),
        'exact_32_of_32_images': int((correct == 32).sum()),
        'minimum_signed_margin': float(signed.min()),
        'mean_minimum_signed_margin': float(signed.min(axis=1).mean()),
        'mean_signed_margins_32': mean_margins.tolist(),
        'per_bit_accuracy_32': (decoded == signs.reshape(1, 32)).mean(axis=0).tolist(),
        'official_key_metrics': groups,
    }


def _record_path(output, index):
    return output / 'state/ofc_validation/marked_latents' / f'{index:04d}.pt'


def _read_record(path, row, seed, gamma, run_identity):
    if not path.is_file():
        return None
    record = torch.load(path, map_location='cpu', weights_only=True)
    latent = record.get('latent')
    if (record.get('objective') != OFC_OBJECTIVE or record.get('image_id') != row['image_id']
            or record.get('text') != row['text'] or record.get('seed') != seed
            or record.get('gamma_budget') != gamma or record.get('run_identity') != run_identity
            or not isinstance(latent, torch.Tensor)
            or not torch.isfinite(latent).all()):
        raise RuntimeError(f'OFC cached latent identity mismatch: {path}')
    return record


def _validate_paired_provenance(baseline, bitwise):
    baseline_path = Path(baseline) / 'provenance.json'
    bitwise_path = Path(bitwise) / 'provenance.json'
    if not baseline_path.is_file() or not bitwise_path.is_file():
        raise RuntimeError('Both baseline and bitwise runs need provenance.json before paired OFC validation')
    baseline_identity, bitwise_identity = read_json(baseline_path), read_json(bitwise_path)
    fields = ('source_run_sha256', 'source_files_sha256', 'carrier_artifact_sha256', 'model')
    for field in fields:
        if not baseline_identity.get(field) or baseline_identity.get(field) != bitwise_identity.get(field):
            raise RuntimeError(f'Baseline and bitwise provenance differ in {field}; refusing paired comparison')
    baseline_model = baseline_identity.get('resolved_model_path')
    bitwise_model = bitwise_identity.get('resolved_model_path')
    if not baseline_model or not bitwise_model or Path(baseline_model).resolve() != Path(bitwise_model).resolve():
        raise RuntimeError('Baseline and bitwise runs do not use the same resolved clean model')
    return {field: baseline_identity[field] for field in fields} | {
        'resolved_model_path': str(Path(baseline_model).resolve())}


def _freeze_run_identity(output, identity):
    path = Path(output) / 'calibration/ofc_validation.json'
    if path.is_file():
        existing = read_json(path)
        if existing.get('identity') != identity:
            raise RuntimeError('Existing OFC output identity differs from requested inputs; use a new --output-run')
        existing['status'] = 'running'
        write_json(path, existing)
        return existing
    payload = {'status': 'running', 'objective': OFC_OBJECTIVE, 'identity': identity}
    write_json(path, payload)
    return payload


def _score_identity(folder, label, metadata):
    files = []
    started = time.perf_counter()
    for index, row in enumerate(metadata, start=1):
        stat = (folder / row['file_name']).stat()
        files.append([row['file_name'], stat.st_size, stat.st_mtime_ns])
        if index == 1 or index % 64 == 0 or index == len(metadata):
            print(progress_line(f'{label} score-cache identity', index, len(metadata),
                                time.perf_counter() - started), flush=True)
    generation_path = folder / 'generation.json'
    generation = read_json(generation_path) if generation_path.is_file() else None
    return {'label': label, 'metadata': metadata, 'generation': generation, 'files': files}


def _score_ofc_pngs(paths, sources, indices, selected_rows):
    output = Path(paths['output_run'])
    folder = output / 'images' / OFC_LABEL
    metadata = [{'file_name': image_name(index), 'text': row['text']}
                for index, row in zip(indices, selected_rows)]
    score_path = output / 'scores' / f'{OFC_LABEL}.json'
    identity_path = score_path.with_suffix('.identity.json')
    identity = _score_identity(folder, OFC_LABEL, metadata)
    if score_path.is_file() and identity_path.is_file():
        cached = read_json(score_path)
        if (_score_identity_matches(read_json(identity_path), identity)
                and valid_score_rows(cached, metadata)):
            print(f'OGSC {OFC_LABEL}: reusing completed score file ({len(cached)} selected images).', flush=True)
            return cached

    checkpoint = output / 'scores/checkpoints' / f'{OFC_LABEL}.json'
    score_rows = []
    if checkpoint.is_file():
        try:
            value = read_json(checkpoint)
            candidate = value.get('rows', [])
            if (value.get('identity') == identity and isinstance(candidate, list)
                    and len(candidate) <= len(metadata)
                    and valid_score_rows(candidate, metadata[:len(candidate)])):
                score_rows = candidate
        except (OSError, ValueError, TypeError):
            score_rows = []
    print(f'OGSC {OFC_LABEL} PNG scoring: {len(score_rows)}/{len(metadata)} cached; batches of 4.', flush=True)
    extractor = sources.extractor(Path(paths['teacher_clean']), device='cuda')
    carrier = sources.carrier(Path(paths['carrier_dir']), 'cuda')
    sources.verify_frozen(extractor)
    teacher = sources.semantic_teacher
    started = time.perf_counter()
    try:
        for start in range(len(score_rows), len(metadata), 4):
            batch = metadata[start:start + 4]
            images = teacher.pixels([folder / row['file_name'] for row in batch], device='cuda')
            with torch.no_grad():
                coordinates = carrier.coordinates(extractor.encode_image(
                    images, [row['text'] for row in batch]))
            values = coordinates.detach().cpu().double().numpy()
            score_rows.extend({'file': row['file_name'], 'coordinates': value.tolist()}
                              for row, value in zip(batch, values))
            completed = len(score_rows)
            if completed == len(metadata) or completed % 32 < 4:
                write_json(checkpoint, {'identity': identity, 'rows': score_rows})
                print(progress_line(f'{OFC_LABEL} PNG scoring', completed, len(metadata),
                                    time.perf_counter() - started), flush=True)
    finally:
        del extractor, carrier
        torch.cuda.empty_cache()
    write_json(score_path, score_rows)
    write_json(identity_path, identity)
    if checkpoint.is_file():
        checkpoint.unlink()
    return score_rows


def _quality_identity(folders, indices, rows, gamma):
    result = {'objective': OFC_OBJECTIVE, 'gamma_budget': gamma, 'indices': indices,
              'files': {}, 'prompts': [row['text'] for row in rows]}
    total = len(indices) * len(folders)
    completed = 0
    started = time.perf_counter()
    for label, folder in folders.items():
        entries = []
        for index in indices:
            path = folder / image_name(index)
            stat = path.stat()
            entries.append([path.name, stat.st_size, stat.st_mtime_ns])
            completed += 1
            if completed == 1 or completed % 128 == 0 or completed == total:
                print(progress_line('paired quality-cache identity', completed, total,
                                    time.perf_counter() - started), flush=True)
        result['files'][label] = entries
    return result


def _paired_quality(paths, folders, indices, rows, teacher, gamma):
    output = Path(paths['output_run'])
    final_path = output / 'result/ofc_validation_quality.json'
    partial_path = output / 'result/ofc_validation_quality.partial.json'
    identity = _quality_identity(folders, indices, rows, gamma)
    if final_path.is_file():
        cached = read_json(final_path)
        if cached.get('identity') == identity:
            print('OGSC OFC paired CLIP/LPIPS: reusing completed quality results.', flush=True)
            return cached
    partial = None
    if partial_path.is_file():
        try:
            candidate = read_json(partial_path)
            count = len(candidate.get('clip_clean', []))
            variants = candidate.get('variants', {})
            if (candidate.get('identity') == identity and count <= len(indices)
                    and all(len(variants.get(label, {}).get(key, [])) == count
                            for label in QUALITY_LABELS for key in ('clip', 'lpips'))):
                partial = candidate
        except (OSError, ValueError, TypeError):
            partial = None
    clip_clean = partial.get('clip_clean', []) if partial else []
    variants = partial.get('variants', {}) if partial else {}
    variants = {label: {'clip': variants.get(label, {}).get('clip', []),
                        'lpips': variants.get(label, {}).get('lpips', [])}
                for label in QUALITY_LABELS}
    resumed = len(clip_clean)

    harness = Path(paths['weight32_repo']) / 'BEST_TEACHER_32bits_refine/harness'
    import sys
    sys.path.insert(0, str(harness))
    from utility import CLIP_REV
    from transformers import CLIPModel, CLIPProcessor
    import lpips

    device = torch.device('cpu' if paths.get('offload') else 'cuda')
    clip = processor = perceptual = None
    if resumed < len(indices):
        with stage_progress('load paired CLIP and LPIPS quality models'):
            clip = CLIPModel.from_pretrained('openai/clip-vit-base-patch32', revision=CLIP_REV).to(device).eval().requires_grad_(False)
            processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32', revision=CLIP_REV)
            perceptual = lpips.LPIPS(net='alex', version='0.1').to(device).eval().requires_grad_(False)
    started = time.perf_counter()
    try:
        for start in range(resumed, len(indices), 8):
            end = min(start + 8, len(indices))
            batch_indices, batch_rows = indices[start:end], rows[start:end]
            clean_files = [folders['teacher_clean_validation'] / image_name(i) for i in batch_indices]
            clean_images = []
            for path in clean_files:
                with Image.open(path) as image:
                    clean_images.append(image.convert('RGB'))
            clean_pixels = teacher.pixels(clean_files, device=str(device))
            clean_inputs = processor(text=[row['text'] for row in batch_rows], images=clean_images,
                                     return_tensors='pt', padding=True, truncation=True).to(device)
            with torch.inference_mode():
                clean_output = clip(**clean_inputs)
                clip_clean.extend((clean_output.image_embeds * clean_output.text_embeds)
                                  .sum(-1).clamp_min(0).mul(2.5).cpu().tolist())
            for label in QUALITY_LABELS:
                paths_batch = [folders[label] / image_name(i) for i in batch_indices]
                images = []
                for path in paths_batch:
                    with Image.open(path) as image:
                        images.append(image.convert('RGB'))
                inputs = processor(text=[row['text'] for row in batch_rows], images=images,
                                   return_tensors='pt', padding=True, truncation=True).to(device)
                marked_pixels = teacher.pixels(paths_batch, device=str(device))
                with torch.inference_mode():
                    clip_output = clip(**inputs)
                    clip_values = (clip_output.image_embeds * clip_output.text_embeds).sum(-1).clamp_min(0).mul(2.5)
                    lpips_values = perceptual(clean_pixels, marked_pixels).flatten()
                variants[label]['clip'].extend(clip_values.cpu().tolist())
                variants[label]['lpips'].extend(lpips_values.cpu().tolist())
            completed = end
            if completed == len(indices) or completed % 32 == 0:
                write_json(partial_path, {'identity': identity, 'clip_clean': clip_clean,
                                          'variants': variants})
                print(progress_line('paired OFC CLIP/LPIPS', completed, len(indices),
                                    time.perf_counter() - started), flush=True)
    finally:
        del clip, processor, perceptual
        torch.cuda.empty_cache()

    clean_clip = float(np.mean(clip_clean))
    report = {'identity': identity, 'n': len(indices), 'clip_revision': CLIP_REV,
              'clip_definition': '2.5 * max(cos(image,text),0), original utility convention',
              'lpips': 'alex, version=0.1', 'limits': QUALITY_LIMITS,
              'clean_clip': clean_clip, 'variants': {}}
    for label in QUALITY_LABELS:
        clip_values, lpips_values = variants[label]['clip'], variants[label]['lpips']
        clip_mean, lpips_mean = float(np.mean(clip_values)), float(np.mean(lpips_values))
        drop = clean_clip - clip_mean
        report['variants'][label] = {
            'clip_score': clip_mean, 'clip_drop': drop,
            'mean_paired_lpips': lpips_mean,
            'median_paired_lpips': float(np.median(lpips_values)),
            'passes_existing_quality_limits': (lpips_mean <= QUALITY_LIMITS['mean_paired_lpips']
                                               and drop <= QUALITY_LIMITS['mean_clip_drop']),
            'clip_per_image': clip_values, 'lpips_per_image': lpips_values,
        }
    write_json(final_path, report)
    if partial_path.is_file():
        partial_path.unlink()
    return report


def _independent_clean_fpr(paths, sources):
    baseline = Path(paths['baseline_run'])
    rows = _rows(paths, 'test')
    folder = baseline / 'images/teacher_clean_test'
    score_path = baseline / 'scores/teacher_clean_test.json'
    try:
        scores = _read_full_scores(score_path, rows, 'teacher_clean_test', folder)
    except (RuntimeError, OSError, ValueError) as error:
        return {'status': 'unavailable', 'reason': str(error)}
    coordinates = np.asarray([entry['coordinates'] for entry in scores], dtype=np.float64)
    signs = sources.carrier(Path(paths['carrier_dir']), 'cpu').signs.numpy()
    exact = np.all(np.where(coordinates >= 0, 1, -1) == signs.reshape(1, 32), axis=1)
    return {'split': 'independent clean test controls', 'n': int(exact.size),
            'false_positives': int(exact.sum()), 'empirical_rate': float(exact.mean()),
            'interpretation': 'Observed sample rate only; zero observed positives is not true FPR=0.'}


def _write_report(output, report):
    json_path = output / 'result/ofc_validation.json'
    md_path = output / 'result/ofc_validation.md'
    write_json(json_path, report)
    comparison = report['comparison']
    lines = ['# OFC-32 paired validation', '',
             f"- Validation images: {report['n']} (evenly spaced indices)",
             f"- Frozen gamma budget from baseline: `{report['gamma_budget']:.9f}`",
             '- No new gamma sweep, clean-latent sampling, weight training, or test-based method selection.',
             '- Final RGB8 PNG scores use the original frozen verifier and exact 32/32 FKR evaluator.',
             '', '| Variant | Bit acc @1 | Mean bits/image | Best bits/image | Exact images | FKR @1 | @8 | @32 | @64 | @256 | LPIPS | CLIP drop |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for label, metrics in comparison.items():
        by_size = {item['images_per_group']: item for item in metrics['official_key_metrics']['rows']}
        quality = report['quality']['variants'].get(label, {})
        lpips_value, clip_drop = quality.get('mean_paired_lpips'), quality.get('clip_drop')
        render = lambda value: '—' if value is None else f'{value:.5f}'
        lines.append(f"| {label.removeprefix('teacher_')} | {metrics['bit_accuracy_at_1']:.5f} | "
                     f"{metrics['mean_correct_bits_per_image']:.3f} | {metrics['best_correct_bits_per_image']} | "
                     f"{metrics['exact_32_of_32_images']} | "
                     + ' | '.join(f"{by_size[size]['full_key_recovery_rate']:.6f}" for size in (1, 8, 32, 64, 256))
                     + f" | {render(lpips_value)} | {render(clip_drop)} |")
    ofc = comparison[OFC_LABEL]
    lines += ['', '## OFC per-bit PNG results', '',
              '| Bit | Mean signed margin | PNG bit accuracy |', '|---:|---:|---:|']
    for index, (margin, accuracy) in enumerate(zip(ofc['mean_signed_margins_32'], ofc['per_bit_accuracy_32']), start=1):
        lines.append(f'| {index} | {margin:.6f} | {accuracy:.5f} |')
    latency = report['control_latency_seconds']
    lines += ['', '## Control latency', '',
              f"- Mean/p50/p95: `{latency['mean']:.3f}/{latency['p50']:.3f}/{latency['p95']:.3f}` seconds/image.",
              f"- Mean VJPs/image: `{latency['mean_vjps']:.3f}`.",
              '- Latency excludes model loading, PNG decoding, and the original 50-step generation.',
              '', '## Independent clean controls', '',
              f"- {report['independent_clean_test_fpr']}",
              '', '## Notes', '',
              '- Group size 256 has one unique group in this 256-image pilot; repeated partitions do not create independent groups.',
              '- OFC success is reported from measured PNG output; the controller does not guarantee exact recovery or quality limits.',
              '- LPIPS limit is mean ≤0.15; CLIP drop limit is mean ≤0.01.', '']
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text('\n'.join(lines), encoding='utf-8')
    return {'json': str(json_path), 'markdown': str(md_path)}


def run_ofc_validation(paths, sources, count=256):
    if not torch.cuda.is_available():
        raise RuntimeError('OFC validation needs server CUDA; local checks use CPU toy tests only')
    baseline, bitwise, output = (Path(paths[name]) for name in ('baseline_run', 'bitwise_run', 'output_run'))
    if output.resolve() in (baseline.resolve(), bitwise.resolve()):
        raise RuntimeError('OFC output must be separate; baseline and bitwise runs are read-only')
    rows = _rows(paths, 'validation')
    if len(rows) != 1000:
        raise RuntimeError(f'OFC paired experiment expects 1,000 validation prompts; found {len(rows)}')
    if not 1 <= int(count) <= len(rows):
        raise ValueError('count must be between 1 and the number of validation prompts')
    indices = evenly_spaced_indices(len(rows), int(count))
    selected_rows = [rows[index] for index in indices]
    config = read_json(Path(paths['source_run']) / 'config.json')
    seeds = [int(config['seed']) * 1_000_000 + int(row['image_id']) for row in selected_rows]
    paired_provenance = _validate_paired_provenance(baseline, bitwise)
    gamma_artifact = read_json(baseline / 'calibration/gamma.json')
    if gamma_artifact.get('status') != 'selected' or not math.isfinite(float(gamma_artifact.get('gamma', 0))):
        raise RuntimeError('Baseline has no finite selected gamma budget')
    gamma = float(gamma_artifact['gamma'])
    if gamma <= 0:
        raise RuntimeError('Baseline selected gamma budget must be positive')

    identity = {
        'objective': OFC_OBJECTIVE, 'count': len(indices), 'indices': indices,
        'gamma_budget': gamma, 'source_run': str(Path(paths['source_run']).resolve()),
        'baseline_run': str(baseline.resolve()), 'bitwise_run': str(bitwise.resolve()),
        **paired_provenance,
    }
    _freeze_run_identity(output, identity)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'result').mkdir(parents=True, exist_ok=True)
    (output / 'state/ofc_validation/marked_latents').mkdir(parents=True, exist_ok=True)
    (output / 'images' / OFC_LABEL).mkdir(parents=True, exist_ok=True)
    (output / 'scores').mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    baseline_scores = {}
    reference_paths = {
        'teacher_clean_validation': baseline / 'scores/teacher_clean_validation.json',
        'teacher_ogsc_validation': baseline / 'scores/teacher_ogsc_validation.json',
        'teacher_bitwise_validation': bitwise / 'scores/teacher_bitwise_validation.json',
    }
    reference_folders = {
        'teacher_clean_validation': baseline / 'images/teacher_clean_validation',
        'teacher_ogsc_validation': baseline / 'images/teacher_ogsc_validation',
        'teacher_bitwise_validation': bitwise / 'images/teacher_bitwise_validation',
    }
    with stage_progress('reuse paired clean, soft-min, and bitwise PNG score caches'):
        all_rows = _rows(paths, 'validation')
        for label in REFERENCE_LABELS:
            baseline_scores[label] = _read_full_scores(reference_paths[label], all_rows,
                                                        label, reference_folders[label])

    marked_records = {}
    pending = []
    decode_pending = []
    for position, (index, row, seed) in enumerate(zip(indices, selected_rows, seeds), start=1):
        record = _read_record(_record_path(output, index), row, seed, gamma, identity)
        png = output / 'images' / OFC_LABEL / image_name(index)
        if record is None:
            if png.is_file():
                raise RuntimeError(f'OFC PNG exists without its latent record at index {index}; refusing to guess its source')
            pending.append((position, index, row, seed))
        elif not png.is_file():
            marked_records[index] = record
            decode_pending.append((index, record['latent']))
        else:
            marked_records[index] = record
        if position == 1 or position % 64 == 0 or position == len(indices):
            print(progress_line('OFC validation cache scan', position, len(indices),
                                time.perf_counter() - started), flush=True)

    if pending:
        extractor = sources.extractor(Path(paths['teacher_clean']), device='cuda')
        carrier = sources.carrier(Path(paths['carrier_dir']), 'cuda')
        sources.verify_frozen(extractor)
        control_started = time.perf_counter()
        for work_i, (position, index, row, seed) in enumerate(pending, start=1):
            record = _read_record(_record_path(output, index), row, seed, gamma, identity)
            if record is None:
                clean = _clean_latent(baseline, bitwise, index, row, seed)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                tick = time.perf_counter()
                view = sources.signed_view(extractor, carrier, row['text'])
                marked, diagnostics = ofc_control(clean.to('cuda'), view, gamma)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - tick
                record = {'objective': OFC_OBJECTIVE, 'index': index,
                          'image_id': row['image_id'], 'text': row['text'], 'seed': seed,
                          'gamma_budget': gamma, 'run_identity': identity,
                          'latent': marked.detach().cpu(),
                          'diagnostics': diagnostics, 'control_seconds': elapsed}
                with atomic_path(_record_path(output, index)) as temporary:
                    torch.save(record, temporary)
            else:
                marked = record['latent']
            if not (output / 'images' / OFC_LABEL / image_name(index)).is_file():
                decode_pending.append((index, marked))
            marked_records[index] = record
            if work_i == 1 or work_i % 8 == 0 or work_i == len(pending):
                print(generation_progress_message('OFC online control and latent cache', work_i,
                                                   len(pending), time.perf_counter() - control_started,
                                                   work_completed=work_i, work_total=len(pending)), flush=True)
        del extractor, carrier
        torch.cuda.empty_cache()
    else:
        print(f'OGSC OFC validation: all {len(indices)} marked latents and PNGs cached.', flush=True)

    if decode_pending:
        with stage_progress('native FP16 VAE decode and save OFC PNGs'):
            native = sources.semantic_teacher.load_native(Path(paths['teacher_clean']))
            decode_started = time.perf_counter()
            try:
                for completed, (index, marked) in enumerate(decode_pending, start=1):
                    image = _decode(native, marked.to('cuda'))[0]
                    _save_png(image, output / 'images' / OFC_LABEL / image_name(index))
                    if completed == 1 or completed % 8 == 0 or completed == len(decode_pending):
                        print(generation_progress_message('OFC native VAE decode', completed,
                                                           len(decode_pending), time.perf_counter() - decode_started,
                                                           work_completed=completed,
                                                           work_total=len(decode_pending)), flush=True)
            finally:
                del native
                torch.cuda.empty_cache()

    image_folder = output / 'images' / OFC_LABEL
    metadata = [{'file_name': image_name(index), 'text': row['text']}
                for index, row in zip(indices, selected_rows)]
    with (image_folder / 'metadata.jsonl').open('w', encoding='utf-8') as stream:
        for item in metadata:
            stream.write(json.dumps(item) + '\n')
    ofc_scores = _score_ofc_pngs(paths, sources, indices, selected_rows)
    signs = sources.carrier(Path(paths['carrier_dir']), 'cpu').signs.numpy()
    metrics = {}
    with stage_progress('aggregate exact 32-bit FKR on paired validation indices'):
        for label in REFERENCE_LABELS:
            metrics[label] = _variant_metrics(_coordinates(baseline_scores[label], indices), signs,
                                              sources.key_metrics.aggregate_metrics)
        by_filename = {row['file']: row for row in ofc_scores}
        chosen = [by_filename[image_name(index)]['coordinates'] for index in indices]
        metrics[OFC_LABEL] = _variant_metrics(np.asarray(chosen, dtype=np.float64), signs,
                                              sources.key_metrics.aggregate_metrics)

    folders = {'teacher_clean_validation': reference_folders['teacher_clean_validation'],
               'teacher_ogsc_validation': reference_folders['teacher_ogsc_validation'],
               'teacher_bitwise_validation': reference_folders['teacher_bitwise_validation'],
               OFC_LABEL: image_folder}
    quality = _paired_quality(paths, folders, indices, selected_rows,
                              sources.semantic_teacher, gamma)
    independent_fpr = _independent_clean_fpr(paths, sources)
    measured = [record for record in marked_records.values() if record.get('control_seconds') is not None]
    seconds = [float(record['control_seconds']) for record in measured]
    vjp_counts = [int(record['diagnostics']['vjp_count']) for record in measured]
    latency = {'n_measured': len(seconds), 'mean': float(np.mean(seconds)) if seconds else 0.0,
               'p50': float(np.percentile(seconds, 50)) if seconds else 0.0,
               'p95': float(np.percentile(seconds, 95)) if seconds else 0.0,
               'mean_vjps': float(np.mean(vjp_counts)) if vjp_counts else 0.0}
    report = {'protocol': OFC_OBJECTIVE, 'split': 'validation', 'n': len(indices),
              'indices': indices, 'gamma_budget': gamma, 'comparison': metrics,
              'quality': quality, 'independent_clean_test_fpr': independent_fpr,
              'control_latency_seconds': latency,
              'per_image_control': [{'index': index,
                                     'diagnostics': marked_records[index]['diagnostics'],
                                     'control_seconds': marked_records[index].get('control_seconds')}
                                    for index in indices],
              'fkr': {'sizes': [1, 8, 32, 64, 256], 'repeats': 100,
                      'seed': FKR_SEED,
                      'group_256_note': 'One unique 256-image group in this pilot.'},
              'selection': 'No test-based selection; no gamma sweep; no FID/KID.'}
    result_paths = _write_report(output, report)
    calibration = read_json(output / 'calibration/ofc_validation.json')
    calibration.update({'status': 'complete', 'result': result_paths['markdown']})
    write_json(output / 'calibration/ofc_validation.json', calibration)
    return {'status': 'complete', 'n': len(indices), 'gamma_budget': gamma,
            'markdown': result_paths['markdown'], 'json': result_paths['json'],
            'control_latency_seconds': latency,
            'ofc_bit_accuracy_at_1': metrics[OFC_LABEL]['bit_accuracy_at_1'],
            'ofc_exact_32_of_32_images': metrics[OFC_LABEL]['exact_32_of_32_images']}
