"""Validation-only gamma selection with cached latents and one-step directions."""

import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .control import OBJECTIVE, one_step
from .generation import _decode, _rows, _save_png, image_name
from .progress import progress_line
from .state import atomic_path, read_json, sha256, write_json


def progress_message(phase, completed, total, elapsed_seconds, work_completed=None, work_total=None):
    total = max(int(total), 1)
    completed = min(max(int(completed), 0), total)
    fraction = completed / total
    work_total = total if work_total is None else max(int(work_total), 0)
    work_completed = completed if work_completed is None else min(max(int(work_completed), 0), work_total)
    if work_total == 0:
        eta_text = 'ETA 0 min (all items cached)'
    elif work_completed < min(10, work_total):
        eta_text = f'ETA warming up ({work_completed}/{min(10, work_total)} timed items)'
    else:
        eta = elapsed_seconds * (work_total - work_completed) / work_completed
        eta_text = f'ETA {eta / 60:.1f} min'
    work_text = f' | {work_completed}/{work_total} this run'
    return (f'OGSC gamma tuning | {phase}: {completed}/{total} ({fraction:.0%}){work_text} | '
            f'{elapsed_seconds / 60:.1f} min elapsed | {eta_text}')


def candidate_gammas(clean_latents, points=17, max_rms_fraction=1.0):
    """Build a bounded uniform search from observed clean latent RMS values."""
    if points < 2 or not 0 < max_rms_fraction <= 1:
        raise ValueError('Invalid bounded gamma search settings')
    rms = torch.cat([z.detach().float().reshape(z.shape[0], -1).square().mean(1).sqrt().cpu()
                     for z in clean_latents])
    upper = float(rms.median()) * max_rms_fraction
    if not math.isfinite(upper) or upper <= 0:
        raise RuntimeError('Could not derive a positive gamma bound from clean latent RMS')
    return torch.linspace(upper / points, upper, points).tolist(), upper


def select_gamma(trials):
    """Select the largest measured positive passing trial; allow nonmonotone quality."""
    passing = [row for row in trials if row['gamma'] > 0 and row['passes']]
    return max(passing, key=lambda row: row['gamma'])['gamma'] if passing else None


def tune_gamma(paths, sources):
    if not torch.cuda.is_available():
        raise RuntimeError('Gamma tuning needs server CUDA; use CPU tests for controller logic')
    source, output = Path(paths['source_run']), Path(paths['output_run'])
    prompts = _rows(paths, 'validation')
    config = read_json(source / 'config.json')
    seed_ids = [int(config['seed']) * 1_000_000 + int(row['image_id']) for row in prompts]
    cache = output / 'calibration/validation'
    latent_dir, clean_dir = cache / 'latents', output / 'images/teacher_clean_validation'
    latent_dir.mkdir(parents=True, exist_ok=True); clean_dir.mkdir(parents=True, exist_ok=True)
    teacher = sources.semantic_teacher
    # Generate validation clean latents once with native FP16 inference.
    print(f'OGSC gamma tuning: preparing {len(prompts)} clean validation latents (50-step PNDM each).', flush=True)
    pending = []
    for i in range(len(prompts)):
        latent_path, image_path = latent_dir / f'{i:04d}.pt', clean_dir / image_name(i)
        if not (latent_path.exists() and image_path.exists()):
            pending.append(i)
    cached_count = len(prompts) - len(pending)
    print(f'OGSC gamma tuning: {cached_count} cached; {len(pending)} clean samples remain.', flush=True)
    if pending:
        pipe = teacher.load_native(Path(paths['teacher_clean']))
        clean_started = time.perf_counter()
        try:
            for work_i, i in enumerate(pending, start=1):
                row, seed = prompts[i], seed_ids[i]
                lp, ip = latent_dir / f'{i:04d}.pt', clean_dir / image_name(i)
                gen = torch.Generator('cuda').manual_seed(seed)
                with torch.inference_mode():
                    z = pipe([row['text']], generator=[gen], num_inference_steps=50,
                             guidance_scale=7.5, height=512, width=512, output_type='latent').images
                _save_png(_decode(pipe, z)[0], ip)
                with atomic_path(lp) as tmp:
                    torch.save({'latent': z.detach().cpu(), 'image_id': row['image_id'], 'text': row['text'], 'seed': seed}, tmp)
                if work_i == 1 or work_i % 10 == 0 or work_i == len(pending):
                    print(progress_message('clean validation sampling', cached_count + work_i, len(prompts),
                                           time.perf_counter() - clean_started,
                                           work_completed=work_i, work_total=len(pending)), flush=True)
        finally:
            del pipe; torch.cuda.empty_cache()
    else:
        print('OGSC gamma tuning: all clean validation latents and images are cached; skipping sampling.', flush=True)
    clean_latents = [torch.load(latent_dir / f'{i:04d}.pt', map_location='cpu', weights_only=True)['latent']
                     for i in range(len(prompts))]
    candidates, upper = candidate_gammas(clean_latents)
    print(f'OGSC gamma tuning: cached clean latents; gamma bound={upper:.6g}, evaluating {len(candidates)} quality candidates.', flush=True)

    # Cache normalized directions once, one VJP per validation image.
    print(f'OGSC gamma tuning: computing one latent direction for {len(prompts)} validation images.', flush=True)
    scorer = sources.extractor(Path(paths['teacher_clean']), device='cuda')
    carrier = sources.carrier(source / 'carrier', 'cuda')
    directions = []
    directions_started = time.perf_counter()
    try:
        for i, (row, latent) in enumerate(zip(prompts, clean_latents)):
            z = latent.to(device='cuda', dtype=torch.float32)
            # gamma=1 returns the normalized negative gradient; subtract z to recover direction.
            unit, _ = one_step(z, sources.signed_view(scorer, carrier, row['text']), 1.0)
            directions.append((unit.float() - z.float()).cpu())
            completed = i + 1
            if completed == 1 or completed % 25 == 0 or completed == len(prompts):
                print(progress_line('validation latent directions', completed, len(prompts),
                                    time.perf_counter() - directions_started), flush=True)
    finally:
        del scorer, carrier; torch.cuda.empty_cache()

    # Quality models use the original weight32 CLIP scoring convention.
    harness = Path(paths['weight32_repo']) / 'BEST_TEACHER_32bits_refine/harness'
    sys.path.insert(0, str(harness))
    from utility import CLIP_REV
    from transformers import CLIPModel, CLIPProcessor
    import lpips
    quality_device = torch.device('cpu' if paths.get('offload') else 'cuda')
    clip = CLIPModel.from_pretrained('openai/clip-vit-base-patch32', revision=CLIP_REV).to(quality_device).eval().requires_grad_(False)
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32', revision=CLIP_REV)
    perceptual = lpips.LPIPS(net='alex', version='0.1').to(quality_device).eval().requires_grad_(False)
    decoder = teacher.load_native(Path(paths['teacher_clean']))
    clean_clip, clean_pixels = [], []
    clean_metric_started = time.perf_counter()
    try:
        for i, (row, clean_path) in enumerate(zip(prompts, [clean_dir / image_name(i) for i in range(len(prompts))])):
            with Image.open(clean_path) as im:
                rgb = im.convert('RGB')
            inputs = processor(text=[row['text']], images=[rgb], return_tensors='pt', padding=True, truncation=True).to(quality_device)
            with torch.inference_mode():
                out = clip(**inputs)
                clean_clip.append(float((out.image_embeds * out.text_embeds).sum(-1).clamp_min(0).mul(2.5).item()))
            clean_pixels.append(teacher.pixels([clean_path], device=str(quality_device)))
            completed = i + 1
            if completed == 1 or completed % 25 == 0 or completed == len(prompts):
                print(progress_line('clean validation quality baseline', completed, len(prompts),
                                    time.perf_counter() - clean_metric_started), flush=True)

        trials = []
        for gamma in candidates:
            candidate_started = time.perf_counter()
            print(f'OGSC gamma tuning: candidate {len(trials) + 1}/{len(candidates)}, gamma={gamma:.6g} started.', flush=True)
            lpips_values, clip_values, mse_values = [], [], []
            trial_dir = output / 'calibration/gamma_trials' / f'{gamma:.10f}'
            trial_dir.mkdir(parents=True, exist_ok=True)
            for i, (row, z, direction) in enumerate(zip(prompts, clean_latents, directions)):
                marked = (z.float() + gamma * direction).to(dtype=z.dtype, device='cuda')
                image = _decode(decoder, marked)[0]
                png = trial_dir / image_name(i)
                _save_png(image, png)
                with Image.open(png) as im:
                    rgb = im.convert('RGB')
                inputs = processor(text=[row['text']], images=[rgb], return_tensors='pt', padding=True, truncation=True).to(quality_device)
                with torch.inference_mode():
                    out = clip(**inputs)
                    clip_values.append(float((out.image_embeds * out.text_embeds).sum(-1).clamp_min(0).mul(2.5).item()))
                    marked_pixels = teacher.pixels([png], device=str(quality_device))
                    lpips_values.append(float(perceptual(clean_pixels[i], marked_pixels).flatten()[0].item()))
                    mse_values.append(float((clean_pixels[i] - marked_pixels).square().mean().item()))
                if i == 0 or (i + 1) % 25 == 0 or i + 1 == len(prompts):
                    print(progress_message(f'gamma={gamma:.6g} quality decode/metrics', i + 1,
                                           len(prompts), time.perf_counter() - candidate_started), flush=True)
            clip_drop = float(np.mean(clean_clip) - np.mean(clip_values))
            mean_lpips, median_lpips = float(np.mean(lpips_values)), float(np.median(lpips_values))
            passed = mean_lpips <= .15 and clip_drop <= .01
            trial = {'gamma': gamma, 'mean_paired_lpips': mean_lpips, 'median_paired_lpips': median_lpips,
                     'max_paired_lpips': float(np.max(lpips_values)), 'mean_clip': float(np.mean(clip_values)),
                     'mean_clip_drop': clip_drop, 'mean_pixel_mse': float(np.mean(mse_values)),
                     'passes': passed, 'lpips_per_image': lpips_values, 'clip_per_image': clip_values,
                     'pixel_mse_per_image': mse_values}
            trials.append(trial)
            write_json(trial_dir / 'quality.json', trial)
        selected = select_gamma(trials)
        if selected is None:
            payload = {'status': 'NO_POSITIVE_GAMMA_PASSES', 'objective': OBJECTIVE,
                       'gamma': None, 'upper_bound': upper,
                       'trials': trials, 'limits': {'mean_paired_lpips': .15, 'mean_clip_drop': .01}}
            write_json(output / 'calibration/gamma.json', payload)
            raise RuntimeError('No positive gamma candidate passed validation quality constraints')
        payload = {'status': 'selected', 'objective': OBJECTIVE,
                   'gamma': selected, 'upper_bound': upper,
                   'selection': 'largest measured positive validation candidate passing mean LPIPS and CLIP drop',
                   'trials': trials, 'limits': {'mean_paired_lpips': .15, 'mean_clip_drop': .01},
                   'validation_prompts_sha256': sha256(source / 'prompts.json')}
        write_json(output / 'calibration/gamma.json', payload)
        return payload
    finally:
        del clip, processor, perceptual, decoder, clean_pixels
        torch.cuda.empty_cache()
