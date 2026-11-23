# OGSC-32

OGSC uses the original weight32 threshold-based 32-bit carrier and key, but controls each clean SD2.1 final latent with one backward of the independent bitwise softplus objective. For signed margins `r=s*q`, the loss is `mean(softplus(-r))` across all 32 coordinates. The update remains `z_wm = z - gamma * grad(L) / RMS(grad(L))`. It does not train teacher weights, use Adam, run iterative latent optimization, create a new key, or use a Gram matrix.

`prepare` verifies the original carrier and source modules, resolves the repository and immutable revision in the source `assets.json`, and reuses a matching local snapshot or downloads it with `huggingface_hub`. A hash mismatch is an error. Outputs are written to the separate `runs/ogsc32` directory by default. Override paths with the CLI options or matching `OGSC_*` environment variables.

## Stages

```bash
python -m ogsc.run prepare
python -m ogsc.run smoke
python -m ogsc.run tune-gamma
python -m ogsc.run generate-one --prompt "a lighthouse in a storm" --seed 42
python -m ogsc.run generate-teacher
python -m ogsc.run evaluate-teacher
python -m ogsc.run train-student
python -m ogsc.run evaluate-student
```

For the controlled first comparison, run `run_ogsc32_bitwise_validation.sh`. It uses fixed `gamma=0.156398773`, evaluates the 1,000 validation prompts, and compares new PNGs with the previous soft-min PNGs. It does not run gamma search or regenerate train images. Clean final latents and clean/reference PNGs are reused from the prior run. The new objective writes to a separate output directory; old soft-min latents, directions, gamma-trial PNGs, and output are never reused for the new controls.

The comparison scores final RGB8 PNGs through the original `key_metrics.aggregate_metrics` (group sizes 1/8/32/64/256, 100 partitions, seed 20261029). It reports exact 32/32 FKR, empirical FPR on the separate clean test controls, all 32 mean signed margins and per-bit accuracy, paired LPIPS, the original CLIP score/drop convention, and synchronized per-image control latency. Quality is reported at the fixed gamma; it is not tuned in this first run.

For the older full workflow, `run_ogsc32_all.sh` runs `python -m ogsc.run full --offload`. It prepares the pinned model, performs a CPU toy smoke test, evaluates gamma candidates, generates all three splits, and evaluates teacher PNGs. Existing soft-min output directories are rejected by the objective identity check; use the dedicated bitwise validation script for the controlled comparison.

Both launchers log stdout/stderr under `runs/<output>/logs/` and show periodic progress, elapsed time, and ETA for long loops. The validation comparison resumes from its own `runs/ogsc32_bitwise_validation` output directory; do not delete it if you want to resume.

For direct inference after `prepare` and `tune-gamma`, `generate-one` samples the clean SD2.1 final latent for the prompt/seed, applies the one-step controller immediately, and decodes only the OGSC PNG. It does not write a clean PNG or generate a clean dataset first. This low-latency path keeps the FP16 sampling pipeline and FP32 semantic scorer resident together; use it on a GPU with enough VRAM. The full benchmark path is phase-separated and also writes clean PNGs because it needs paired quality and clean-teacher FKR references.

The separate `tune-gamma` command remains available for later quality tuning. It evaluates candidates with the current bitwise objective and records the objective identity. The first controlled comparison deliberately does not invoke it.

`smoke` is CPU-only and uses a tiny linear model. The other model stages require server CUDA and the pinned source/runtime modules. No GPU/model execution should be done on a constrained local workstation. The exact 12,500-step student bundle is present at `structure_student12500/campaign.py`; its README requires a separate completed `structure_margin_native` data run with thresholds, validation cache, and the historical teacher adapter. That run is not included here, so the full command records `STUDENT_SOURCE_ARTIFACTS_MISSING` and does not start the campaign or use the old non-OGSC teacher images. Teacher results do not establish student transfer or research success.

## 64-image direction diagnostic

Run `run_ogsc32_direction_probe.sh` after the fixed-gamma bitwise comparison completes. This diagnostic samples 64 evenly spaced validation indices, reuses cached clean and bitwise FP16 latents plus existing clean/bitwise coordinate score files, and estimates the old update direction from their latent difference. It computes `gamma_small=gamma_old/16`, decodes only those 64 small-gamma latents with the native SD2.1 VAE, then compares FP32 surrogate margins with scores from the decoded PNGs. It does not alter `control.py`, run diffusion sampling, tune a new gamma, rescore the full validation set, train weights, or run CLIP/FID/KID. Missing marked-latent records are recovered with one `one_step` call only for the affected probe samples.

Results are written to `runs/ogsc32_bitwise_validation/result/direction_probe.md` and `.json`; small-gamma PNGs, per-image latent records, and resumable PNG scores remain under the same run. The command logs progress and ETA to the console and `runs/ogsc32_bitwise_validation/logs/direction_probe.log`. FKR is reported only for group sizes 1, 8, 32, and 64 and is explicitly marked as a small-sample diagnostic.
