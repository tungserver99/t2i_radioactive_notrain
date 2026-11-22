# OGSC-32

OGSC uses the original weight32 threshold-based 32-bit carrier and key, but controls each clean SD2.1 final latent with one gradient of the 32-coordinate soft-min objective. It does not train teacher weights, use Adam, run a denoising-loop gradient, create a new key, or depend on threshold/clean-score calibration files or a Gram matrix.

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

For the complete teacher pipeline in one server command, run `run_ogsc32_all.sh` from the repository root. The script contains only `python -m ogsc.run full --offload`. The full command prepares the pinned model, performs a CPU toy smoke test, evaluates bounded gamma candidates on paired validation images, creates clean and OGSC teacher images for all three splits, then scores saved PNGs with the original `key_metrics.aggregate_metrics` (sizes 1/8/32/64/256, 100 partitions, seed 20261029). It also calls the original paired quality utility for CLIP/FID/KID when `torch-fidelity` is installed.

The `full` command resumes from the same output directory: it preserves the run manifest and stage state, reuses a positive validation gamma when the source prompts match, and skips completed per-image PNG/latent artifacts. Re-run the same `.sh` after interruption; do not delete `runs/ogsc32`. If the source/carrier identity changed, or a gamma artifact is missing after teacher generation started, it stops with a clear error instead of mixing outputs or silently selecting a new gamma.

For direct inference after `prepare` and `tune-gamma`, `generate-one` samples the clean SD2.1 final latent for the prompt/seed, applies the one-step controller immediately, and decodes only the OGSC PNG. It does not write a clean PNG or generate a clean dataset first. This low-latency path keeps the FP16 sampling pipeline and FP32 semantic scorer resident together; use it on a GPU with enough VRAM. The full benchmark path is phase-separated and also writes clean PNGs because it needs paired quality and clean-teacher FKR references.

Gamma candidates are bounded by the observed median RMS of clean validation latents; one-step directions are computed once and reused across candidates. The selected value is the largest measured positive candidate passing mean paired LPIPS <= 0.15 and aggregate CLIP drop <= 0.01. The full trial distributions and per-image LPIPS, CLIP, and pixel MSE are saved. If none passes, gamma tuning fails explicitly.

`smoke` is CPU-only and uses a tiny linear model. The other model stages require server CUDA and the pinned source/runtime modules. No GPU/model execution should be done on a constrained local workstation. The exact 12,500-step student bundle is present at `structure_student12500/campaign.py`; its README requires a separate completed `structure_margin_native` data run with thresholds, validation cache, and the historical teacher adapter. That run is not included here, so the full command records `STUDENT_SOURCE_ARTIFACTS_MISSING` and does not start the campaign or use the old non-OGSC teacher images. Teacher results do not establish student transfer or research success.
