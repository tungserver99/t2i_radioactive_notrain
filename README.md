# OGSC-32 (notrain)

This repository contains the OGSC-32 implementation and its CPU/tiny-model source tests. The teacher weights are not trained or modified. The current controlled validation experiment compares the prior soft-min PNGs with a new independent bitwise softplus objective.

## Run the 1,000-prompt validation comparison

From this repository root, run:

```bash
bash run_ogsc32_bitwise_validation.sh
```

The script contains only a Python command. The first comparison uses fixed `gamma=0.156398773`; it does not search gamma or generate the 5,000-image training split. It reuses the previous run's clean validation latents, clean PNGs, soft-min PNGs, and completed score files when their input identities match. Empirical FPR is measured separately on the 1,000 clean test controls. New bitwise outputs and results go to `runs/ogsc32_bitwise_validation`, leaving the previous experiment untouched. Rerun the same script to resume.

Every command writes live progress to the console and appends stdout/stderr to `runs/<output>/logs/<command>.log`. Long stages emit periodic heartbeats; per-image generation, scoring, and quality evaluation report counts, elapsed time, and ETA.

## Existing source dependencies

The frozen 32-bit carrier, original semantic harness, key-metrics implementation, and oracle helpers are intentionally not copied here. The runner reuses those existing source trees and validates their required files and carrier artifacts before starting. It searches parent folders for the weight32 checkout and `Hmark_inmodel`. Set paths through command-line options to `python -m ogsc.run` or matching `OGSC_*` environment variables when the default sibling layout does not apply:

- `OGSC_WEIGHT32_REPO`: root containing `BEST_TEACHER_32bits_refine/` and the original carrier run.
- `OGSC_ORACLE_REPO`: root containing the existing `Malicious_quantization/bitmark_lora/` oracle helpers (or the helper files directly).
- `OGSC_SOURCE_RUN`, `OGSC_CARRIER_DIR`, `OGSC_OUTPUT_RUN`, `OGSC_BASELINE_RUN`: optional source, carrier, new-output, and old-comparison-run overrides.

The pinned SD2.1 model is resolved from the validated local snapshot or downloaded from Hugging Face by the Python pipeline. No model weights, generated images, run cache, or credentials are tracked by Git.

See [`ogsc/README.md`](ogsc/README.md) for stages, resume behavior, gamma selection, and evaluation details, and [`Codex_OGSC_32bit_Implementation_Spec.md`](Codex_OGSC_32bit_Implementation_Spec.md) for the implementation contract.
