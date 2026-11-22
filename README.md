# OGSC-32 (notrain)

This repository contains the OGSC-32 implementation and its CPU/tiny-model source tests. The teacher weights are not trained or modified. The full teacher pipeline samples clean SD2.1 latents, applies one soft-min semantic-control step, decodes clean/OGSC benchmark images, and evaluates the saved outputs.

## Run

From this repository root, run:

```bash
bash run_ogsc32_all.sh
```

The script contains only the Python pipeline command. It resumes from the configured output directory when rerun.

## Existing source dependencies

The frozen 32-bit carrier, original semantic harness, key-metrics implementation, and oracle helpers are intentionally not copied here. The runner reuses those existing source trees and validates their required files and carrier hashes before starting. Set paths through the command-line options to `python -m ogsc.run` or matching `OGSC_*` environment variables when the default sibling layout does not apply:

- `OGSC_WEIGHT32_REPO`: root containing `BEST_TEACHER_32bits_refine/` and the original carrier run.
- `OGSC_ORACLE_REPO`: root containing the existing `Malicious_quantization/bitmark_lora/` oracle helpers (or the helper files directly).
- `OGSC_SOURCE_RUN`, `OGSC_CARRIER_DIR`, `OGSC_OUTPUT_RUN`: optional source, carrier, and output overrides.

The pinned SD2.1 model is resolved from the validated local snapshot or downloaded from Hugging Face by the Python pipeline. No model weights, generated images, run cache, or credentials are tracked by Git.

See [`ogsc/README.md`](ogsc/README.md) for stages, resume behavior, gamma selection, and evaluation details, and [`Codex_OGSC_32bit_Implementation_Spec.md`](Codex_OGSC_32bit_Implementation_Spec.md) for the implementation contract.
