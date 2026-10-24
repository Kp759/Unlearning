#!/usr/bin/env bash
# Ours: the paper table with ONE forward pass. The classifier reads layer 19 and Δe is added at
# layer 19 (last prompt token + every generated token); no logit lens; routers with 20 positives
# + 20 negatives per fact and no PCA. Same stages, evaluators and outputs as run_l16_paper_one.sh.
#   bash scripts/run_onepass_paper_one.sh DATASET SEED                   -> outputs/onepass_l19_v1/...
#   WRITE_LAYER=16 bash scripts/run_onepass_paper_one.sh DATASET SEED    # ablation: Δe at 16, two passes
# Other models: MODEL=qwen3_4b|llama31_8b with ONEPASS=1 through scripts/paper_model_config.sh.
set -euo pipefail
ONEPASS=1 exec bash "$(dirname "$0")/run_l16_paper_one.sh" "$@"
