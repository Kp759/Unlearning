#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

MODEL_PATH="${1:-${MODEL_PATH:-/scratch/yl258/kp759/hf/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95}}"
OUT_ROOT="${OUT_ROOT:-outputs/gagd_compare}"
COMMON=(
  --model-path "${MODEL_PATH}"
  --steps 100
  --batch-size 1
  --retain-batch-size 1
  --lr 1e-5
  --dtype bf16
  --mode all
)

python scripts/gagd_compare.py \
  --dataset mcf \
  --forget-num 50 \
  --retain-num 1000 \
  --output-dir "${OUT_ROOT}/mcf" \
  "${COMMON[@]}"

if [[ "${RUN_TOFU:-0}" == "1" ]]; then
  python scripts/gagd_compare.py \
    --dataset tofu \
    --forget-split forget05 \
    --retain-split retain95 \
    --forget-num 50 \
    --retain-num 1000 \
    --output-dir "${OUT_ROOT}/tofu" \
    "${COMMON[@]}"
else
  echo "Skipping TOFU. Set RUN_TOFU=1 to run TOFU after MCF."
fi
