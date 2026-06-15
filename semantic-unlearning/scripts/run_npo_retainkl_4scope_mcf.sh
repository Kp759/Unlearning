#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

DEFAULT_MODEL_PATH="/scratch/yl258/kp759/hf/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95"
MODEL_PATH="${1:-${MODEL_PATH:-${DEFAULT_MODEL_PATH}}}"
OUT_ROOT="outputs/npo_retainkl_4scope_mcf"
MCF_PATH="semantic-unlearning/data/mcf/multi_counterfact.json"

FULL_OPTIMIZER="${FULL_OPTIMIZER:-adamw8bit}"
EMB_OPTIMIZER="${EMB_OPTIMIZER:-adam}"
FULL_LR="${FULL_LR:-1e-5}"
EMB_LR="${EMB_LR:-5e-5}"
REFERENCE_DEVICE="${REFERENCE_DEVICE:-auto}"
STEPS="${STEPS:-300}"
SEED="${SEED:-1}"

COMMON=(
  --model-path "${MODEL_PATH}"
  --mcf-path "${MCF_PATH}"
  --forget-num 50
  --retain-num 1000
  --steps "${STEPS}"
  --seed "${SEED}"
  --reference-device "${REFERENCE_DEVICE}"
  --dtype bf16
  --device-map single
  --save-model
  --skip-ppl
)

run_mode() {
  local mode="$1"
  local optimizer lr
  if [[ "${mode}" == full_* ]]; then
    optimizer="${FULL_OPTIMIZER}"
    lr="${FULL_LR}"
  else
    optimizer="${EMB_OPTIMIZER}"
    lr="${EMB_LR}"
  fi
  python semantic-unlearning/scripts/npo_retainkl_compare.py \
    "${COMMON[@]}" \
    --output-dir "${OUT_ROOT}" \
    --mode "${mode}" \
    --optimizer "${optimizer}" \
    --lr "${lr}"
}

run_mode full_all_tokens
run_mode full_selective_tokens
run_mode emb_lm_all_tokens
run_mode emb_lm_selective_tokens

python semantic-unlearning/scripts/run_same_mcf_eval.py \
  --model-dirs \
    "base=${MODEL_PATH}" \
    "full_all_tokens=${OUT_ROOT}/full_all_tokens/checkpoint" \
    "full_selective_tokens=${OUT_ROOT}/full_selective_tokens/checkpoint" \
    "emb_lm_all_tokens=${OUT_ROOT}/emb_lm_all_tokens/checkpoint" \
    "emb_lm_selective_tokens=${OUT_ROOT}/emb_lm_selective_tokens/checkpoint" \
  --mcf-path "${MCF_PATH}" \
  --out-dir "${OUT_ROOT}/same_eval_with_base" \
  --unlearn-num 50 \
  --retain-num 1000 \
  --seed "${SEED}" \
  --sample-mode official \
  --dtype bfloat16 \
  --device-map auto \
  --skip-ppl

echo "Final table: ${OUT_ROOT}/same_eval_with_base/official_eval_comparison.md"
