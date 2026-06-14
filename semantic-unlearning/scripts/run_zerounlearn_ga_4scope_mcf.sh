#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL_PATH="${1:-${MODEL_PATH:-/scratch/yl258/kp759/hf/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/zerounlearn_ga_4scope_mcf}"
FORGET_NUM="${FORGET_NUM:-50}"
RETAIN_NUM="${RETAIN_NUM:-1000}"
SEED="${SEED:-1}"
STEPS="${STEPS:-25}"
LR="${LR:-5e-3}"
FULL_OPTIMIZER="${FULL_OPTIMIZER:-adamw8bit}"
EMB_OPTIMIZER="${EMB_OPTIMIZER:-adam}"
DTYPE="${DTYPE:-bf16}"
MCF_PATH="${MCF_PATH:-data/mcf/multi_counterfact.json}"
WIKIDATA_DIR="${WIKIDATA_DIR:-data/wikidata}"
OFFICIAL_DEVICE_MAP="${OFFICIAL_DEVICE_MAP:-auto}"

run_mode() {
  local mode="$1"
  local optimizer="$2"
  local out_dir="${OUTPUT_ROOT}/${mode}_run"
  python scripts/gagd_compare.py \
    --dataset mcf \
    --model-path "${MODEL_PATH}" \
    --output-dir "${out_dir}" \
    --mode "${mode}" \
    --forget-loss-type zerounlearn_ga \
    --forget-num "${FORGET_NUM}" \
    --retain-num "${RETAIN_NUM}" \
    --seed "${SEED}" \
    --steps "${STEPS}" \
    --lr "${LR}" \
    --weight-decay 0.0 \
    --forget-weight 1.0 \
    --retain-weight 0.0 \
    --batch-size 1 \
    --retain-batch-size 1 \
    --dtype "${DTYPE}" \
    --optimizer "${optimizer}" \
    --save-model
}

run_mode full_all_tokens "${FULL_OPTIMIZER}"
run_mode full_selective_tokens "${FULL_OPTIMIZER}"
run_mode emb_lm_all_tokens "${EMB_OPTIMIZER}"
run_mode emb_lm_selective_tokens "${EMB_OPTIMIZER}"

python scripts/run_same_mcf_eval.py \
  --model-dirs \
    base="${MODEL_PATH}" \
    full_all_tokens="${OUTPUT_ROOT}/full_all_tokens_run/full_all_tokens/checkpoint" \
    full_selective_tokens="${OUTPUT_ROOT}/full_selective_tokens_run/full_selective_tokens/checkpoint" \
    emb_lm_all_tokens="${OUTPUT_ROOT}/emb_lm_all_tokens_run/emb_lm_all_tokens/checkpoint" \
    emb_lm_selective_tokens="${OUTPUT_ROOT}/emb_lm_selective_tokens_run/emb_lm_selective_tokens/checkpoint" \
  --mcf-path "${MCF_PATH}" \
  --wikidata-dir "${WIKIDATA_DIR}" \
  --out-dir "${OUTPUT_ROOT}/same_eval_with_base" \
  --unlearn-num "${FORGET_NUM}" \
  --retain-num "${RETAIN_NUM}" \
  --seed "${SEED}" \
  --sample-mode official \
  --dtype "${DTYPE}" \
  --device-map "${OFFICIAL_DEVICE_MAP}"

echo "Final official-compatible table: ${OUTPUT_ROOT}/same_eval_with_base/official_eval_comparison.md"
