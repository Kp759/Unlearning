#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL_PATH="${1:-${MODEL_PATH:-/scratch/yl258/kp759/hf/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95}}"
OUT_ROOT="${OUT_ROOT:-outputs/gagd_vs_json_lmhead}"
SEEDS="${SEEDS:-0 1 2 3 4}"
FORGET_NUM="${FORGET_NUM:-50}"
RETAIN_NUM="${RETAIN_NUM:-1000}"
STEPS="${STEPS:-100}"
LR="${LR:-1e-5}"
FORGET_WEIGHT="${FORGET_WEIGHT:-1.0}"
RETAIN_WEIGHT="${RETAIN_WEIGHT:-1.0}"
DTYPE="${DTYPE:-bf16}"
MCF_PATH="${MCF_PATH:-data/multi_counterfact.json}"
WIKIDATA_DIR="${WIKIDATA_DIR:-data/wikidata}"
BASELINE_PATTERN="${BASELINE_PATTERN:-outputs/official_eval_lmhead_zero_true_restore150_seed{seed}_spefix.json}"
BASE_PATTERN="${BASE_PATTERN:-outputs/official_eval_base_seed{seed}_spefix.json}"

read -r -a SEED_ARRAY <<< "${SEEDS}"

for seed in "${SEED_ARRAY[@]}"; do
  python scripts/gagd_compare.py \
    --dataset mcf \
    --model-path "${MODEL_PATH}" \
    --mcf-cache-path "${MCF_PATH}" \
    --mcf-sample-mode official \
    --official-sample-mode official \
    --output-dir "${OUT_ROOT}/seed${seed}" \
    --mode all \
    --forget-loss-type answer_nll \
    --mcf-answer-field target_new \
    --forget-num "${FORGET_NUM}" \
    --retain-num "${RETAIN_NUM}" \
    --seed "${seed}" \
    --steps "${STEPS}" \
    --batch-size 1 \
    --retain-batch-size 1 \
    --lr "${LR}" \
    --forget-weight "${FORGET_WEIGHT}" \
    --retain-weight "${RETAIN_WEIGHT}" \
    --dtype "${DTYPE}" \
    --wikidata-dir "${WIKIDATA_DIR}" \
    --run-official-mcf-eval
done

python scripts/compare_gagd_to_json_lmhead.py \
  --seeds "${SEED_ARRAY[@]}" \
  --gagd-pattern "${OUT_ROOT}/seed{seed}/official_eval/{method}_official_eval.json" \
  --baseline-pattern "${BASELINE_PATTERN}" \
  --base-pattern "${BASE_PATTERN}" \
  --output-dir "${OUT_ROOT}/comparison" \
  --unlearn-num "${FORGET_NUM}" \
  --retain-num "${RETAIN_NUM}"

echo "Final aggregate table: ${OUT_ROOT}/comparison/aggregate.md"
