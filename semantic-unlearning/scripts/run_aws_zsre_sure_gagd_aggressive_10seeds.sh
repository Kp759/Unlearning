#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL="${MODEL:-/home/ec2-user/models/Llama-3.2-3B-Instruct}"
ZSRE_JSON="${ZSRE_JSON:-data/zsre_mend_eval.json}"
WIKIDATA_DIR="${WIKIDATA_DIR:-data/wikidata_aws_diag}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/aws_zsre_sure_no_neutral_gagd_aggressive_seeds1_10_3b}"

export MODEL ZSRE_JSON WIKIDATA_DIR OUTPUT_ROOT
export ZSRE_FORGET_NUM=50
export ZSRE_RETAIN_EVAL_NUM=1000

# Frozen aggressive Stage 1.
export ZSRE_STEPS=600
export ZSRE_BATCH_SIZE=1
export ZSRE_CACHE_BATCH_SIZE=8
export ZSRE_EMB_LM_LR=0.0001
export ZSRE_GA_WEIGHT=2.0
export ZSRE_GD_WEIGHT=1.0

# Frozen Stage 2.
export REPAIR_STEPS=800
export REPAIR_LR=0.005
export REPAIR_MARGIN=0.05
export REPAIR_L2=0.000001
export REPAIR_BATCH_SIZE=8

# Evaluation.
export EVAL_BATCH_SIZE=8
export DTYPE=bf16
export DEVICE_MAP=single
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

test -d "${MODEL}"
test -f "${ZSRE_JSON}"
test -d "${WIKIDATA_DIR}"

mkdir -p "${OUTPUT_ROOT}"

for SEED in 1 2 3 4 5 6 7 8 9 10; do
  export ZSRE_SEEDS="${SEED}"
  SEED_ROOT="${OUTPUT_ROOT}/seed${SEED}"

  echo "============================================================"
  echo "AWS ZsRE SURE aggressive GA/GD — seed ${SEED}/10"
  echo "output: ${SEED_ROOT}"
  echo "============================================================"

  bash scripts/run_zsre_sure_no_neutral_gagd_aggressive.sh \
    "${MODEL}" "${ZSRE_JSON}"

  # Only delete checkpoints after both official evaluations and repair summary exist.
  test -f "${SEED_ROOT}/official_eval_locked.json"
  test -f "${SEED_ROOT}/official_eval_stage1_posthoc.json"
  test -f "${SEED_ROOT}/stage2_sensitive_row_repair/repair_summary.json"

  find "${SEED_ROOT}" -type d -name checkpoint -prune -exec rm -rf {} +
  echo "seed ${SEED} complete; checkpoints removed"
done

python scripts/aggregate_zsre_sure_gagd_10seeds.py \
  --root "${OUTPUT_ROOT}" \
  --seeds 1-10 \
  --require-all \
  --output-prefix aggregate_10seeds

echo "===== 10-SEED AGGREGATE ====="
cat "${OUTPUT_ROOT}/aggregate_10seeds.md"
