#!/usr/bin/env bash
set -euo pipefail

# End-to-end leak-free protected-subspace SURE for target_true-sensitive MCF.
#
# Usage:
#   OUT_ROOT=outputs/mcf_protected_subspace_seed1 \
#   bash scripts/run_mcf_sure_protected_subspace.sh \
#     /path/to/base-model \
#     /abs/path/training_visible_mcf_target_true.json \
#     /abs/path/split_manifest.json
#
# Optional environment overrides:
#   SEED=1 FORGET_NUM=50
#   STAGE1_STEPS=600 STAGE1_LR=1e-4
#   PROTECTED_RANK=32 SENSITIVE_RANK=4
#   STAGE2_STEPS=800 STAGE2_LR=5e-3 REPAIR_RANK=4
#   ATOMIC_MARGIN=0.05 PROTECTED_KL_MAX=0.05
#   RUN_OFFICIAL_EVAL=1 MCF_PATH=data/multi_counterfact.json WIKIDATA_DIR=data/wikidata

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 MODEL_PATH TRAINING_VISIBLE_PATH SPLIT_MANIFEST" >&2
  exit 2
fi

MODEL_PATH="$1"
TRAINING_VISIBLE_PATH="$2"
SPLIT_MANIFEST="$3"

SEED="${SEED:-1}"
FORGET_NUM="${FORGET_NUM:-50}"
OUT_ROOT="${OUT_ROOT:-outputs/mcf_protected_subspace_seed${SEED}}"

STAGE1_STEPS="${STAGE1_STEPS:-600}"
STAGE1_LR="${STAGE1_LR:-1e-4}"
STAGE1_BATCH_SIZE="${STAGE1_BATCH_SIZE:-2}"
CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-8}"
GA_WEIGHT="${GA_WEIGHT:-2.0}"
KL_WEIGHT="${KL_WEIGHT:-1.0}"
STAGE1_L2="${STAGE1_L2:-1e-6}"
PROTECTED_RANK="${PROTECTED_RANK:-32}"
SENSITIVE_RANK="${SENSITIVE_RANK:-4}"
ATOMIC_MARGIN="${ATOMIC_MARGIN:-0.05}"

STAGE2_STEPS="${STAGE2_STEPS:-800}"
STAGE2_LR="${STAGE2_LR:-5e-3}"
REPAIR_RANK="${REPAIR_RANK:-4}"
PROTECTED_KL_WEIGHT="${PROTECTED_KL_WEIGHT:-1.0}"
PROTECTED_KL_MAX="${PROTECTED_KL_MAX:-0.05}"
STAGE2_L2="${STAGE2_L2:-1e-6}"
CHECK_EVERY="${CHECK_EVERY:-25}"

DTYPE="${DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-single}"
RUN_OFFICIAL_EVAL="${RUN_OFFICIAL_EVAL:-1}"
MCF_PATH="${MCF_PATH:-data/multi_counterfact.json}"
WIKIDATA_DIR="${WIKIDATA_DIR:-data/wikidata}"

STAGE1_DIR="$OUT_ROOT/stage1_protected_subspace"
STAGE2_DIR="$OUT_ROOT/stage2_protected_subspace"

mkdir -p "$OUT_ROOT"

printf '\n=== STAGE 1: protected sensitive subspace Emb + LM GA ===\n'
python -u scripts/mcf_sure_protected_subspace_stage1.py \
  --model-path "$MODEL_PATH" \
  --training-visible-path "$TRAINING_VISIBLE_PATH" \
  --split-manifest "$SPLIT_MANIFEST" \
  --output-dir "$STAGE1_DIR" \
  --seed "$SEED" \
  --forget-num "$FORGET_NUM" \
  --steps "$STAGE1_STEPS" \
  --batch-size "$STAGE1_BATCH_SIZE" \
  --cache-batch-size "$CACHE_BATCH_SIZE" \
  --lr "$STAGE1_LR" \
  --ga-weight "$GA_WEIGHT" \
  --kl-weight "$KL_WEIGHT" \
  --delta-l2 "$STAGE1_L2" \
  --protected-rank "$PROTECTED_RANK" \
  --sensitive-rank "$SENSITIVE_RANK" \
  --atomic-margin "$ATOMIC_MARGIN" \
  --dtype "$DTYPE" \
  --device-map "$DEVICE_MAP" \
  2>&1 | tee "$OUT_ROOT/stage1.log"

STAGE1_CKPT="$STAGE1_DIR/checkpoint"
STAGE1_CONFIG="$STAGE1_DIR/config_used.json"

test -d "$STAGE1_CKPT"
test -f "$STAGE1_CONFIG"

printf '\n=== STAGE 2: failure residual minus protected-success subspace ===\n'
python -u scripts/mcf_sure_protected_subspace_stage2.py \
  --model-path "$STAGE1_CKPT" \
  --stage1-config-path "$STAGE1_CONFIG" \
  --training-visible-path "$TRAINING_VISIBLE_PATH" \
  --split-manifest "$SPLIT_MANIFEST" \
  --output-dir "$STAGE2_DIR" \
  --seed "$SEED" \
  --forget-num "$FORGET_NUM" \
  --repair-steps "$STAGE2_STEPS" \
  --repair-lr "$STAGE2_LR" \
  --atomic-margin "$ATOMIC_MARGIN" \
  --protected-rank "$PROTECTED_RANK" \
  --repair-rank "$REPAIR_RANK" \
  --protected-kl-weight "$PROTECTED_KL_WEIGHT" \
  --protected-kl-max "$PROTECTED_KL_MAX" \
  --delta-l2 "$STAGE2_L2" \
  --check-every "$CHECK_EVERY" \
  --cache-batch-size "$CACHE_BATCH_SIZE" \
  --dtype "$DTYPE" \
  --device-map "$DEVICE_MAP" \
  2>&1 | tee "$OUT_ROOT/stage2.log"

FINAL_CKPT="$STAGE2_DIR/checkpoint"
test -d "$FINAL_CKPT"

if [[ "$RUN_OFFICIAL_EVAL" == "1" ]]; then
  printf '\n=== OFFICIAL MCF EVAL: frozen final checkpoint only ===\n'
  python -u scripts/mcf_zero_unlearn_official_eval.py \
    --model-dir "$FINAL_CKPT" \
    --mcf-path "$MCF_PATH" \
    --wikidata-dir "$WIKIDATA_DIR" \
    --out "$OUT_ROOT/final_official_eval.json" \
    --unlearn-num "$FORGET_NUM" \
    --retain-num 1000 \
    --seed "$SEED" \
    --sample-mode official \
    --dtype "$DTYPE" \
    --device-map "$DEVICE_MAP" \
    2>&1 | tee "$OUT_ROOT/final_official_eval.log"
fi

printf '\nProtected-subspace SURE finished.\n'
printf 'Stage 1: %s\n' "$STAGE1_CKPT"
printf 'Final  : %s\n' "$FINAL_CKPT"
printf 'Root   : %s\n' "$OUT_ROOT"
