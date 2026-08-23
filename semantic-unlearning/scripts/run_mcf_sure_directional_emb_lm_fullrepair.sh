#!/usr/bin/env bash
set -euo pipefail

# Two-stage MCF SURE experiment:
#   Stage 1: untie Emb/LM, directional GA on target_true-sensitive rows.
#   Stage 2: direct-failure-only unrestricted sparse LM-head row repair.
#
# No LoRA. No rank sweep. No generated paraphrases.
# No official paraphrase/neighborhood/retain/PPL training visibility.
#
# Usage:
#   bash scripts/run_mcf_sure_directional_emb_lm_fullrepair.sh \
#     /path/to/model \
#     outputs/mcf_targettrue_clean_seed1/seed1/protocol/training_visible_mcf_target_true.json \
#     outputs/mcf_targettrue_clean_seed1/seed1/protocol/split_manifest.json

MODEL_PATH=${1:?model path required}
VISIBLE=${2:?training-visible JSON required}
MANIFEST=${3:?split manifest required}

SEED=${SEED:-1}
FORGET_NUM=${FORGET_NUM:-50}
OUT_ROOT=${OUT_ROOT:-outputs/mcf_directional_emb_lm_fullrepair_seed${SEED}}

STAGE1_STEPS=${STAGE1_STEPS:-600}
STAGE1_BATCH_SIZE=${STAGE1_BATCH_SIZE:-2}
STAGE1_CACHE_BATCH_SIZE=${STAGE1_CACHE_BATCH_SIZE:-8}
STAGE1_LR=${STAGE1_LR:-1e-4}
STAGE1_GA_WEIGHT=${STAGE1_GA_WEIGHT:-2.0}
STAGE1_KL_WEIGHT=${STAGE1_KL_WEIGHT:-1.0}
STAGE1_DELTA_L2=${STAGE1_DELTA_L2:-1e-6}
DIRECTION_RANK=${DIRECTION_RANK:-1}
CONSTRAINT_MARGIN=${CONSTRAINT_MARGIN:-0.05}

REPAIR_STEPS=${REPAIR_STEPS:-800}
REPAIR_LR=${REPAIR_LR:-0.005}
REPAIR_L2=${REPAIR_L2:-1e-6}
REPAIR_PASS_GUARD_WEIGHT=${REPAIR_PASS_GUARD_WEIGHT:-1.0}
REPAIR_BATCH_SIZE=${REPAIR_BATCH_SIZE:-8}
REPAIR_CHECK_EVERY=${REPAIR_CHECK_EVERY:-25}

DTYPE=${DTYPE:-bf16}
DEVICE_MAP=${DEVICE_MAP:-single}
CANDIDATE_SCALES=${CANDIDATE_SCALES:-1,.875,.75,.625,.5,.375,.25,.1875,.125,.09375,.0625,.046875,.03125,.015625,.0078125,0}

STAGE1_OUT="$OUT_ROOT/stage1"
STAGE2_OUT="$OUT_ROOT/stage2_fullrow_repair"

mkdir -p "$OUT_ROOT"

python -u scripts/mcf_sure_directional_emb_lm_stage1.py \
  --model-path "$MODEL_PATH" \
  --training-visible-path "$VISIBLE" \
  --split-manifest "$MANIFEST" \
  --output-dir "$STAGE1_OUT" \
  --seed "$SEED" \
  --forget-num "$FORGET_NUM" \
  --steps "$STAGE1_STEPS" \
  --batch-size "$STAGE1_BATCH_SIZE" \
  --cache-batch-size "$STAGE1_CACHE_BATCH_SIZE" \
  --lr "$STAGE1_LR" \
  --ga-weight "$STAGE1_GA_WEIGHT" \
  --distribution-kl-weight "$STAGE1_KL_WEIGHT" \
  --delta-l2 "$STAGE1_DELTA_L2" \
  --direction-rank "$DIRECTION_RANK" \
  --stage1-constraint-margin "$CONSTRAINT_MARGIN" \
  --candidate-scales "$CANDIDATE_SCALES" \
  --dtype "$DTYPE" \
  --device-map "$DEVICE_MAP" \
  2>&1 | tee "$OUT_ROOT/stage1.log"

python -u scripts/mcf_sure_fullrow_failure_repair.py \
  --model-path "$STAGE1_OUT/checkpoint" \
  --training-visible-path "$VISIBLE" \
  --split-manifest "$MANIFEST" \
  --output-dir "$STAGE2_OUT" \
  --seed "$SEED" \
  --forget-num "$FORGET_NUM" \
  --repair-steps "$REPAIR_STEPS" \
  --repair-lr "$REPAIR_LR" \
  --constraint-margin "$CONSTRAINT_MARGIN" \
  --repair-l2 "$REPAIR_L2" \
  --pass-guard-weight "$REPAIR_PASS_GUARD_WEIGHT" \
  --batch-size "$REPAIR_BATCH_SIZE" \
  --check-every "$REPAIR_CHECK_EVERY" \
  --candidate-scales "$CANDIDATE_SCALES" \
  --dtype "$DTYPE" \
  --device-map "$DEVICE_MAP" \
  2>&1 | tee "$OUT_ROOT/stage2_fullrow_repair.log"

printf '\nFinished directional Emb+LM -> unrestricted full-row LM-head repair.\n'
printf 'Stage 1: %s\n' "$STAGE1_OUT/checkpoint"
printf 'Final  : %s\n' "$STAGE2_OUT/checkpoint"
printf 'Summary: %s\n' "$STAGE2_OUT/repair_summary.json"
