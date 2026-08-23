#!/usr/bin/env bash
set -euo pipefail

# Protected-subspace SURE v2 for target_true-sensitive MCF.
#
# Stage 1:
#   B_NS = rowspace(preceding direct context hidden states), rank 32
#   B_S  = rowspace(H_S - Proj_BNS(H_S)), rank 4
#   Delta E_A = C_E B_S; Delta W_A = C_W B_S
#   2x GA(target_true) + full-vocab KL(Base||Edited) + tiny L2
#
# Stage 2:
#   P/F from atomic direct target_true margin max_other-target_true >= .05
#   B_P = rowspace(H_P), rank 32
#   B_F = rowspace(H_F - Proj_BP(H_F)), rank 4
#   LM-head only: Delta W_AF = C_F B_F
#   squared hinge(F) + KL(P) + tiny L2
#   hard P-regression=0 and KL(P)<=.05 with step backtracking
#
# Official MCF eval is run only after the frozen final gate passes.
# No LoRA. No official paraphrase/neighborhood/retain/PPL visibility in training.
#
# Usage:
#   bash scripts/run_mcf_sure_protected_subspace_v2.sh MODEL VISIBLE MANIFEST

MODEL_PATH=${1:?model path required}
VISIBLE=${2:?training-visible JSON required}
MANIFEST=${3:?split manifest required}

SEED=${SEED:-1}
FORGET_NUM=${FORGET_NUM:-50}
RETAIN_NUM=${RETAIN_NUM:-1000}
OUT_ROOT=${OUT_ROOT:-outputs/mcf_protected_subspace_v2_seed${SEED}}

STAGE1_STEPS=${STAGE1_STEPS:-600}
STAGE1_BATCH_SIZE=${STAGE1_BATCH_SIZE:-2}
CACHE_BATCH_SIZE=${CACHE_BATCH_SIZE:-8}
STAGE1_LR=${STAGE1_LR:-0.0001}
GA_WEIGHT=${GA_WEIGHT:-2.0}
STAGE1_KL_WEIGHT=${STAGE1_KL_WEIGHT:-1.0}
STAGE1_L2=${STAGE1_L2:-1e-6}
STAGE1_PROTECTED_RANK=${STAGE1_PROTECTED_RANK:-32}
SENSITIVE_RANK=${SENSITIVE_RANK:-4}
ATOMIC_MARGIN=${ATOMIC_MARGIN:-0.05}

REPAIR_STEPS=${REPAIR_STEPS:-800}
REPAIR_LR=${REPAIR_LR:-0.005}
STAGE2_PROTECTED_RANK=${STAGE2_PROTECTED_RANK:-32}
REPAIR_RANK=${REPAIR_RANK:-4}
PROTECTED_KL_WEIGHT=${PROTECTED_KL_WEIGHT:-1.0}
PROTECTED_KL_MAX=${PROTECTED_KL_MAX:-0.05}
REPAIR_L2=${REPAIR_L2:-1e-6}
REPAIR_CHECK_EVERY=${REPAIR_CHECK_EVERY:-25}

DTYPE=${DTYPE:-bf16}
DEVICE_MAP=${DEVICE_MAP:-single}
MCF_PATH=${MCF_PATH:-data/multi_counterfact.json}
WIKIDATA_DIR=${WIKIDATA_DIR:-data/wikidata}
RUN_OFFICIAL_EVAL=${RUN_OFFICIAL_EVAL:-1}

STAGE1_OUT="$OUT_ROOT/stage1_protected_subspace"
STAGE2_OUT="$OUT_ROOT/stage2_protected_subspace"
STAGE1_CONFIG="$STAGE1_OUT/stage1_config.json"
FINAL_CKPT="$STAGE2_OUT/checkpoint"

mkdir -p "$OUT_ROOT"

python -u scripts/mcf_sure_protected_subspace_stage1.py \
  --model-path "$MODEL_PATH" \
  --training-visible-path "$VISIBLE" \
  --split-manifest "$MANIFEST" \
  --output-dir "$STAGE1_OUT" \
  --seed "$SEED" \
  --forget-num "$FORGET_NUM" \
  --steps "$STAGE1_STEPS" \
  --batch-size "$STAGE1_BATCH_SIZE" \
  --cache-batch-size "$CACHE_BATCH_SIZE" \
  --lr "$STAGE1_LR" \
  --ga-weight "$GA_WEIGHT" \
  --kl-weight "$STAGE1_KL_WEIGHT" \
  --delta-l2 "$STAGE1_L2" \
  --protected-rank "$STAGE1_PROTECTED_RANK" \
  --sensitive-rank "$SENSITIVE_RANK" \
  --atomic-margin "$ATOMIC_MARGIN" \
  --dtype "$DTYPE" \
  --device-map "$DEVICE_MAP" \
  2>&1 | tee "$OUT_ROOT/stage1.log"

test -d "$STAGE1_OUT/checkpoint"
test -f "$STAGE1_CONFIG"

python -u scripts/mcf_sure_protected_subspace_stage2.py \
  --model-path "$STAGE1_OUT/checkpoint" \
  --stage1-config-path "$STAGE1_CONFIG" \
  --training-visible-path "$VISIBLE" \
  --split-manifest "$MANIFEST" \
  --output-dir "$STAGE2_OUT" \
  --seed "$SEED" \
  --forget-num "$FORGET_NUM" \
  --repair-steps "$REPAIR_STEPS" \
  --repair-lr "$REPAIR_LR" \
  --atomic-margin "$ATOMIC_MARGIN" \
  --protected-rank "$STAGE2_PROTECTED_RANK" \
  --repair-rank "$REPAIR_RANK" \
  --protected-kl-weight "$PROTECTED_KL_WEIGHT" \
  --protected-kl-max "$PROTECTED_KL_MAX" \
  --delta-l2 "$REPAIR_L2" \
  --check-every "$REPAIR_CHECK_EVERY" \
  --cache-batch-size "$CACHE_BATCH_SIZE" \
  --dtype "$DTYPE" \
  --device-map "$DEVICE_MAP" \
  2>&1 | tee "$OUT_ROOT/stage2.log"

test -d "$FINAL_CKPT"
test -f "$STAGE2_OUT/repair_summary.json"

if [[ "$RUN_OFFICIAL_EVAL" == "1" ]]; then
  test -f "$MCF_PATH"
  test -d "$WIKIDATA_DIR"
  python -u scripts/mcf_zero_unlearn_official_eval.py \
    --model-dir "$FINAL_CKPT" \
    --mcf-path "$MCF_PATH" \
    --wikidata-dir "$WIKIDATA_DIR" \
    --out "$OUT_ROOT/final_official_eval.json" \
    --unlearn-num "$FORGET_NUM" \
    --retain-num "$RETAIN_NUM" \
    --seed "$SEED" \
    --sample-mode official \
    --dtype "$DTYPE" \
    --device-map "$DEVICE_MAP" \
    2>&1 | tee "$OUT_ROOT/final_official_eval.log"
fi

printf '\nProtected-subspace SURE v2 complete.\n'
printf 'Stage 1: %s\n' "$STAGE1_OUT/checkpoint"
printf 'Final  : %s\n' "$FINAL_CKPT"
printf 'Summary: %s\n' "$STAGE2_OUT/repair_summary.json"
if [[ "$RUN_OFFICIAL_EVAL" == "1" ]]; then
  printf 'Official eval: %s\n' "$OUT_ROOT/final_official_eval.json"
fi
