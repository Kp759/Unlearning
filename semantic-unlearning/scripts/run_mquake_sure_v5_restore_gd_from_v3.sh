#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BASE_MODEL="${1:?Usage: bash scripts/run_mquake_sure_v5_restore_gd_from_v3.sh BASE_MODEL [V3_OUTPUT_ROOT] [MQUAKE_JSON]}"
V3_ROOT="${2:-outputs/mquake_pure_two_stage_directional_v3_3b}"
MQUAKE="${3:-data/MQuAKE-CF-3k-v2.json}"
WIKIDATA_DIR="${WIKIDATA_DIR:-data/wikidata}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/mquake_sure_v5_restore_gd_3b}"
SEEDS_TEXT="${MQUAKE_SEEDS:-1}"
SOURCE_FORGET="${MQUAKE_FORGET_NUM:-50}"
RETAIN_NUM="${MQUAKE_RETAIN_EVAL_NUM:-1000}"
DTYPE="${DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-single}"
CACHE_BATCH="${SURE_CACHE_BATCH_SIZE:-8}"
MARGIN="${MQUAKE_SURE_CONSTRAINT_MARGIN:-0.05}"
MAX_PKL="${SURE_MAX_PROTECTED_KL:-0.05}"
EVAL_BATCH="${MQUAKE_EVAL_BATCH_SIZE:-8}"

GD_STEPS="${SURE_V5_GD_STEPS:-200}"
GD_LR="${SURE_V5_GD_LR:-0.00005}"
GD_BATCH="${SURE_V5_GD_BATCH_SIZE:-4}"
GD_ANCHOR="${SURE_V5_GD_ANCHOR_WEIGHT:-0.001}"
GD_CHECK="${SURE_V5_GD_CHECK_EVERY:-25}"

REPAIR_STEPS="${SURE_V5_REPAIR_STEPS:-800}"
REPAIR_LR="${SURE_V5_REPAIR_LR:-0.0005}"
REPAIR_BATCH="${SURE_V5_REPAIR_BATCH_SIZE:-8}"
REPAIR_CHECK="${SURE_V5_REPAIR_CHECK_EVERY:-25}"
REPAIR_L2="${SURE_V5_REPAIR_L2_WEIGHT:-0.000001}"
BACKTRACK="${SURE_V5_BACKTRACK_SCALES:-0.5,0.25,0.125,0.0625,0.03125,0.015625}"

read -r -a SEEDS <<< "${SEEDS_TEXT}"
test -d "${WIKIDATA_DIR}"

for SEED in "${SEEDS[@]}"; do
  SRC="${V3_ROOT}/seed${SEED}"
  PROTOCOL="${SRC}/protocol"
  VISIBLE="${PROTOCOL}/training_visible_forget.json"
  MANIFEST="${PROTOCOL}/split_manifest.json"
  STAGE1="${SRC}/level1_full_residual_directional_ga/checkpoint"

  test -d "${STAGE1}"
  test -f "${VISIBLE}"
  test -f "${MANIFEST}"

  ROOT="${OUTPUT_ROOT}/seed${SEED}"
  GD_OUT="${ROOT}/stage2a_restore_embedding_gd"
  HEAD_OUT="${ROOT}/stage2b_stable_head_nullspace"
  SUMMARY="${HEAD_OUT}/two_stage_summary.json"
  FINAL="${ROOT}/official_eval_with_atomicgen.json"
  EVAL_MANIFEST="${ROOT}/final_eval_split_manifest.json"
  mkdir -p "${ROOT}"
  rm -rf "${GD_OUT}" "${HEAD_OUT}"

  DIRECT_COUNT="$(python -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["sampling"]["forget_atomic_fact_count"])' "${MANIFEST}")"

  echo "===== SURE v5 SEED ${SEED}: STAGE2A RESTORE E TO BASE + NON-SENSITIVE GD ====="
  echo "      source Stage1: ${STAGE1}"
  python scripts/mquake_sure_stage2_restore_embedding_gd_v5.py \
    --base-model-path "${BASE_MODEL}" \
    --model-path "${STAGE1}" \
    --training-visible-path "${VISIBLE}" \
    --split-manifest "${MANIFEST}" \
    --output-dir "${GD_OUT}" \
    --seed "${SEED}" \
    --forget-num "${DIRECT_COUNT}" \
    --steps "${GD_STEPS}" \
    --batch-size "${GD_BATCH}" \
    --cache-batch-size "${CACHE_BATCH}" \
    --learning-rate "${GD_LR}" \
    --anchor-weight "${GD_ANCHOR}" \
    --check-every "${GD_CHECK}" \
    --constraint-margin "${MARGIN}" \
    --optimizer adamw \
    --dtype "${DTYPE}" \
    --device-map "${DEVICE_MAP}"

  echo "===== SURE v5 SEED ${SEED}: STAGE2B RECOMPUTE H + STABLE W-ONLY NULLSPACE REPAIR ====="
  python scripts/mquake_sure_stage2_head_nullspace_v5.py \
    --model-path "${GD_OUT}/checkpoint" \
    --training-visible-path "${VISIBLE}" \
    --split-manifest "${MANIFEST}" \
    --output-dir "${HEAD_OUT}" \
    --seed "${SEED}" \
    --forget-num "${DIRECT_COUNT}" \
    --repair-steps "${REPAIR_STEPS}" \
    --repair-lr "${REPAIR_LR}" \
    --batch-size "${REPAIR_BATCH}" \
    --cache-batch-size "${CACHE_BATCH}" \
    --check-every "${REPAIR_CHECK}" \
    --constraint-margin "${MARGIN}" \
    --max-protected-kl "${MAX_PKL}" \
    --l2-weight "${REPAIR_L2}" \
    --backtrack-scales "${BACKTRACK}" \
    --dtype "${DTYPE}" \
    --device-map "${DEVICE_MAP}"

  FINAL_PASS="$(python -c 'import json,sys; d=json.load(open(sys.argv[1])); print("1" if d.get("final_gates_pass") else "0")' "${SUMMARY}")"
  if [[ "${FINAL_PASS}" != "1" ]]; then
    echo "===== OFFICIAL EVAL SKIPPED: v5 FINAL DIRECT TRAINING GATE FAILED ====="
    continue
  fi

  echo "===== SURE v5 SEED ${SEED}: OFFICIAL HELD-OUT MQUAKE EVAL ====="
  python scripts/mquake_zero_unlearn_official_eval.py \
    --model-dir "${HEAD_OUT}/checkpoint" \
    --mquake-path "${MQUAKE}" \
    --wikidata-dir "${WIKIDATA_DIR}" \
    --out "${FINAL}" \
    --split-manifest "${EVAL_MANIFEST}" \
    --method "MQuAKE SURE v5 Restore-E + Sparse Embedding GD + Stable Head Nullspace" \
    --unlearn-num "${SOURCE_FORGET}" \
    --retain-num "${RETAIN_NUM}" \
    --seed "${SEED}" \
    --batch-size "${EVAL_BATCH}" \
    --dtype "${DTYPE}" \
    --device-map "${DEVICE_MAP}"

  python scripts/annotate_ppl_provenance.py \
    --eval-json "${FINAL}" \
    --model-dir "${HEAD_OUT}/checkpoint" \
    --wikidata-dir "${WIKIDATA_DIR}"
done

echo "SURE v5 restore-E + GD experiment complete: ${OUTPUT_ROOT}"
