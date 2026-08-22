#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:?Usage: bash scripts/run_mquake_pure_two_stage_sure.sh MODEL [MQUAKE_JSON]}"
MQUAKE="${2:-data/MQuAKE-CF-3k-v2.json}"
WIKIDATA_DIR="${WIKIDATA_DIR:-data/wikidata}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/mquake_pure_two_stage_sure_3b}"
SEEDS_TEXT="${MQUAKE_SEEDS:-1 2 3 4 5 6 7 8 9 10}"
FORGET_INSTANCES="${MQUAKE_FORGET_NUM:-50}"; RETAIN_INSTANCES="${MQUAKE_RETAIN_EVAL_NUM:-1000}"
DTYPE="${DTYPE:-bf16}"; DEVICE_MAP="${DEVICE_MAP:-single}"
STAGE1_STEPS="${SURE_STAGE1_STEPS:-600}"; STAGE1_LR="${SURE_STAGE1_LR:-0.0001}"
STAGE1_BATCH="${SURE_STAGE1_BATCH_SIZE:-1}"; CACHE_BATCH="${SURE_CACHE_BATCH_SIZE:-8}"
STAGE2_STEPS="${SURE_STAGE2_STEPS:-800}"; STAGE2_LR="${SURE_STAGE2_LR:-0.0005}"
STAGE2_BATCH="${SURE_STAGE2_BATCH_SIZE:-8}"; CHECK_EVERY="${SURE_CHECK_EVERY:-25}"
LAMBDA_F="${SURE_LAMBDA_F:-2.0}"; LAMBDA_P="${SURE_LAMBDA_P:-1.0}"
MARGIN="${MQUAKE_SURE_CONSTRAINT_MARGIN:-0.05}"; MAX_PKL="${SURE_MAX_PROTECTED_KL:-0.05}"
EVAL_BATCH="${MQUAKE_EVAL_BATCH_SIZE:-8}"; RUN_ATOMIC_GEN="${MQUAKE_RUN_ATOMIC_GEN:-0}"
read -r -a SEEDS <<< "${SEEDS_TEXT}"
# Do not require the MQuAKE JSON to exist up front: the canonical split builder
# uses the repository's pinned downloader when this path is missing.
test -d "${WIKIDATA_DIR}"

for SEED in "${SEEDS[@]}"; do
  ROOT="${OUTPUT_ROOT}/seed${SEED}"; PROTOCOL="${ROOT}/protocol"
  VISIBLE="${PROTOCOL}/training_visible_forget.json"; MANIFEST="${PROTOCOL}/split_manifest.json"
  LEVEL1="${ROOT}/level1_directional_sure"; LEVEL2="${ROOT}/level2_residual_repair"
  FINAL="${ROOT}/official_eval_locked.json"; EVAL_MANIFEST="${ROOT}/final_eval_split_manifest.json"
  mkdir -p "${ROOT}"; rm -rf "${PROTOCOL}" "${LEVEL1}" "${LEVEL2}"

  echo "===== MQUAKE SEED ${SEED}: LOCKED DIRECT-ONLY SPLIT ====="
  python scripts/build_mquake_sure_canonical_split.py --mquake-path "${MQUAKE}" \
    --output-dir "${PROTOCOL}" --seed "${SEED}" --forget-num "${FORGET_INSTANCES}" --retain-num "${RETAIN_INSTANCES}"
  DIRECT_COUNT="$(python -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["sampling"]["forget_atomic_fact_count"])' "${MANIFEST}")"

  echo "===== LEVEL 1: DIRECTIONAL SURE (Delta E_A / Delta W_A; transformer frozen) ====="
  python scripts/sure_stage1_gagd.py --dataset zsre --model-path "${MODEL}" \
    --training-visible-path "${VISIBLE}" --split-manifest "${MANIFEST}" --output-dir "${LEVEL1}" \
    --seed "${SEED}" --forget-num "${DIRECT_COUNT}" --steps "${STAGE1_STEPS}" \
    --batch-size "${STAGE1_BATCH}" --cache-batch-size "${CACHE_BATCH}" --emb-lm-lr "${STAGE1_LR}" \
    --ga-weight "${LAMBDA_F}" --gd-weight "${LAMBDA_P}" --optimizer adamw \
    --dtype "${DTYPE}" --device-map "${DEVICE_MAP}"

  echo "===== LEVEL 1 GATE + LEVEL 2 RESIDUAL-ONLY REPAIR ====="
  python scripts/mquake_sure_stage2_residual_gagd.py --model-path "${LEVEL1}/checkpoint" \
    --training-visible-path "${VISIBLE}" --split-manifest "${MANIFEST}" --output-dir "${LEVEL2}" \
    --seed "${SEED}" --forget-num "${DIRECT_COUNT}" --repair-steps "${STAGE2_STEPS}" \
    --repair-lr "${STAGE2_LR}" --batch-size "${STAGE2_BATCH}" --cache-batch-size "${CACHE_BATCH}" \
    --check-every "${CHECK_EVERY}" --lambda-f "${LAMBDA_F}" --lambda-p "${LAMBDA_P}" \
    --constraint-margin "${MARGIN}" --max-protected-kl "${MAX_PKL}" --dtype "${DTYPE}" --device-map "${DEVICE_MAP}"

  echo "===== FINAL OFFICIAL MQUAKE EVAL (POST-TRAINING ONLY) ====="
  EVAL_ARGS=(--model-dir "${LEVEL2}/checkpoint" --mquake-path "${MQUAKE}" --wikidata-dir "${WIKIDATA_DIR}"
    --out "${FINAL}" --split-manifest "${EVAL_MANIFEST}" --method "MQuAKE Pure Two-Stage Directional SURE"
    --unlearn-num "${FORGET_INSTANCES}" --retain-num "${RETAIN_INSTANCES}" --seed "${SEED}"
    --batch-size "${EVAL_BATCH}" --dtype "${DTYPE}" --device-map "${DEVICE_MAP}")
  if [[ "${RUN_ATOMIC_GEN}" != "1" ]]; then EVAL_ARGS+=(--skip-atomic-gen); fi
  python scripts/mquake_zero_unlearn_official_eval.py "${EVAL_ARGS[@]}"
  python scripts/annotate_ppl_provenance.py --eval-json "${FINAL}" --model-dir "${LEVEL2}/checkpoint" --wikidata-dir "${WIKIDATA_DIR}"
done

echo "Pure two-stage MQuAKE complete: ${OUTPUT_ROOT}"
