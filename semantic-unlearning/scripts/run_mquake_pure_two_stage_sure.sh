#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL="${1:?Usage: bash scripts/run_mquake_pure_two_stage_sure.sh MODEL [MQUAKE_JSON]}"
MQUAKE="${2:-data/MQuAKE-CF-3k-v2.json}"
WIKIDATA_DIR="${WIKIDATA_DIR:-data/wikidata}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/mquake_pure_two_stage_directional_v2_3b}"
SEEDS_TEXT="${MQUAKE_SEEDS:-1 2 3 4 5 6 7 8 9 10}"
FORGET_INSTANCES="${MQUAKE_FORGET_NUM:-50}"
RETAIN_INSTANCES="${MQUAKE_RETAIN_EVAL_NUM:-1000}"
DTYPE="${DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-single}"
CACHE_BATCH="${SURE_CACHE_BATCH_SIZE:-8}"

# Stage 1: untie E/W, derive B_S by residualizing sensitive prediction hidden
# states against a protected context-hidden subspace, then learn only
# Delta E_A=C_E B_S and Delta W_A=C_W B_S.
STAGE1_STEPS="${SURE_STAGE1_STEPS:-600}"
STAGE1_LR="${SURE_STAGE1_LR:-0.0001}"
STAGE1_BATCH="${SURE_STAGE1_BATCH_SIZE:-1}"
STAGE1_DIRECTION_RANK="${SURE_STAGE1_DIRECTION_RANK:-4}"
STAGE1_PROTECTED_RANK="${SURE_STAGE1_PROTECTED_RANK:-32}"
STAGE1_CONTEXT_TOKENS="${SURE_STAGE1_PROTECTED_CONTEXT_TOKENS:-4}"
STAGE1_GA_WEIGHT="${SURE_STAGE1_GA_WEIGHT:-2.0}"
STAGE1_PROTECTION_WEIGHT="${SURE_STAGE1_PROTECTION_WEIGHT:-1.0}"

# Stage 2: embedding/transformer frozen; selected residual LM-head rows only.
# Every accepted step must keep all Stage-1 successes and the exact full-vocab
# Stage1||Stage2 KL budget.
STAGE2_STEPS="${SURE_STAGE2_STEPS:-800}"
STAGE2_LR="${SURE_STAGE2_LR:-0.0005}"
STAGE2_BATCH="${SURE_STAGE2_BATCH_SIZE:-8}"
PROTECTION_BATCH="${SURE_STAGE2_PROTECTION_BATCH_SIZE:-16}"
STAGE2_REPAIR_RANK="${SURE_STAGE2_REPAIR_RANK:-4}"
STAGE2_PROTECTED_RANK="${SURE_STAGE2_PROTECTED_RANK:-32}"
STAGE2_REPAIR_WEIGHT="${SURE_STAGE2_REPAIR_WEIGHT:-1.0}"
STAGE2_PROTECTION_WEIGHT="${SURE_STAGE2_PROTECTION_WEIGHT:-10.0}"
STAGE2_L2_WEIGHT="${SURE_STAGE2_L2_WEIGHT:-0.000001}"
BACKTRACK_SCALES="${SURE_STAGE2_BACKTRACK_SCALES:-0.5,0.25,0.125,0.0625,0.03125,0.015625}"

MARGIN="${MQUAKE_SURE_CONSTRAINT_MARGIN:-0.05}"
MAX_PKL="${SURE_MAX_PROTECTED_KL:-0.05}"
EVAL_BATCH="${MQUAKE_EVAL_BATCH_SIZE:-8}"
RUN_ATOMIC_GEN="${MQUAKE_RUN_ATOMIC_GEN:-0}"
EVAL_FAILED_GATE="${MQUAKE_EVAL_FAILED_GATE:-0}"

read -r -a SEEDS <<< "${SEEDS_TEXT}"
# The canonical split builder uses the repository's pinned downloader if the
# MQuAKE JSON is absent. PPL evaluation still requires the pinned wikidata dir.
test -d "${WIKIDATA_DIR}"

for SEED in "${SEEDS[@]}"; do
  ROOT="${OUTPUT_ROOT}/seed${SEED}"
  PROTOCOL="${ROOT}/protocol"
  VISIBLE="${PROTOCOL}/training_visible_forget.json"
  MANIFEST="${PROTOCOL}/split_manifest.json"
  LEVEL1="${ROOT}/level1_directional_sure"
  LEVEL2="${ROOT}/level2_head_only_residual_repair"
  SUMMARY="${LEVEL2}/two_stage_summary.json"
  FINAL="${ROOT}/official_eval_locked.json"
  EVAL_MANIFEST="${ROOT}/final_eval_split_manifest.json"
  mkdir -p "${ROOT}"
  rm -rf "${PROTOCOL}" "${LEVEL1}" "${LEVEL2}"

  echo "===== MQUAKE SEED ${SEED}: LOCKED DIRECT-ONLY SPLIT ====="
  python scripts/build_mquake_sure_canonical_split.py \
    --mquake-path "${MQUAKE}" \
    --output-dir "${PROTOCOL}" \
    --seed "${SEED}" \
    --forget-num "${FORGET_INSTANCES}" \
    --retain-num "${RETAIN_INSTANCES}"

  DIRECT_COUNT="$(python -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["sampling"]["forget_atomic_fact_count"])' "${MANIFEST}")"

  echo "===== LEVEL 1: UNTIED DIRECTIONAL SURE ====="
  echo "      B_S = SVD(H_F - Proj_Bp(H_F)); Delta E_A=C_E B_S; Delta W_A=C_W B_S"
  python scripts/mquake_sure_stage1_directional.py \
    --model-path "${MODEL}" \
    --training-visible-path "${VISIBLE}" \
    --split-manifest "${MANIFEST}" \
    --output-dir "${LEVEL1}" \
    --seed "${SEED}" \
    --forget-num "${DIRECT_COUNT}" \
    --steps "${STAGE1_STEPS}" \
    --batch-size "${STAGE1_BATCH}" \
    --cache-batch-size "${CACHE_BATCH}" \
    --learning-rate "${STAGE1_LR}" \
    --ga-weight "${STAGE1_GA_WEIGHT}" \
    --protection-weight "${STAGE1_PROTECTION_WEIGHT}" \
    --direction-rank "${STAGE1_DIRECTION_RANK}" \
    --protected-rank "${STAGE1_PROTECTED_RANK}" \
    --protected-context-tokens "${STAGE1_CONTEXT_TOKENS}" \
    --optimizer adamw \
    --dtype "${DTYPE}" \
    --device-map "${DEVICE_MAP}"

  echo "===== LEVEL 1 GATE + LEVEL 2 HARD-PROTECTED LM-HEAD-ONLY REPAIR ====="
  echo "      B_F = SVD(H_F_residual - Proj_Bp(H_F_residual)); Delta W_AF=C_F B_F"
  python scripts/mquake_sure_stage2_head_directional.py \
    --model-path "${LEVEL1}/checkpoint" \
    --training-visible-path "${VISIBLE}" \
    --split-manifest "${MANIFEST}" \
    --output-dir "${LEVEL2}" \
    --seed "${SEED}" \
    --forget-num "${DIRECT_COUNT}" \
    --repair-steps "${STAGE2_STEPS}" \
    --repair-lr "${STAGE2_LR}" \
    --batch-size "${STAGE2_BATCH}" \
    --protection-batch-size "${PROTECTION_BATCH}" \
    --cache-batch-size "${CACHE_BATCH}" \
    --repair-rank "${STAGE2_REPAIR_RANK}" \
    --protected-rank "${STAGE2_PROTECTED_RANK}" \
    --repair-weight "${STAGE2_REPAIR_WEIGHT}" \
    --protection-weight "${STAGE2_PROTECTION_WEIGHT}" \
    --l2-weight "${STAGE2_L2_WEIGHT}" \
    --backtrack-scales "${BACKTRACK_SCALES}" \
    --constraint-margin "${MARGIN}" \
    --max-protected-kl "${MAX_PKL}" \
    --dtype "${DTYPE}" \
    --device-map "${DEVICE_MAP}"

  GATES_PASS="$(python -c 'import json,sys; print("1" if json.load(open(sys.argv[1])).get("final_gates_pass") else "0")' "${SUMMARY}")"
  if [[ "${GATES_PASS}" == "1" || "${EVAL_FAILED_GATE}" == "1" ]]; then
    echo "===== FINAL OFFICIAL MQUAKE EVAL (POST-TRAINING ONLY) ====="
    if [[ "${GATES_PASS}" != "1" ]]; then
      echo "WARNING: evaluating a failed gate only because MQUAKE_EVAL_FAILED_GATE=1"
    fi
    EVAL_ARGS=(
      --model-dir "${LEVEL2}/checkpoint"
      --mquake-path "${MQUAKE}"
      --wikidata-dir "${WIKIDATA_DIR}"
      --out "${FINAL}"
      --split-manifest "${EVAL_MANIFEST}"
      --method "MQuAKE Pure Two-Stage Untied Directional SURE"
      --unlearn-num "${FORGET_INSTANCES}"
      --retain-num "${RETAIN_INSTANCES}"
      --seed "${SEED}"
      --batch-size "${EVAL_BATCH}"
      --dtype "${DTYPE}"
      --device-map "${DEVICE_MAP}"
    )
    if [[ "${RUN_ATOMIC_GEN}" != "1" ]]; then
      EVAL_ARGS+=(--skip-atomic-gen)
    fi
    python scripts/mquake_zero_unlearn_official_eval.py "${EVAL_ARGS[@]}"
    python scripts/annotate_ppl_provenance.py \
      --eval-json "${FINAL}" \
      --model-dir "${LEVEL2}/checkpoint" \
      --wikidata-dir "${WIKIDATA_DIR}"
  else
    echo "===== OFFICIAL EVAL SKIPPED: FINAL TRAINING GATES FAILED ====="
    echo "Set MQUAKE_EVAL_FAILED_GATE=1 only for diagnostic evaluation of a failed checkpoint."
  fi
done

echo "Pure two-stage untied directional MQuAKE complete: ${OUTPUT_ROOT}"
