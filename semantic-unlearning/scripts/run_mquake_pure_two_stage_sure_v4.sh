#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL="${1:?Usage: bash scripts/run_mquake_pure_two_stage_sure_v4.sh MODEL [MQUAKE_JSON]}"
MQUAKE="${2:-data/MQuAKE-CF-3k-v2.json}"
WIKIDATA_DIR="${WIKIDATA_DIR:-data/wikidata}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/mquake_pure_two_stage_prompt_invariant_v4_3b}"
SEEDS_TEXT="${MQUAKE_SEEDS:-1}"
FORGET_INSTANCES="${MQUAKE_FORGET_NUM:-50}"
RETAIN_INSTANCES="${MQUAKE_RETAIN_EVAL_NUM:-1000}"
DTYPE="${DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-single}"
CACHE_BATCH="${SURE_CACHE_BATCH_SIZE:-8}"
DIRECT_MARGIN="${MQUAKE_SURE_CONSTRAINT_MARGIN:-0.05}"
ROBUST_MARGIN="${SURE_ROBUST_MARGIN:-0.25}"
MAX_PKL="${SURE_MAX_PROTECTED_KL:-0.05}"
EVAL_BATCH="${MQUAKE_EVAL_BATCH_SIZE:-8}"
RUN_ATOMIC_GEN="${MQUAKE_RUN_ATOMIC_GEN:-1}"
EVAL_FAILED_GATE="${MQUAKE_EVAL_FAILED_GATE:-0}"
SYNTHETIC_VIEWS="${SURE_SYNTHETIC_VIEW_COUNT:-3}"

# Stage 1 v4: prompt-invariant H_F^aug, prefix-only embedding edits, active GA.
STAGE1_STEPS="${SURE_STAGE1_STEPS:-600}"
STAGE1_LR="${SURE_STAGE1_LR:-0.0001}"
STAGE1_BATCH="${SURE_STAGE1_BATCH_SIZE:-4}"
STAGE1_EMBED_SCALE="${SURE_STAGE1_EMBEDDING_SCALE:-0.25}"
STAGE1_CONTEXT_TOKENS="${SURE_STAGE1_PROTECTED_CONTEXT_TOKENS:-4}"
STAGE1_CHECK_EVERY="${SURE_STAGE1_CHECK_EVERY:-25}"

# Stage 2 v4: direct+synthetic residual gate, exact robust-P nullspace, head only,
# then minimum-norm scalar shrink before materialization.
STAGE2_STEPS="${SURE_STAGE2_STEPS:-800}"
STAGE2_LR="${SURE_STAGE2_LR:-0.0005}"
STAGE2_BATCH="${SURE_STAGE2_BATCH_SIZE:-8}"
STAGE2_CHECK_EVERY="${SURE_STAGE2_CHECK_EVERY:-25}"
STAGE2_L2="${SURE_STAGE2_L2_WEIGHT:-0.000001}"
BACKTRACK_SCALES="${SURE_STAGE2_BACKTRACK_SCALES:-0.5,0.25,0.125,0.0625,0.03125,0.015625}"
SHRINK_ITERS="${SURE_STAGE2_SHRINK_ITERS:-24}"
SHRINK_SAFETY="${SURE_STAGE2_SHRINK_SAFETY:-1.01}"

read -r -a SEEDS <<< "${SEEDS_TEXT}"
test -d "${WIKIDATA_DIR}"

for SEED in "${SEEDS[@]}"; do
  ROOT="${OUTPUT_ROOT}/seed${SEED}"
  PROTOCOL="${ROOT}/protocol"
  VISIBLE="${PROTOCOL}/training_visible_forget.json"
  MANIFEST="${PROTOCOL}/split_manifest.json"
  LEVEL1="${ROOT}/level1_prompt_invariant_active_ga"
  LEVEL2="${ROOT}/level2_robust_nullspace_min_norm"
  SUMMARY="${LEVEL2}/two_stage_summary.json"
  FINAL="${ROOT}/official_eval_with_atomicgen.json"
  EVAL_MANIFEST="${ROOT}/final_eval_split_manifest.json"

  mkdir -p "${ROOT}"
  rm -rf "${PROTOCOL}" "${LEVEL1}" "${LEVEL2}"

  echo "===== MQUAKE SEED ${SEED}: LOCKED TRAINING-VISIBLE DIRECT SPLIT ====="
  python scripts/build_mquake_sure_canonical_split.py \
    --mquake-path "${MQUAKE}" \
    --output-dir "${PROTOCOL}" \
    --seed "${SEED}" \
    --forget-num "${FORGET_INSTANCES}" \
    --retain-num "${RETAIN_INSTANCES}"

  DIRECT_COUNT="$(python -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["sampling"]["forget_atomic_fact_count"])' "${MANIFEST}")"

  echo "===== LEVEL 1 v4: PROMPT-INVARIANT ACTIVE DIRECTIONAL GA ====="
  echo "      H_F^aug = direct + deterministic training-only synthetic views"
  echo "      B_S = rowspace(H_F^aug - Proj_BNS(H_F^aug))"
  echo "      Delta E_AE=eta*C_E*B_S (prefix-only); Delta W_AW=C_W*B_S (all targets)"
  python scripts/mquake_sure_stage1_prompt_invariant_v4.py \
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
    --embedding-scale "${STAGE1_EMBED_SCALE}" \
    --synthetic-view-count "${SYNTHETIC_VIEWS}" \
    --protected-context-tokens "${STAGE1_CONTEXT_TOKENS}" \
    --check-every "${STAGE1_CHECK_EVERY}" \
    --constraint-margin "${DIRECT_MARGIN}" \
    --robust-margin "${ROBUST_MARGIN}" \
    --optimizer adamw \
    --dtype "${DTYPE}" \
    --device-map "${DEVICE_MAP}"

  echo "===== LEVEL 2 v4: ROBUST EXACT-P-NULLSPACE HEAD REPAIR + MIN-NORM SHRINK ====="
  echo "      F/P are defined over direct + training-only synthetic gates"
  echo "      B_F = rowspace(H_F - Proj_rowspace(H_P)(H_F)); Delta W_AF=C_F*B_F"
  python scripts/mquake_sure_stage2_robust_nullspace_v4.py \
    --model-path "${LEVEL1}/checkpoint" \
    --training-visible-path "${VISIBLE}" \
    --split-manifest "${MANIFEST}" \
    --output-dir "${LEVEL2}" \
    --seed "${SEED}" \
    --forget-num "${DIRECT_COUNT}" \
    --synthetic-view-count "${SYNTHETIC_VIEWS}" \
    --repair-steps "${STAGE2_STEPS}" \
    --repair-lr "${STAGE2_LR}" \
    --batch-size "${STAGE2_BATCH}" \
    --cache-batch-size "${CACHE_BATCH}" \
    --check-every "${STAGE2_CHECK_EVERY}" \
    --constraint-margin "${DIRECT_MARGIN}" \
    --robust-margin "${ROBUST_MARGIN}" \
    --max-protected-kl "${MAX_PKL}" \
    --l2-weight "${STAGE2_L2}" \
    --backtrack-scales "${BACKTRACK_SCALES}" \
    --shrink-iters "${SHRINK_ITERS}" \
    --shrink-safety "${SHRINK_SAFETY}" \
    --dtype "${DTYPE}" \
    --device-map "${DEVICE_MAP}"

  FINAL_PASS="$(python -c 'import json,sys; d=json.load(open(sys.argv[1])); print("1" if d.get("final_gates_pass") else "0")' "${SUMMARY}")"

  if [[ "${FINAL_PASS}" != "1" && "${EVAL_FAILED_GATE}" != "1" ]]; then
    echo "===== OFFICIAL EVAL SKIPPED: FINAL TRAINING GATES FAILED ====="
    echo "Set MQUAKE_EVAL_FAILED_GATE=1 only for diagnostic evaluation of a failed checkpoint."
    continue
  fi

  echo "===== FINAL OFFICIAL MQUAKE EVAL: HELD OUT UNTIL CHECKPOINT FIXED ====="
  EVAL_ARGS=(
    --model-dir "${LEVEL2}/checkpoint"
    --mquake-path "${MQUAKE}"
    --wikidata-dir "${WIKIDATA_DIR}"
    --out "${FINAL}"
    --split-manifest "${EVAL_MANIFEST}"
    --method "MQuAKE Prompt-Invariant Two-Stage SURE v4"
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
done

echo "Prompt-invariant two-stage MQuAKE SURE v4 complete: ${OUTPUT_ROOT}"
