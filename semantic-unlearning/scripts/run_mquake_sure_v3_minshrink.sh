#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

V3_ROOT="${1:-outputs/mquake_pure_two_stage_directional_v3_3b}"
MQUAKE="${2:-data/MQuAKE-CF-3k-v2.json}"
WIKIDATA_DIR="${WIKIDATA_DIR:-data/wikidata}"
SEED="${MQUAKE_MINSHRINK_SEED:-1}"
SOURCE_FORGET="${MQUAKE_FORGET_NUM:-50}"
RETAIN_NUM="${MQUAKE_RETAIN_EVAL_NUM:-1000}"
DTYPE="${DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-single}"
CACHE_BATCH="${SURE_CACHE_BATCH_SIZE:-8}"
EVAL_BATCH="${MQUAKE_EVAL_BATCH_SIZE:-8}"
MARGIN="${MQUAKE_SURE_CONSTRAINT_MARGIN:-0.05}"
GLOBAL_GRID="${SURE_V3_SHRINK_GLOBAL_GRID_POINTS:-2001}"
ROW_GRID="${SURE_V3_SHRINK_ROW_GRID_POINTS:-101}"
ROW_PASSES="${SURE_V3_SHRINK_ROW_PASSES:-2}"
RECOVERY_POINTS="${SURE_V3_SHRINK_RECOVERY_POINTS:-33}"
RUN_OFFICIAL="${SURE_V3_SHRINK_RUN_OFFICIAL:-1}"

ROOT="${V3_ROOT}/seed${SEED}"
PROTOCOL="${ROOT}/protocol"
VISIBLE="${PROTOCOL}/training_visible_forget.json"
MANIFEST="${PROTOCOL}/split_manifest.json"
STAGE1="${ROOT}/level1_full_residual_directional_ga/checkpoint"
FINAL="${ROOT}/level2_head_exact_p_nullspace/checkpoint"
OUT="${ROOT}/stage2_minshrink_v3"
SUMMARY="${OUT}/minshrink_summary.json"

test -d "${STAGE1}"
test -d "${FINAL}"
test -f "${VISIBLE}"
test -f "${MANIFEST}"
test -f "${MQUAKE}"

DIRECT_COUNT="$(python -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["sampling"]["forget_atomic_fact_count"])' "${MANIFEST}")"

rm -rf "${OUT}"
mkdir -p "${OUT}"

echo "===== SURE v3 STAGE2 DIRECT-ONLY MINIMUM SHRINK: seed ${SEED} ====="
echo "Stage1: ${STAGE1}"
echo "Final:  ${FINAL}"
echo "Direct PredictionCases: ${DIRECT_COUNT}"

python scripts/mquake_sure_v3_stage2_minshrink.py \
  --stage1-model-path "${STAGE1}" \
  --final-model-path "${FINAL}" \
  --training-visible-path "${VISIBLE}" \
  --split-manifest "${MANIFEST}" \
  --output-dir "${OUT}" \
  --seed "${SEED}" \
  --forget-num "${DIRECT_COUNT}" \
  --constraint-margin "${MARGIN}" \
  --cache-batch-size "${CACHE_BATCH}" \
  --global-grid-points "${GLOBAL_GRID}" \
  --row-grid-points "${ROW_GRID}" \
  --row-passes "${ROW_PASSES}" \
  --actual-recovery-points "${RECOVERY_POINTS}" \
  --dtype "${DTYPE}" \
  --device-map "${DEVICE_MAP}"

python - "${SUMMARY}" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
for name in ("global", "rowwise"):
    g = p[name]["materialized_gate"]
    if int(g["failed"]) != 0:
        raise SystemExit(f"{name} materialized direct gate failed: {g}")
print("Both min-shrink variants pass the locked direct gate.")
PY

if [[ "${RUN_OFFICIAL}" != "1" ]]; then
  echo "===== OFFICIAL EVAL DISABLED ====="
  echo "Set SURE_V3_SHRINK_RUN_OFFICIAL=1 to evaluate the preselected variants."
  exit 0
fi

test -d "${WIKIDATA_DIR}"

for VARIANT in global_min rowwise_min; do
  CKPT="${OUT}/${VARIANT}/checkpoint"
  EVAL_OUT="${OUT}/${VARIANT}/official_eval_with_atomicgen.json"
  EVAL_MANIFEST="${OUT}/${VARIANT}/official_eval_split_manifest.json"

  if [[ "${VARIANT}" == "global_min" ]]; then
    METHOD="MQuAKE SURE v3 + Direct-Only Global Minimum Stage2 Shrink"
  else
    METHOD="MQuAKE SURE v3 + Direct-Only Rowwise Minimum Stage2 Shrink"
  fi

  echo "===== OFFICIAL POST-SELECTION EVAL: ${VARIANT} ====="
  python scripts/mquake_zero_unlearn_official_eval.py \
    --model-dir "${CKPT}" \
    --mquake-path "${MQUAKE}" \
    --wikidata-dir "${WIKIDATA_DIR}" \
    --out "${EVAL_OUT}" \
    --split-manifest "${EVAL_MANIFEST}" \
    --method "${METHOD}" \
    --unlearn-num "${SOURCE_FORGET}" \
    --retain-num "${RETAIN_NUM}" \
    --seed "${SEED}" \
    --batch-size "${EVAL_BATCH}" \
    --dtype "${DTYPE}" \
    --device-map "${DEVICE_MAP}"

  python scripts/annotate_ppl_provenance.py \
    --eval-json "${EVAL_OUT}" \
    --model-dir "${CKPT}" \
    --wikidata-dir "${WIKIDATA_DIR}"
done

echo "===== SURE v3 MIN-SHRINK COMPLETE ====="
echo "Summary: ${SUMMARY}"
echo "Global eval:  ${OUT}/global_min/official_eval_with_atomicgen.json"
echo "Rowwise eval: ${OUT}/rowwise_min/official_eval_with_atomicgen.json"
