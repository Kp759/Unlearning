#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BASE_MODEL="${1:?Usage: bash scripts/run_mquake_sure_v3_forensics.sh BASE_MODEL [V3_ROOT] [MQUAKE_JSON]}"
V3_ROOT="${2:-outputs/mquake_pure_two_stage_directional_v3_3b}"
MQUAKE="${3:-data/MQuAKE-CF-3k-v2.json}"
WIKIDATA_DIR="${WIKIDATA_DIR:-data/wikidata}"
SEED="${MQUAKE_FORENSICS_SEED:-1}"
FORGET_NUM="${MQUAKE_FORGET_NUM:-50}"
RETAIN_NUM="${MQUAKE_RETAIN_EVAL_NUM:-1000}"
BATCH="${MQUAKE_FORENSICS_BATCH_SIZE:-8}"
DTYPE="${DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-single}"
WITH_PPL="${MQUAKE_FORENSICS_WITH_PPL:-1}"

ROOT="${V3_ROOT}/seed${SEED}"
STAGE1="${ROOT}/level1_full_residual_directional_ga/checkpoint"
FINAL="${ROOT}/level2_head_exact_p_nullspace/checkpoint"
OUT="${ROOT}/forensics_v3"

test -d "${BASE_MODEL}"
test -d "${STAGE1}"
test -d "${FINAL}"
test -f "${MQUAKE}"

ARGS=(
  --base-model-path "${BASE_MODEL}"
  --stage1-model-path "${STAGE1}"
  --final-model-path "${FINAL}"
  --mquake-path "${MQUAKE}"
  --wikidata-dir "${WIKIDATA_DIR}"
  --output-dir "${OUT}"
  --seed "${SEED}"
  --forget-num "${FORGET_NUM}"
  --retain-num "${RETAIN_NUM}"
  --batch-size "${BATCH}"
  --dtype "${DTYPE}"
  --device-map "${DEVICE_MAP}"
)
if [[ "${WITH_PPL}" == "1" ]]; then
  test -d "${WIKIDATA_DIR}"
  ARGS+=(--with-ppl)
fi

echo "===== READ-ONLY SURE v3 FORENSICS: seed ${SEED} ====="
echo "base:   ${BASE_MODEL}"
echo "stage1: ${STAGE1}"
echo "final:  ${FINAL}"
echo "out:    ${OUT}"
python scripts/mquake_sure_v3_forensics.py "${ARGS[@]}"
