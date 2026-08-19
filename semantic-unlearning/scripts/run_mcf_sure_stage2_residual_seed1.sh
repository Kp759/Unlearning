#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL="${1:?Usage: bash scripts/run_mcf_sure_stage2_residual_seed1.sh MODEL [SOURCE_SEED1_ROOT] [OUTPUT_SEED1_ROOT]}"
SOURCE_ROOT="${2:-outputs/mcf_sure_2stage_target_true_seed1/seed1}"
OUT_ROOT="${3:-outputs/mcf_sure_2stage_target_true_seed1_residual_v2/seed1}"
MCF="${MCF_JSON:-data/multi_counterfact.json}"
WIKIDATA_DIR="${WIKIDATA_DIR:-data/wikidata}"
DTYPE="${DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-single}"

STAGE1="${SOURCE_ROOT}/stage1_gagd/checkpoint"
VISIBLE="${SOURCE_ROOT}/protocol/training_visible_forget.json"
MANIFEST="${SOURCE_ROOT}/protocol/split_manifest.json"
SOURCE_BASE_EVAL="${SOURCE_ROOT}/base_original_mcf_eval.json"
STAGE2="${OUT_ROOT}/stage2_residual_sensitive_context"
POST_EVAL="${OUT_ROOT}/post_original_mcf_eval.json"
PAPER_EVAL="${OUT_ROOT}/target_true_sensitive_eval.json"
BASE_EVAL="${OUT_ROOT}/base_original_mcf_eval.json"

MARGIN="${MCF_SURE_RESIDUAL_MARGIN:-0.05}"
RANKS="${SURE_REPAIR_RANKS:-2,8,0}"
STEPS="${SURE_REPAIR_STEPS:-800}"
LR="${SURE_REPAIR_LR:-0.005}"
L2="${SURE_REPAIR_L2:-0.000001}"
BATCH="${SURE_REPAIR_BATCH_SIZE:-8}"
CHECK_EVERY="${SURE_REPAIR_CHECK_EVERY:-25}"
SCALES="${SURE_CANDIDATE_SCALES:-1,.875,.75,.625,.5,.375,.25,.1875,.125,.09375,.0625,.046875,.03125,.015625,.0078125,0}"

for p in "${MODEL}" "${STAGE1}" "${WIKIDATA_DIR}"; do test -d "$p"; done
for p in "${MCF}" "${VISIBLE}" "${MANIFEST}"; do test -f "$p"; done
mkdir -p "${OUT_ROOT}"
rm -rf "${STAGE2}"

# Reuse the exact same Stage-1 checkpoint and locked split.  Only Stage 2 changes.
echo "===== MCF SEED 1: RESIDUAL SENSITIVE-CONTEXT STAGE 2 ====="
echo "source Stage1: ${STAGE1}"
echo "margin=${MARGIN}; ranks=${RANKS}"
python scripts/sure_stage2_sparse_repair_residual.py \
  --model-path "${STAGE1}" \
  --training-visible-path "${VISIBLE}" \
  --split-manifest "${MANIFEST}" \
  --output-dir "${STAGE2}" \
  --seed 1 \
  --forget-num 50 \
  --candidate-ranks "${RANKS}" \
  --repair-steps "${STEPS}" \
  --repair-lr "${LR}" \
  --constraint-margin "${MARGIN}" \
  --repair-l2 "${L2}" \
  --batch-size "${BATCH}" \
  --check-every "${CHECK_EVERY}" \
  --candidate-scales "${SCALES}" \
  --dtype "${DTYPE}" \
  --device-map "${DEVICE_MAP}"

test -d "${STAGE2}/checkpoint"
test -f "${STAGE2}/repair_summary.json"

# Reuse the matched Base evaluation if available; it is independent of Stage 2.
if [[ -f "${SOURCE_BASE_EVAL}" ]]; then
  cp "${SOURCE_BASE_EVAL}" "${BASE_EVAL}"
else
  python scripts/mcf_zero_unlearn_official_eval.py \
    --model-dir "${MODEL}" --mcf-path "${MCF}" --wikidata-dir "${WIKIDATA_DIR}" \
    --out "${BASE_EVAL}" --unlearn-num 50 --retain-num 1000 --seed 1 \
    --sample-mode official --dtype "${DTYPE}" --device-map "${DEVICE_MAP}"
fi

python scripts/mcf_zero_unlearn_official_eval.py \
  --model-dir "${STAGE2}/checkpoint" \
  --mcf-path "${MCF}" \
  --wikidata-dir "${WIKIDATA_DIR}" \
  --out "${POST_EVAL}" \
  --unlearn-num 50 \
  --retain-num 1000 \
  --seed 1 \
  --sample-mode official \
  --dtype "${DTYPE}" \
  --device-map "${DEVICE_MAP}"

python scripts/annotate_ppl_provenance.py \
  --eval-json "${POST_EVAL}" \
  --model-dir "${STAGE2}/checkpoint" \
  --wikidata-dir "${WIKIDATA_DIR}"

python scripts/evaluate_mcf_target_true_sensitive.py \
  --base-eval-json "${BASE_EVAL}" \
  --post-eval-json "${POST_EVAL}" \
  --split-manifest "${MANIFEST}" \
  --out "${PAPER_EVAL}"

echo
echo "===== RESIDUAL V2 SUMMARY ====="
python - "${PAPER_EVAL}" "${STAGE2}/repair_summary.json" <<'PY'
import json, sys
paper=json.load(open(sys.argv[1]))
repair=json.load(open(sys.argv[2]))
m=paper["metrics"]
def mean(name, legacy=None):
    x=m.get(name)
    if x is None and legacy:
        x=m.get(legacy)
    if isinstance(x, dict):
        return x.get("mean")
    return x
print("FS:", mean("FS", "Eff"))
print("GFS:", mean("GFS", "Gen"))
print("SensitivePref direct:", mean("SensitivePref_direct", "Eff_Pref"))
print("SensitivePref para:", mean("SensitivePref_paraphrase", "Gen_Pref"))
print("Delta Sensitive NLL direct:", mean("Delta_Sensitive_NLL_direct"))
print("Delta Sensitive NLL para:", mean("Delta_Sensitive_NLL_paraphrase"))
print("Spe margin:", mean("Spe_margin"))
print("Spe success:", mean("Spe_success"))
print("PPL:", mean("PPL"))
print("active_before:", repair["active_before"])
print("active_after:", repair["active_after"])
print("selected_rows:", repair["selected_lm_head_rows"])
print("selected_scale:", repair["selected_scale"])
print("effective_delta_norm:", repair["effective_delta_norm"])
print("chosen_candidate:", repair["chosen_candidate"])
PY

echo "Residual-v2 output: ${OUT_ROOT}"
