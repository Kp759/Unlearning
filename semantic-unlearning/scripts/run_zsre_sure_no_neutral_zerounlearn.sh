#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL="${1:?Usage: bash scripts/run_zsre_sure_no_neutral_zerounlearn.sh MODEL [ZSRE_JSON]}"
ZSRE="${2:-data/zsre_mend_eval.json}"
WIKIDATA_DIR="${WIKIDATA_DIR:-data/wikidata_aws_diag}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/aws_zsre_sure_no_neutral_exact_3b}"
SEEDS_TEXT="${ZSRE_SEEDS:-1}"
FORGET_NUM="${ZSRE_FORGET_NUM:-50}"
RETAIN_NUM="${ZSRE_RETAIN_EVAL_NUM:-1000}"
DTYPE="${DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-single}"

# Keep the previous Stage-1 operating point for the first causal comparison.
STEPS="${ZSRE_STEPS:-600}"
BATCH_SIZE="${ZSRE_BATCH_SIZE:-1}"
EMB_LM_LR="${ZSRE_EMB_LM_LR:-0.0001}"
FORGET_WEIGHT="${ZSRE_FORGET_WEIGHT:-2.0}"
FORGET_MARGIN="${ZSRE_FORGET_MARGIN:-1.0}"

# Sparse direct-rewrite sensitive-row repair only.
REPAIR_STEPS="${REPAIR_STEPS:-800}"
REPAIR_LR="${REPAIR_LR:-0.005}"
REPAIR_MARGIN="${REPAIR_MARGIN:-0.05}"
REPAIR_L2="${REPAIR_L2:-0.000001}"
REPAIR_BATCH_SIZE="${REPAIR_BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
CANDIDATE_SCALES="${CANDIDATE_SCALES:-1,.875,.75,.625,.5,.375,.25,.1875,.125,.09375,.0625,.046875,.03125,.015625,.0078125,0}"

read -r -a SEEDS <<< "${SEEDS_TEXT}"
test -f "${ZSRE}"
test -d "${WIKIDATA_DIR}"

for SEED in "${SEEDS[@]}"; do
  ROOT="${OUTPUT_ROOT}/seed${SEED}"
  PROTOCOL="${ROOT}/protocol_no_neutral"
  VISIBLE="${PROTOCOL}/training_visible_forget.json"
  MANIFEST="${PROTOCOL}/split_manifest.json"
  STAGE1="${ROOT}/stage1_emb_lm_no_neutral"
  STAGE1_CKPT="${STAGE1}/checkpoint"
  STAGE2="${ROOT}/stage2_sensitive_row_repair"
  STAGE2_CKPT="${STAGE2}/checkpoint"

  mkdir -p "${ROOT}"
  echo "===== SEED ${SEED}: EXACT ZEROUnlearn NO-NEUTRAL SPLIT ====="
  python scripts/build_zsre_zerounlearn_locked_no_neutral_split.py \
    --zsre-path "${ZSRE}" --output-dir "${PROTOCOL}" --seed "${SEED}" \
    --forget-num "${FORGET_NUM}" --retain-num "${RETAIN_NUM}"

  python scripts/verify_zsre_zerounlearn_locked_split.py \
    --zsre-path "${ZSRE}" --split-manifest "${MANIFEST}" \
    --repair-visible-path "${VISIBLE}" --seed "${SEED}" \
    --forget-num "${FORGET_NUM}" --retain-num "${RETAIN_NUM}"

  echo "===== SEED ${SEED}: STAGE 1 — EMB+LM SENSITIVE SUPPRESSION ====="
  rm -rf "${STAGE1}"
  python scripts/zsre_no_neutral_stage1_emb_lm.py \
    --model-path "${MODEL}" --training-visible-path "${VISIBLE}" \
    --split-manifest "${MANIFEST}" --output-dir "${STAGE1}" \
    --seed "${SEED}" --forget-num "${FORGET_NUM}" \
    --steps "${STEPS}" --batch-size "${BATCH_SIZE}" \
    --emb-lm-lr "${EMB_LM_LR}" --forget-weight "${FORGET_WEIGHT}" \
    --forget-margin "${FORGET_MARGIN}" --optimizer adamw \
    --dtype "${DTYPE}" --device-map "${DEVICE_MAP}"

  echo "===== SEED ${SEED}: LOCKED EVAL AFTER STAGE 1 ====="
  python scripts/zsre_zero_unlearn_official_eval.py \
    --model-dir "${STAGE1_CKPT}" --zsre-path "${ZSRE}" \
    --wikidata-dir "${WIKIDATA_DIR}" --out "${ROOT}/official_eval_stage1_locked.json" \
    --method "SURE ZsRE no-neutral Stage1 Emb+LM" \
    --unlearn-num "${FORGET_NUM}" --retain-num "${RETAIN_NUM}" --seed "${SEED}" \
    --batch-size "${EVAL_BATCH_SIZE}" --dtype "${DTYPE}" --device-map "${DEVICE_MAP}"

  echo "===== SEED ${SEED}: STAGE 2 — ACTIVE SENSITIVE LM-HEAD ROWS ====="
  rm -rf "${STAGE2}"
  python scripts/zsre_no_neutral_active_sensitive_row_repair.py \
    --model-path "${STAGE1_CKPT}" --training-visible-path "${VISIBLE}" \
    --split-manifest "${MANIFEST}" --output-dir "${STAGE2}" \
    --seed "${SEED}" --forget-num "${FORGET_NUM}" \
    --repair-steps "${REPAIR_STEPS}" --repair-lr "${REPAIR_LR}" \
    --repair-margin "${REPAIR_MARGIN}" --repair-l2 "${REPAIR_L2}" \
    --batch-size "${REPAIR_BATCH_SIZE}" --candidate-scales "${CANDIDATE_SCALES}" \
    --dtype "${DTYPE}" --device-map "${DEVICE_MAP}"

  echo "===== SEED ${SEED}: FINAL LOCKED OFFICIAL EVAL ====="
  python scripts/zsre_zero_unlearn_official_eval.py \
    --model-dir "${STAGE2_CKPT}" --zsre-path "${ZSRE}" \
    --wikidata-dir "${WIKIDATA_DIR}" --out "${ROOT}/official_eval_locked.json" \
    --method "SURE ZsRE no-neutral Emb+LM plus sensitive-row repair" \
    --unlearn-num "${FORGET_NUM}" --retain-num "${RETAIN_NUM}" --seed "${SEED}" \
    --batch-size "${EVAL_BATCH_SIZE}" --dtype "${DTYPE}" --device-map "${DEVICE_MAP}"

  python - "${ROOT}" <<'PY'
import json,sys
r=sys.argv[1]
for name,f in (("STAGE1","official_eval_stage1_locked.json"),("STAGE2","official_eval_locked.json")):
    x=json.load(open(f"{r}/{f}"))
    print(name, "F-Eff",x["forget"]["Eff"],"F-Gen",x["forget"]["Gen"],"F-Spe",x["forget"]["Spe"],
          "R-Eff",x["retain"]["Eff"],"R-Gen",x["retain"]["Gen"],"R-Spe",x["retain"]["Spe"],"PPL",x["forget_PPL"])
PY

done

echo "Done. Results: ${OUTPUT_ROOT}/seed*/official_eval_{stage1_locked,locked}.json"
