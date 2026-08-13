#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${1:?MODEL required}"
ZSRE="${2:-data/zsre_mend_eval.json}"
WIKIDATA_DIR="${WIKIDATA_DIR:-data/wikidata_aws_diag}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/aws_zsre_sure_no_neutral_gagd_3b}"
SEEDS_TEXT="${ZSRE_SEEDS:-1}"
FORGET_NUM="${ZSRE_FORGET_NUM:-50}"
RETAIN_NUM="${ZSRE_RETAIN_EVAL_NUM:-1000}"
STEPS="${ZSRE_STEPS:-450}"
BATCH_SIZE="${ZSRE_BATCH_SIZE:-1}"
CACHE_BATCH_SIZE="${ZSRE_CACHE_BATCH_SIZE:-8}"
EMB_LM_LR="${ZSRE_EMB_LM_LR:-0.000075}"
GA_WEIGHT="${ZSRE_GA_WEIGHT:-1.75}"
GD_WEIGHT="${ZSRE_GD_WEIGHT:-1.0}"
REPAIR_STEPS="${REPAIR_STEPS:-800}"
REPAIR_LR="${REPAIR_LR:-0.005}"
REPAIR_MARGIN="${REPAIR_MARGIN:-0.05}"
REPAIR_L2="${REPAIR_L2:-0.000001}"
REPAIR_BATCH_SIZE="${REPAIR_BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
DTYPE="${DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-single}"
CANDIDATE_SCALES="${CANDIDATE_SCALES:-1,.875,.75,.625,.5,.375,.25,.1875,.125,.09375,.0625,.046875,.03125,.015625,.0078125,0}"
read -r -a SEEDS <<< "${SEEDS_TEXT}"
for SEED in "${SEEDS[@]}"; do
  ROOT="${OUTPUT_ROOT}/seed${SEED}"
  PROTOCOL="${ROOT}/protocol_no_neutral"
  VISIBLE="${PROTOCOL}/training_visible_forget.json"
  MANIFEST="${PROTOCOL}/split_manifest.json"
  STAGE1="${ROOT}/stage1_emb_lm_gagd_no_neutral"
  STAGE2="${ROOT}/stage2_sensitive_row_repair"
  mkdir -p "${ROOT}"
  python scripts/build_zsre_zerounlearn_locked_no_neutral_split.py --zsre-path "${ZSRE}" --output-dir "${PROTOCOL}" --seed "${SEED}" --forget-num "${FORGET_NUM}" --retain-num "${RETAIN_NUM}"
  python scripts/verify_zsre_zerounlearn_locked_split.py --zsre-path "${ZSRE}" --split-manifest "${MANIFEST}" --repair-visible-path "${VISIBLE}" --seed "${SEED}" --forget-num "${FORGET_NUM}" --retain-num "${RETAIN_NUM}"
  rm -rf "${STAGE1}"
  python scripts/zsre_no_neutral_stage1_gagd.py --model-path "${MODEL}" --training-visible-path "${VISIBLE}" --split-manifest "${MANIFEST}" --output-dir "${STAGE1}" --seed "${SEED}" --forget-num "${FORGET_NUM}" --steps "${STEPS}" --batch-size "${BATCH_SIZE}" --cache-batch-size "${CACHE_BATCH_SIZE}" --emb-lm-lr "${EMB_LM_LR}" --ga-weight "${GA_WEIGHT}" --gd-weight "${GD_WEIGHT}" --optimizer adamw --dtype "${DTYPE}" --device-map "${DEVICE_MAP}"
  echo "Stage 2 begins before any held-out evaluation."
  rm -rf "${STAGE2}"
  python scripts/zsre_no_neutral_active_sensitive_row_repair.py --model-path "${STAGE1}/checkpoint" --training-visible-path "${VISIBLE}" --split-manifest "${MANIFEST}" --output-dir "${STAGE2}" --seed "${SEED}" --forget-num "${FORGET_NUM}" --repair-steps "${REPAIR_STEPS}" --repair-lr "${REPAIR_LR}" --repair-margin "${REPAIR_MARGIN}" --repair-l2 "${REPAIR_L2}" --batch-size "${REPAIR_BATCH_SIZE}" --candidate-scales "${CANDIDATE_SCALES}" --dtype "${DTYPE}" --device-map "${DEVICE_MAP}"
  echo "Final checkpoint frozen; held-out evaluation begins."
  python scripts/zsre_zero_unlearn_official_eval.py --model-dir "${STAGE2}/checkpoint" --zsre-path "${ZSRE}" --wikidata-dir "${WIKIDATA_DIR}" --out "${ROOT}/official_eval_locked.json" --method "SURE ZsRE no-neutral Emb+LM GA/GD plus sensitive-row repair" --unlearn-num "${FORGET_NUM}" --retain-num "${RETAIN_NUM}" --seed "${SEED}" --batch-size "${EVAL_BATCH_SIZE}" --dtype "${DTYPE}" --device-map "${DEVICE_MAP}"
  python scripts/zsre_zero_unlearn_official_eval.py --model-dir "${STAGE1}/checkpoint" --zsre-path "${ZSRE}" --wikidata-dir "${WIKIDATA_DIR}" --out "${ROOT}/official_eval_stage1_posthoc.json" --method "SURE ZsRE no-neutral Emb+LM GA/GD Stage1 posthoc" --unlearn-num "${FORGET_NUM}" --retain-num "${RETAIN_NUM}" --seed "${SEED}" --batch-size "${EVAL_BATCH_SIZE}" --dtype "${DTYPE}" --device-map "${DEVICE_MAP}"
  python - "${ROOT}" <<'PY'
import json,sys
r=sys.argv[1]
for n,f in (("STAGE1_POSTHOC","official_eval_stage1_posthoc.json"),("STAGE2_FINAL","official_eval_locked.json")):
 x=json.load(open(f"{r}/{f}")); print(n,"F-Eff",x["forget"]["Eff"],"F-Gen",x["forget"]["Gen"],"F-Spe",x["forget"]["Spe"],"R-Eff",x["retain"]["Eff"],"R-Gen",x["retain"]["Gen"],"R-Spe",x["retain"]["Spe"],"PPL",x["forget_PPL"])
PY
done
