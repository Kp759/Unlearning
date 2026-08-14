#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL="${1:?MODEL required}"
MQUAKE="${2:-data/MQuAKE-CF-3k-v2.json}"
WIKIDATA_DIR="${WIKIDATA_DIR:-data/wikidata_aws_diag}"
SEED="${MQUAKE_SEED:-1}"
FORGET_NUM="${MQUAKE_FORGET_NUM:-50}"
RETAIN_NUM="${MQUAKE_RETAIN_EVAL_NUM:-1000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/aws_mquake_sure_collision_safe_seed1_3b}"
ROOT="${OUTPUT_ROOT}/seed${SEED}"
PROTOCOL="${ROOT}/protocol_no_neutral"
VISIBLE="${PROTOCOL}/training_visible_forget.json"
MANIFEST="${PROTOCOL}/split_manifest.json"
STAGE1="${ROOT}/stage1_collision_safe"
STAGE2="${ROOT}/stage2_sensitive_row_repair"
DTYPE="${DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-single}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
mkdir -p "${ROOT}"

python scripts/build_mquake_zerounlearn_locked_no_neutral_split.py \
  --mquake-path "${MQUAKE}" --output-dir "${PROTOCOL}" --seed "${SEED}" \
  --forget-num "${FORGET_NUM}" --retain-num "${RETAIN_NUM}"

rm -rf "${STAGE1}"
python scripts/mquake_no_neutral_stage1_collision_safe.py \
  --model-path "${MODEL}" --training-visible-path "${VISIBLE}" --split-manifest "${MANIFEST}" \
  --output-dir "${STAGE1}" --seed "${SEED}" --forget-num "${FORGET_NUM}" \
  --steps 600 --batch-size 1 --cache-batch-size 8 \
  --emb-lr 0.0001 --row-lr 0.0005 \
  --ga-weight 2.0 --gd-weight 1.0 --active-margin 0.05 \
  --context-weight 1.0 --context-batch-size 64 --row-l2 0.0001 --grad-clip 1.0 \
  --dtype "${DTYPE}" --device-map "${DEVICE_MAP}"

echo "===== STAGE2 SPARSE SENSITIVE-ROW CLEANUP ====="
rm -rf "${STAGE2}"
python scripts/mquake_forget_only_no_neutral.py \
  --model-path "${STAGE1}/checkpoint" --training-visible-path "${VISIBLE}" --split-manifest "${MANIFEST}" \
  --output-dir "${STAGE2}" --seed "${SEED}" --forget-num "${FORGET_NUM}" \
  --stage1-steps 0 --stage1-lr 0.005 --stage1-margin 0.25 \
  --stage2-steps 800 --stage2-lr 0.005 --stage2-margin 0.05 --l2 0.000001 \
  --batch-size 8 --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --candidate-scales "1,.875,.75,.625,.5,.375,.25,.1875,.125,.09375,.0625,.03125,.015625,.0078125,0" \
  --dtype "${DTYPE}" --device-map "${DEVICE_MAP}"

echo "Final checkpoint frozen; held-out evaluation begins."
python scripts/mquake_zero_unlearn_official_eval.py \
  --model-dir "${STAGE2}/checkpoint" --mquake-path "${MQUAKE}" --wikidata-dir "${WIKIDATA_DIR}" \
  --out "${ROOT}/official_eval_locked.json" --split-manifest "${ROOT}/official_eval_split_manifest.json" \
  --method "SURE MQuAKE collision-safe contextual GA/GD plus sensitive-row repair" \
  --unlearn-num "${FORGET_NUM}" --retain-num "${RETAIN_NUM}" --seed "${SEED}" \
  --batch-size "${EVAL_BATCH_SIZE}" --dtype "${DTYPE}" --device-map "${DEVICE_MAP}"

python scripts/mquake_zero_unlearn_official_eval.py \
  --model-dir "${STAGE1}/checkpoint" --mquake-path "${MQUAKE}" --wikidata-dir "${WIKIDATA_DIR}" \
  --out "${ROOT}/official_eval_stage1_posthoc.json" \
  --method "SURE MQuAKE collision-safe Stage1 posthoc" \
  --unlearn-num "${FORGET_NUM}" --retain-num "${RETAIN_NUM}" --seed "${SEED}" \
  --batch-size "${EVAL_BATCH_SIZE}" --dtype "${DTYPE}" --device-map "${DEVICE_MAP}"

python scripts/mquake_multihop_unlearning_eval.py \
  --model-dir "${STAGE2}/checkpoint" --mquake-path "${MQUAKE}" \
  --split-manifest "${ROOT}/official_eval_split_manifest.json" \
  --out "${ROOT}/multihop_eval_final.json" --mode both --batch-size 4 \
  --dtype "${DTYPE}" --device-map "${DEVICE_MAP}"

python - "${ROOT}" <<'PY'
import json,sys
r=sys.argv[1]
cfg=json.load(open(f"{r}/stage1_collision_safe/config_used.json"))
print("COLLISION_SAFE_STAGE1 delta_norm",cfg["sensitive_delta_fro_norm"],"ordinary_states",cfg["ordinary_context_hidden_states"],"correct_after",cfg["correct_after_restoration"])
for name,f in (("STAGE1_POSTHOC","official_eval_stage1_posthoc.json"),("STAGE2_FINAL","official_eval_locked.json")):
 x=json.load(open(f"{r}/{f}"))
 print(name,"F-Eff",x["forget"]["Eff"],"F-AtomicGen",x["forget"].get("AtomicGen"),"R-Eff",x["retain"]["Eff"],"R-AtomicGen",x["retain"].get("AtomicGen"),"PPL",x["forget_PPL"])
m=json.load(open(f"{r}/multihop_eval_final.json"))
for mode,v in m["results"].items(): print("MULTIHOP",mode,"MHLeak_exact_any",v["MHLeak_exact_any"],"MHLeak_contains_any",v["MHLeak_contains_any"])
PY
