#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MQUAKE="${1:-data/MQuAKE-CF-3k-v2.json}"
WIKIDATA_DIR="${WIKIDATA_DIR:-data/wikidata_aws_diag}"
SEED="${MQUAKE_SEED:-1}"
FORGET_NUM="${MQUAKE_FORGET_NUM:-50}"
RETAIN_NUM="${MQUAKE_RETAIN_EVAL_NUM:-1000}"
SOURCE_ROOT="${SOURCE_ROOT:-outputs/aws_mquake_sure_gagd_output_only_restore_seed1_3b/seed${SEED}}"
SOURCE_STAGE1="${SOURCE_STAGE1:-${SOURCE_ROOT}/stage1_emb_lm_gagd_output_only_restore/checkpoint}"
VISIBLE="${SOURCE_ROOT}/protocol_no_neutral/training_visible_forget.json"
MANIFEST="${SOURCE_ROOT}/protocol_no_neutral/split_manifest.json"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/aws_mquake_sure_gagd_projected_restore_seed1_3b}"
ROOT="${OUTPUT_ROOT}/seed${SEED}"
PROJECTED="${ROOT}/stage1_projected_min_norm"
STAGE2="${ROOT}/stage2_sensitive_row_repair"
DTYPE="${DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-single}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"

for p in "${SOURCE_STAGE1}" "${VISIBLE}" "${MANIFEST}"; do
  test -e "${p}" || { echo "missing required source artifact: ${p}" >&2; exit 2; }
done
mkdir -p "${ROOT}"

echo "===== PROJECT EXISTING OUTPUT-ONLY STAGE1 TO MINIMUM-NORM FORGET SUBSPACE ====="
rm -rf "${PROJECTED}"
python scripts/mquake_project_sensitive_rows_to_forget_subspace.py \
  --model-path "${SOURCE_STAGE1}" \
  --training-visible-path "${VISIBLE}" \
  --split-manifest "${MANIFEST}" \
  --output-dir "${PROJECTED}" \
  --seed "${SEED}" --forget-num "${FORGET_NUM}" \
  --batch-size 8 --svd-relative-tol 1e-6 \
  --dtype "${DTYPE}" --device-map "${DEVICE_MAP}"

echo "===== STAGE2 SPARSE SENSITIVE-ROW CLEANUP ====="
rm -rf "${STAGE2}"
python scripts/mquake_forget_only_no_neutral.py \
  --model-path "${PROJECTED}/checkpoint" \
  --training-visible-path "${VISIBLE}" --split-manifest "${MANIFEST}" \
  --output-dir "${STAGE2}" --seed "${SEED}" --forget-num "${FORGET_NUM}" \
  --stage1-steps 0 --stage1-lr 0.005 --stage1-margin 0.25 \
  --stage2-steps 800 --stage2-lr 0.005 --stage2-margin 0.05 --l2 0.000001 \
  --batch-size 8 --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --candidate-scales "1,.875,.75,.625,.5,.375,.25,.1875,.125,.09375,.0625,.03125,.015625,.0078125,0" \
  --dtype "${DTYPE}" --device-map "${DEVICE_MAP}"

echo "Final projected+repaired checkpoint frozen; held-out evaluation begins."
python scripts/mquake_zero_unlearn_official_eval.py \
  --model-dir "${STAGE2}/checkpoint" --mquake-path "${MQUAKE}" --wikidata-dir "${WIKIDATA_DIR}" \
  --out "${ROOT}/official_eval_locked.json" --split-manifest "${ROOT}/official_eval_split_manifest.json" \
  --method "SURE MQuAKE GA/GD output-only minimum-norm projection plus sensitive-row repair" \
  --unlearn-num "${FORGET_NUM}" --retain-num "${RETAIN_NUM}" --seed "${SEED}" \
  --batch-size "${EVAL_BATCH_SIZE}" --dtype "${DTYPE}" --device-map "${DEVICE_MAP}"

# Post-hoc only: evaluate the projected Stage1 after the final checkpoint is frozen.
python scripts/mquake_zero_unlearn_official_eval.py \
  --model-dir "${PROJECTED}/checkpoint" --mquake-path "${MQUAKE}" --wikidata-dir "${WIKIDATA_DIR}" \
  --out "${ROOT}/official_eval_stage1_projected_posthoc.json" \
  --method "SURE MQuAKE projected minimum-norm Stage1 posthoc" \
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
p=json.load(open(f"{r}/stage1_projected_min_norm/projection_report.json"))
print("PROJECTION","rank",p["hidden_subspace_rank"],"rows",p["sensitive_lm_head_rows"],"norm_before",p["delta_fro_norm_before"],"norm_after",p["delta_fro_norm_after_projection"],"retained_fraction",p["norm_retained_fraction"],"max_action_error",p["max_direct_forget_logit_action_error_fp32"])
for name,f in (("STAGE1_PROJECTED","official_eval_stage1_projected_posthoc.json"),("STAGE2_FINAL","official_eval_locked.json")):
 x=json.load(open(f"{r}/{f}"))
 print(name,"F-Eff",x["forget"]["Eff"],"F-AtomicGen",x["forget"].get("AtomicGen"),"R-Eff",x["retain"]["Eff"],"R-AtomicGen",x["retain"].get("AtomicGen"),"PPL",x["forget_PPL"])
m=json.load(open(f"{r}/multihop_eval_final.json"))
for mode,v in m["results"].items():
 print("MULTIHOP",mode,"MHLeak_exact_any",v["MHLeak_exact_any"],"MHLeak_contains_any",v["MHLeak_contains_any"])
PY
