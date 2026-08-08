#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 MODEL_PATH [SEED]"
  exit 2
fi

MODEL_PATH="$1"
SEED="${2:-${SEED:-0}}"
OUT_ROOT="${OUT_ROOT:-outputs/mquake_setting5e_active/seed${SEED}}"
MQUAKE_PATH="${MQUAKE_PATH:-data/MQuAKE-CF-3k-v2.json}"
WIKIDATA_DIR="${WIKIDATA_DIR:-data/wikidata}"
DTYPE="${DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-single}"

FORGET_NUM="${FORGET_NUM:-1000}"
RETAIN_NUM="${RETAIN_NUM:-1000}"
STEPS="${STEPS:-600}"
REPAIR_STEPS="${REPAIR_STEPS:-600}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-4}"

mkdir -p "${OUT_ROOT}"

python scripts/mquake_gagd_setting5e_active_repair.py \
  --model-path "${MODEL_PATH}" \
  --output-dir "${OUT_ROOT}" \
  --mquake-path "${MQUAKE_PATH}" \
  --wikidata-dir "${WIKIDATA_DIR}" \
  --seed "${SEED}" \
  --forget-num "${FORGET_NUM}" \
  --retain-num "${RETAIN_NUM}" \
  --steps "${STEPS}" \
  --batch-size 1 \
  --retain-batch-size 4 \
  --emb-lm-lr 1e-4 \
  --forget-weight 2.0 \
  --retain-weight 1.0 \
  --forget-margin 1.0 \
  --emb-lm-optimizer adamw \
  --sampling-strategy epoch \
  --repair-steps "${REPAIR_STEPS}" \
  --repair-lr 5e-3 \
  --repair-rank 0 \
  --active-logit-margin 0.25 \
  --selection-logit-margin 0.05 \
  --retain-calibration-num 128 \
  --target-eff-max 0.0 \
  --utility-drop-tolerance 0.10 \
  --max-ppl-ratio 1.02 \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --cache-batch-size "${CACHE_BATCH_SIZE}" \
  --dtype "${DTYPE}" \
  --device-map "${DEVICE_MAP}" \
  --save-selected-checkpoint \
  --fail-if-target-missed

# Paper-level utility gate is against BASE, not merely against Setting 5e.  The
# Python runner already prevents repair-specific regressions; this final guard
# prevents accepting a zero-Eff model that paid for it with global utility.
python - "${OUT_ROOT}/mquake_results.json" <<'PY'
import json
import math
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
data = json.loads(path.read_text(encoding="utf-8"))
base = data["base"]
selected = data["selected"]

base_ret = base["retain"]["Eff"]
sel_ret = selected["retain"]["Eff"]
base_ppl = base.get("PPL")
sel_ppl = selected.get("PPL")

fail = []
if base_ret is not None and sel_ret is not None and float(sel_ret) < float(base_ret) - 0.10:
    fail.append(
        f"retain Eff dropped by {float(base_ret)-float(sel_ret):.6f} pp "
        f"({base_ret} -> {sel_ret})"
    )
if base_ppl is not None and sel_ppl is not None and float(sel_ppl) > float(base_ppl) * 1.02:
    fail.append(
        f"PPL ratio {float(sel_ppl)/float(base_ppl):.6f} exceeds 1.02 "
        f"({base_ppl} -> {sel_ppl})"
    )

if fail:
    raise SystemExit("Paper utility gate failed: " + "; ".join(fail))
print(
    "Paper utility gate passed: "
    f"retain Eff {base_ret} -> {sel_ret}; PPL {base_ppl} -> {sel_ppl}"
)
PY

# Multi-hop is post-selection-only.  It never affects the checkpoint.
python scripts/mquake_multihop_unlearning_eval.py \
  --model-dir "${OUT_ROOT}/selected_checkpoint" \
  --mquake-path "${MQUAKE_PATH}" \
  --split-manifest "${OUT_ROOT}/split_manifest.json" \
  --out "${OUT_ROOT}/multihop_unlearning_eval.json" \
  --mode both \
  --dtype "${DTYPE}" \
  --device-map "${DEVICE_MAP}"

echo "MQuAKE run complete: ${OUT_ROOT}"
echo "Main result: ${OUT_ROOT}/mquake_results.json"
echo "Multi-hop: ${OUT_ROOT}/multihop_unlearning_eval.json"
