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
# MQuAKE instances contain multiple requested_rewrite facts.  Batch 1 under-
# covers the flattened forget set at 600 steps, unlike the 50-record MCF/ZsRE
# runs.  Keep the optimizer-step budget fixed at 600 and expose batch coverage.
BATCH_SIZE="${BATCH_SIZE:-8}"
RETAIN_BATCH_SIZE="${RETAIN_BATCH_SIZE:-4}"
REPAIR_STEPS="${REPAIR_STEPS:-600}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-4}"
FAIL_IF_TARGET_MISSED="${FAIL_IF_TARGET_MISSED:-1}"
RUN_MULTIHOP="${RUN_MULTIHOP:-1}"

if [[ "${FAIL_IF_TARGET_MISSED}" != "0" && "${FAIL_IF_TARGET_MISSED}" != "1" ]]; then
  echo "FAIL_IF_TARGET_MISSED must be 0 or 1"
  exit 2
fi
if [[ "${RUN_MULTIHOP}" != "0" && "${RUN_MULTIHOP}" != "1" ]]; then
  echo "RUN_MULTIHOP must be 0 or 1"
  exit 2
fi

mkdir -p "${OUT_ROOT}"

echo "MQuAKE configuration: steps=${STEPS}, forget_batch=${BATCH_SIZE}, retain_batch=${RETAIN_BATCH_SIZE}, repair_steps=${REPAIR_STEPS}"

PY_ARGS=(
  --model-path "${MODEL_PATH}"
  --output-dir "${OUT_ROOT}"
  --mquake-path "${MQUAKE_PATH}"
  --wikidata-dir "${WIKIDATA_DIR}"
  --seed "${SEED}"
  --forget-num "${FORGET_NUM}"
  --retain-num "${RETAIN_NUM}"
  --steps "${STEPS}"
  --batch-size "${BATCH_SIZE}"
  --retain-batch-size "${RETAIN_BATCH_SIZE}"
  --emb-lm-lr 1e-4
  --forget-weight 2.0
  --retain-weight 1.0
  --forget-margin 1.0
  --emb-lm-optimizer adamw
  --sampling-strategy epoch
  --repair-steps "${REPAIR_STEPS}"
  --repair-lr 5e-3
  --repair-rank 0
  --active-logit-margin 0.25
  --selection-logit-margin 0.05
  --retain-calibration-num 128
  --target-eff-max 0.0
  --utility-drop-tolerance 0.10
  --max-ppl-ratio 1.02
  --eval-batch-size "${EVAL_BATCH_SIZE}"
  --cache-batch-size "${CACHE_BATCH_SIZE}"
  --dtype "${DTYPE}"
  --device-map "${DEVICE_MAP}"
  --save-selected-checkpoint
)
if [[ "${FAIL_IF_TARGET_MISSED}" == "1" ]]; then
  PY_ARGS+=(--fail-if-target-missed)
else
  PY_ARGS+=(--no-fail-if-target-missed)
fi

python scripts/mquake_gagd_setting5e_active_repair.py "${PY_ARGS[@]}"

# Print the diagnostics before any outer paper gate so failed exploratory runs
# still show exactly where efficacy or utility was lost.
python - "${OUT_ROOT}/mquake_results.json" <<'PY'
import json
import pathlib
import sys

data = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
print("\n===== MQUAKE DIAGNOSTICS =====")
print(
    "instances/atomic facts: "
    f"forget={data['forget_num_instances']}/{data['forget_num_atomic_facts']}, "
    f"retain={data['retain_num_instances']}/{data['retain_num_atomic_facts']}"
)
for name in ("base", "setting5e", "candidate", "selected"):
    row = data[name]
    print(
        f"{name:10s} "
        f"Eff={row['forget'].get('Eff')} "
        f"RetainEff={row['retain'].get('Eff')} "
        f"PPL={row.get('PPL')}"
    )
repair = data.get("repair", {})
print(
    "repair: "
    f"active_before={repair.get('active_tokens_before')} "
    f"active_after_candidate={repair.get('active_tokens_after_candidate')} "
    f"candidate_scale={repair.get('candidate_scale')} "
    f"accepted={repair.get('candidate_accepted')} "
    f"reason={repair.get('selection_reason')}"
)
print("==============================\n")
PY

# Paper-level utility gate is against BASE, not merely against Setting 5e.  The
# Python runner already prevents repair-specific regressions; this final guard
# prevents accepting a zero-Eff model that paid for it with global utility.
python - "${OUT_ROOT}/mquake_results.json" <<'PY'
import json
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
if [[ "${RUN_MULTIHOP}" == "1" ]]; then
  python scripts/mquake_multihop_unlearning_eval.py \
    --model-dir "${OUT_ROOT}/selected_checkpoint" \
    --mquake-path "${MQUAKE_PATH}" \
    --split-manifest "${OUT_ROOT}/split_manifest.json" \
    --out "${OUT_ROOT}/multihop_unlearning_eval.json" \
    --mode both \
    --dtype "${DTYPE}" \
    --device-map "${DEVICE_MAP}"
fi

echo "MQuAKE run complete: ${OUT_ROOT}"
echo "Main result: ${OUT_ROOT}/mquake_results.json"
if [[ "${RUN_MULTIHOP}" == "1" ]]; then
  echo "Multi-hop: ${OUT_ROOT}/multihop_unlearning_eval.json"
fi
