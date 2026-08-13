#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

MODEL_PATH="${1:?Usage: bash scripts/run_zsre_stagewise_ppl_diagnostic.sh MODEL_PATH [ZSRE_PATH]}"
ORIGINAL_ZSRE="${2:-${ZSRE_PATH:-data/zsre_mend_eval.json}}"
WIKIDATA_DIR="${WIKIDATA_DIR:-data/wikidata}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/zsre_stagewise_ppl_diag_seeds1_10}"
SEEDS_TEXT="${ZSRE_DIAG_SEEDS:-1 10}"
FORGET_NUM="${ZSRE_FORGET_NUM:-50}"
RETAIN_NUM="${ZSRE_RETAIN_EVAL_NUM:-1000}"
DTYPE="${DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-single}"

STEPS="${ZSRE_STEPS:-600}"
BATCH_SIZE="${ZSRE_BATCH_SIZE:-1}"
EMB_LM_LR="${ZSRE_EMB_LM_LR:-0.0001}"
FORGET_WEIGHT="${ZSRE_FORGET_WEIGHT:-2.0}"
FORGET_MARGIN="${ZSRE_FORGET_MARGIN:-1.0}"

REPAIR_STEPS="${REPAIR_STEPS:-800}"
REPAIR_LR="${REPAIR_LR:-0.005}"
REPAIR_OPTIMIZER="${REPAIR_OPTIMIZER:-adamw}"
ACTIVE_LOGIT_MARGIN="${ACTIVE_LOGIT_MARGIN:-0.25}"
SELECTION_LOGIT_MARGIN="${SELECTION_LOGIT_MARGIN:-0.05}"
REPAIR_RANK="${REPAIR_RANK:-0}"
REPAIR_L2_LAMBDA="${REPAIR_L2_LAMBDA:-0.000001}"
CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-8}"
CANDIDATE_SCALES="${CANDIDATE_SCALES:-1.0,0.875,0.75,0.625,0.5,0.375,0.25,0.1875,0.125,0.09375,0.0625,0.046875,0.03125,0.015625,0.0078125,0.0}"

mkdir -p "${OUTPUT_ROOT}"
test -f "${ORIGINAL_ZSRE}"
test -d "${WIKIDATA_DIR}"

BASE_PPL_JSON="${OUTPUT_ROOT}/base_ppl.json"
python scripts/eval_ppl_only.py \
  --model-path "${MODEL_PATH}" \
  --wikidata-dir "${WIKIDATA_DIR}" \
  --out "${BASE_PPL_JSON}" \
  --label base \
  --dtype "${DTYPE}" \
  --device-map "${DEVICE_MAP}"

read -r -a SEEDS <<< "${SEEDS_TEXT}"
for SEED in "${SEEDS[@]}"; do
  SEED_ROOT="${OUTPUT_ROOT}/seed${SEED}"
  PROTOCOL_DIR="${SEED_ROOT}/protocol"
  REPAIR_VISIBLE="${PROTOCOL_DIR}/repair_visible_forget.json"
  STAGE1_DIR="${SEED_ROOT}/stage1"
  STAGE1_CKPT="${STAGE1_DIR}/emb_lm_all_restore_post_training_true/checkpoint"
  STAGE1_PPL_JSON="${SEED_ROOT}/stage1_ppl.json"
  STAGE2_DIR="${SEED_ROOT}/stage2"
  STAGE2_CKPT="${STAGE2_DIR}/checkpoint"
  STAGE2_PPL_JSON="${SEED_ROOT}/stage2_ppl.json"

  echo
  echo "===== seed ${SEED}: build locked split ====="
  rm -rf "${SEED_ROOT}"
  mkdir -p "${SEED_ROOT}" "${PROTOCOL_DIR}"
  python scripts/build_zsre_zerounlearn_locked_split.py \
    --zsre-path "${ORIGINAL_ZSRE}" \
    --output-dir "${PROTOCOL_DIR}" \
    --seed "${SEED}" \
    --forget-num "${FORGET_NUM}" \
    --retain-num "${RETAIN_NUM}"

  echo "===== seed ${SEED}: Stage 1 ====="
  python scripts/zsre_forget_only_setting5e.py \
    --model-path "${MODEL_PATH}" \
    --repair-visible-path "${REPAIR_VISIBLE}" \
    --output-dir "${STAGE1_DIR}" \
    --seed "${SEED}" \
    --forget-num "${FORGET_NUM}" \
    --steps "${STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --emb-lm-lr "${EMB_LM_LR}" \
    --forget-weight "${FORGET_WEIGHT}" \
    --forget-margin "${FORGET_MARGIN}" \
    --optimizer adamw \
    --dtype "${DTYPE}" \
    --device-map "${DEVICE_MAP}" \
    --post-training-new-true-alpha 0.75 \
    --post-training-new-retain-alpha 0.50 \
    --post-training-new-true-retain-alpha 0.25

  echo "===== seed ${SEED}: PPL after Stage 1 ====="
  python scripts/eval_ppl_only.py \
    --model-path "${STAGE1_CKPT}" \
    --wikidata-dir "${WIKIDATA_DIR}" \
    --out "${STAGE1_PPL_JSON}" \
    --label "seed${SEED}_stage1" \
    --dtype "${DTYPE}" \
    --device-map "${DEVICE_MAP}"

  echo "===== seed ${SEED}: Stage 2 ====="
  python scripts/zsre_forget_only_active_repair.py \
    --model-path "${STAGE1_CKPT}" \
    --base-model-path "${MODEL_PATH}" \
    --repair-visible-path "${REPAIR_VISIBLE}" \
    --output-dir "${STAGE2_DIR}" \
    --seed "${SEED}" \
    --forget-num "${FORGET_NUM}" \
    --repair-steps "${REPAIR_STEPS}" \
    --repair-lr "${REPAIR_LR}" \
    --repair-optimizer "${REPAIR_OPTIMIZER}" \
    --active-logit-margin "${ACTIVE_LOGIT_MARGIN}" \
    --selection-logit-margin "${SELECTION_LOGIT_MARGIN}" \
    --repair-rank "${REPAIR_RANK}" \
    --repair-l2-lambda "${REPAIR_L2_LAMBDA}" \
    --candidate-scales "${CANDIDATE_SCALES}" \
    --cache-batch-size "${CACHE_BATCH_SIZE}" \
    --dtype "${DTYPE}" \
    --device-map "${DEVICE_MAP}"

  echo "===== seed ${SEED}: PPL after Stage 2 ====="
  python scripts/eval_ppl_only.py \
    --model-path "${STAGE2_CKPT}" \
    --wikidata-dir "${WIKIDATA_DIR}" \
    --out "${STAGE2_PPL_JSON}" \
    --label "seed${SEED}_stage2" \
    --dtype "${DTYPE}" \
    --device-map "${DEVICE_MAP}"
done

python - "${OUTPUT_ROOT}" "${SEEDS_TEXT}" <<'PY'
import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
seeds = [int(x) for x in sys.argv[2].split()]
base = json.loads((root / "base_ppl.json").read_text())["ppl"]
rows = []
for seed in seeds:
    sroot = root / f"seed{seed}"
    s1 = json.loads((sroot / "stage1_ppl.json").read_text())["ppl"]
    s2 = json.loads((sroot / "stage2_ppl.json").read_text())["ppl"]
    repair = json.loads((sroot / "stage2/repair_summary.json").read_text())
    train_log = [json.loads(x) for x in (sroot / "stage1/emb_lm_all_restore_post_training_true/train_log.jsonl").read_text().splitlines() if x.strip()]
    last = train_log[-1]
    rows.append({
        "seed": seed,
        "base_ppl": float(base),
        "stage1_ppl": float(s1),
        "stage2_ppl": float(s2),
        "stage1_over_base_ratio": float(s1 / base),
        "stage2_over_stage1_ratio": float(s2 / s1),
        "stage2_over_base_ratio": float(s2 / base),
        "final_sensitive_target_nll": float(last["sensitive_target_nll"]),
        "final_neutral_target_nll": float(last["neutral_target_nll"]),
        "repair_delta_norm": float(repair["effective_delta_norm"]),
        "repair_scale": float(repair["selected_scale"]),
        "active_before": int(repair["active_rewrite_correct_tokens_before"]),
    })

payload = {
    "schema_version": 1,
    "kind": "zsre_stagewise_ppl_diagnostic",
    "seeds": seeds,
    "base_ppl": float(base),
    "benchmark_probe_access": {
        "stage1": "50 direct forget requested_rewrite only",
        "stage2": "same 50 direct forget requested_rewrite only",
        "ppl_only": "fixed Wikidata text only",
        "zsre_rephrases_loaded": 0,
        "zsre_locality_loaded": 0,
        "zsre_retain_loaded": 0,
    },
    "selection_or_tuning_use": False,
    "rows": rows,
}
(root / "stagewise_ppl_summary.json").write_text(json.dumps(payload, indent=2) + "\n")

lines = [
    "# ZsRE stage-wise PPL diagnostic",
    "",
    f"Base PPL: **{base:.4f}**",
    "",
    "| Seed | Base PPL | Stage-1 PPL | Stage-2 PPL | S1/Base | S2/S1 | Sensitive NLL | Repair norm | Scale |",
    "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
]
for r in rows:
    lines.append(
        f"| {r['seed']} | {r['base_ppl']:.4f} | {r['stage1_ppl']:.4f} | {r['stage2_ppl']:.4f} | "
        f"{r['stage1_over_base_ratio']:.3f} | {r['stage2_over_stage1_ratio']:.3f} | "
        f"{r['final_sensitive_target_nll']:.4f} | {r['repair_delta_norm']:.4f} | {r['repair_scale']:.4f} |"
    )
lines += [
    "",
    "Diagnostic only: held-out ZsRE rephrases, locality probes, and 1000 retain records were never loaded.",
]
(root / "stagewise_ppl_summary.md").write_text("\n".join(lines) + "\n")

print("\n===== STAGE-WISE PPL SUMMARY =====")
print((root / "stagewise_ppl_summary.md").read_text())
PY

echo "JSON: ${OUTPUT_ROOT}/stagewise_ppl_summary.json"
echo "MD:   ${OUTPUT_ROOT}/stagewise_ppl_summary.md"
