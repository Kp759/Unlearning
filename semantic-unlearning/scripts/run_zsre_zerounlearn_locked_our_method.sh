#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_zsre_zerounlearn_locked_our_method.sh MODEL_PATH [ZSRE_PATH]

Locked ZeroUnlearn-style ZsRE SURE protocol:
  Stage 1: 50 sampled forget requested_rewrite records, 0 benchmark retain.
  Stage 2: same 50 direct rewrites only; no rephrase/locality/retain access.
  Final eval: original ZsRE file, same 50 forget + 1000 sampled retain.
              Rephrase and locality probes first enter at this final step.
EOF
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="$1"
ORIGINAL_ZSRE="${2:-${ZSRE_PATH:-data/zsre_mend_eval.json}}"
WIKIDATA_DIR="${WIKIDATA_DIR:-data/wikidata}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/zsre_zerounlearn_forget_only_locked_3b}"
SEEDS_TEXT="${ZSRE_SEEDS:-1 2 3 4 5 6 7 8 9 10}"
FORGET_NUM="${ZSRE_FORGET_NUM:-50}"
EVAL_RETAIN_NUM="${ZSRE_RETAIN_EVAL_NUM:-1000}"
DTYPE="${DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-single}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
SKIP_PPL="${SKIP_PPL:-0}"

# Stage 1: preserve established SURE Setting-5e forget hyperparameters.
STEPS="${ZSRE_STEPS:-600}"
BATCH_SIZE="${ZSRE_BATCH_SIZE:-1}"
EMB_LM_LR="${ZSRE_EMB_LM_LR:-0.0001}"
FORGET_WEIGHT="${ZSRE_FORGET_WEIGHT:-2.0}"
FORGET_MARGIN="${ZSRE_FORGET_MARGIN:-1.0}"

# Stage 2: preserve established ZsRE neutral-row active-repair controls, but
# remove all protected/retain/probe access.
REPAIR_STEPS="${REPAIR_STEPS:-800}"
REPAIR_LR="${REPAIR_LR:-0.005}"
REPAIR_OPTIMIZER="${REPAIR_OPTIMIZER:-adamw}"
ACTIVE_LOGIT_MARGIN="${ACTIVE_LOGIT_MARGIN:-0.25}"
SELECTION_LOGIT_MARGIN="${SELECTION_LOGIT_MARGIN:-0.05}"
REPAIR_RANK="${REPAIR_RANK:-0}"
REPAIR_L2_LAMBDA="${REPAIR_L2_LAMBDA:-0.000001}"
CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
CANDIDATE_SCALES="${CANDIDATE_SCALES:-1.0,0.875,0.75,0.625,0.5,0.375,0.25,0.1875,0.125,0.09375,0.0625,0.046875,0.03125,0.015625,0.0078125,0.0}"

read -r -a SEEDS <<< "${SEEDS_TEXT}"
if [[ "${#SEEDS[@]}" -eq 0 ]]; then
  echo "ZSRE_SEEDS resolved to an empty list" >&2
  exit 2
fi

test -f "${ORIGINAL_ZSRE}"
test -d "${WIKIDATA_DIR}"

for SEED in "${SEEDS[@]}"; do
  SEED_ROOT="${OUTPUT_ROOT}/seed${SEED}"
  PROTOCOL_DIR="${ZSRE_PROTOCOL_DIR:-${SEED_ROOT}/protocol}"
  REPAIR_VISIBLE="${PROTOCOL_DIR}/repair_visible_forget.json"
  SPLIT_MANIFEST="${PROTOCOL_DIR}/split_manifest.json"

  SETTING_DIR="${SEED_ROOT}/setting5e_forget_only"
  SETTING_CKPT="${SETTING_DIR}/emb_lm_all_restore_post_training_true/checkpoint"
  SETTING_CONFIG="${SETTING_DIR}/config_used.json"

  REPAIR_DIR="${SEED_ROOT}/repair_forget_only"
  REPAIR_CKPT="${REPAIR_DIR}/checkpoint"
  FINAL_EVAL="${SEED_ROOT}/official_eval_locked.json"
  RUN_MANIFEST="${SEED_ROOT}/run_manifest.json"

  mkdir -p "${SEED_ROOT}" "${PROTOCOL_DIR}"

  if [[ "${SKIP_EXISTING}" == "1" && -f "${FINAL_EVAL}" ]]; then
    echo "Seed ${SEED}: final locked ZsRE evaluation exists; skipping."
    continue
  fi

  echo
  echo "===== SEED ${SEED}: BUILD EXACT ZEROUnlearn ZsRE SPLIT ====="
  "${PYTHON_BIN}" scripts/build_zsre_zerounlearn_locked_split.py \
    --zsre-path "${ORIGINAL_ZSRE}" \
    --output-dir "${PROTOCOL_DIR}" \
    --seed "${SEED}" \
    --forget-num "${FORGET_NUM}" \
    --retain-num "${EVAL_RETAIN_NUM}"

  test -f "${REPAIR_VISIBLE}"
  test -f "${SPLIT_MANIFEST}"

  echo "===== SEED ${SEED}: STAGE 1 — 50 DIRECT FORGET / 0 RETAIN ====="
  if [[ ! -d "${SETTING_CKPT}" ]]; then
    "${PYTHON_BIN}" scripts/zsre_forget_only_setting5e.py \
      --model-path "${MODEL_PATH}" \
      --repair-visible-path "${REPAIR_VISIBLE}" \
      --output-dir "${SETTING_DIR}" \
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
  else
    echo "Seed ${SEED}: reusing locked forget-only Setting-5e checkpoint."
  fi
  test -d "${SETTING_CKPT}"
  test -f "${SETTING_CONFIG}"

  echo "===== SEED ${SEED}: STAGE 2 — DIRECT-REWRITE UNKNOWN-ROW REPAIR ====="
  rm -rf "${REPAIR_DIR}"
  "${PYTHON_BIN}" scripts/zsre_forget_only_active_repair.py \
    --model-path "${SETTING_CKPT}" \
    --base-model-path "${MODEL_PATH}" \
    --repair-visible-path "${REPAIR_VISIBLE}" \
    --output-dir "${REPAIR_DIR}" \
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

  test -d "${REPAIR_CKPT}"

  echo "===== SEED ${SEED}: FINAL OFFICIAL EVAL — ORIGINAL ZsRE REOPENED ====="
  echo "Rephrases, locality probes, and 1000 retain records first enter here."
  EVAL_ARGS=(
    --model-dir "${REPAIR_CKPT}"
    --zsre-path "${ORIGINAL_ZSRE}"
    --wikidata-dir "${WIKIDATA_DIR}"
    --out "${FINAL_EVAL}"
    --method "SURE-LM locked ZsRE"
    --unlearn-num "${FORGET_NUM}"
    --retain-num "${EVAL_RETAIN_NUM}"
    --seed "${SEED}"
    --batch-size "${EVAL_BATCH_SIZE}"
    --dtype "${DTYPE}"
    --device-map "${DEVICE_MAP}"
  )
  if [[ "${SKIP_PPL}" == "1" ]]; then
    EVAL_ARGS+=(--skip-ppl)
  fi
  "${PYTHON_BIN}" scripts/zsre_zero_unlearn_official_eval.py "${EVAL_ARGS[@]}"

  "${PYTHON_BIN}" - \
    "${RUN_MANIFEST}" "${SPLIT_MANIFEST}" "${SETTING_CKPT}" \
    "${REPAIR_CKPT}" "${FINAL_EVAL}" "${SEED}" <<PY
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "protocol": "zsre_zerounlearn_forget_only_locked_probes",
    "seed": int(sys.argv[6]),
    "split_manifest": str(pathlib.Path(sys.argv[2]).resolve()),
    "setting5e_checkpoint": str(pathlib.Path(sys.argv[3]).resolve()),
    "repair_checkpoint": str(pathlib.Path(sys.argv[4]).resolve()),
    "final_official_evaluation": str(pathlib.Path(sys.argv[5]).resolve()),
    "training_data_access": {
        "forget_records": ${FORGET_NUM},
        "benchmark_retain_records": 0,
        "prompt_types": ["requested_rewrite"],
        "rephrases": 0,
        "locality_prompts": 0,
    },
    "repair_data_access": {
        "forget_records": ${FORGET_NUM},
        "benchmark_retain_records": 0,
        "prompt_types": ["requested_rewrite"],
        "rephrases": 0,
        "locality_prompts": 0,
    },
    "final_evaluation_data": {
        "forget_records": ${FORGET_NUM},
        "retain_records": ${EVAL_RETAIN_NUM},
        "forget_rephrases_enabled": True,
        "forget_locality_enabled": True,
    },
    "final_selection_uses_heldout_gen_or_spe": False,
    "hyperparameters": {
        "setting5e_steps": ${STEPS},
        "emb_lm_lr": ${EMB_LM_LR},
        "forget_weight": ${FORGET_WEIGHT},
        "forget_margin": ${FORGET_MARGIN},
        "active_logit_margin": ${ACTIVE_LOGIT_MARGIN},
        "selection_logit_margin": ${SELECTION_LOGIT_MARGIN},
        "repair_steps": ${REPAIR_STEPS},
        "repair_lr": ${REPAIR_LR},
        "repair_rank": ${REPAIR_RANK},
    },
}
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

  echo "Seed ${SEED} complete: ${FINAL_EVAL}"
done

echo
echo "Locked ZeroUnlearn-style ZsRE SURE track complete."
echo "Stage 1/2 access: ${FORGET_NUM} direct forget, 0 benchmark retain/probes."
echo "Final evaluation: ${FORGET_NUM} forget + ${EVAL_RETAIN_NUM} retain."
echo "Results: ${OUTPUT_ROOT}/seed*/official_eval_locked.json"
