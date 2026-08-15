#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_tofu_zerounlearn_locked_our_method.sh [FULL_TOFU_MODEL]

Default starting model:
  outputs/tofu_full_utility_sweep_v7_repro_20260815/lr4e-5_epochs6/checkpoint_epoch_5

Author-balanced protocol per seed:
  * choose 5 forget05 author blocks;
  * Stage 1/2 see 10 direct QAs per selected author = 50 training QAs;
  * final efficacy evaluates the same 50 direct QAs and their paraphrases;
  * same-author generalization evaluates the other 10 QAs/author = 50 held-out
    direct QAs plus their paraphrases;
  * utility evaluates 1,000 retain95 QAs;
  * Stage 1/2 see zero retain/paraphrase/utility records.
EOF
}

if [[ $# -gt 1 ]]; then
  usage
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
FULL_TOFU_MODEL="${1:-${TOFU_FULL_MODEL_PATH:-outputs/tofu_full_utility_sweep_v7_repro_20260815/lr4e-5_epochs6/checkpoint_epoch_5}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/tofu_author_balanced_locked_3b}"
PROTOCOL_DIR="${TOFU_PROTOCOL_DIR:-${OUTPUT_ROOT}/protocol}"
SEEDS_TEXT="${TOFU_SEEDS:-1 2 3 4 5 6 7 8 9 10}"
FORGET_AUTHORS="${TOFU_FORGET_AUTHORS:-5}"
QAS_PER_AUTHOR="${TOFU_QAS_PER_AUTHOR:-20}"
TRAIN_QAS_PER_AUTHOR="${TOFU_TRAIN_QAS_PER_AUTHOR:-10}"
FORGET_NUM=$(( FORGET_AUTHORS * TRAIN_QAS_PER_AUTHOR ))
HELDOUT_NUM=$(( FORGET_AUTHORS * (QAS_PER_AUTHOR - TRAIN_QAS_PER_AUTHOR) ))
RETAIN_EVAL_NUM="${TOFU_RETAIN_EVAL_NUM:-1000}"
DATASET_REVISION="${TOFU_DATASET_REVISION:-}"
DTYPE="${DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-single}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

# Stage 1: one pass over the 50 visible QAs by default.
STAGE1_STEPS="${TOFU_STAGE1_STEPS:-50}"
STAGE1_BATCH_SIZE="${TOFU_STAGE1_BATCH_SIZE:-1}"
STAGE1_LR="${TOFU_STAGE1_LR:-0.0002}"
STAGE1_FORGET_WEIGHT="${TOFU_STAGE1_FORGET_WEIGHT:-1.0}"

# Stage 2 defaults are intentionally bounded.  The previous rank-64 / LR .02 /
# unbounded repair was observed to destroy TOFU retain utility.
TARGET_FORGET_PROB="${TOFU_TARGET_FORGET_PROB:-0.0003}"
TARGET_NLL_BUFFER="${TOFU_TARGET_NLL_BUFFER:-0.25}"
REPAIR_STEPS="${TOFU_REPAIR_STEPS:-5000}"
REPAIR_LR="${TOFU_REPAIR_LR:-0.002}"
REPAIR_RANK="${TOFU_REPAIR_RANK:-8}"
BASIS_MAX_ROWS="${TOFU_BASIS_MAX_ROWS:-2048}"
MAX_DELTA_NORM="${TOFU_MAX_DELTA_NORM:-1.0}"
ROW_SELECTION="${TOFU_ROW_SELECTION:-all}"
ROWS_PER_EXAMPLE="${TOFU_ROWS_PER_EXAMPLE:-3}"
FORGET_HINGE_WEIGHT="${TOFU_FORGET_HINGE_WEIGHT:-100.0}"
HARDEST_FORGET_HINGE_WEIGHT="${TOFU_HARDEST_FORGET_HINGE_WEIGHT:-25.0}"
DELTA_L2_LAMBDA="${TOFU_DELTA_L2_LAMBDA:-1e-5}"
REPAIR_BATCH_SIZE="${TOFU_REPAIR_BATCH_SIZE:-8}"
MAX_LENGTH="${TOFU_MAX_LENGTH:-256}"

RUN_LOCKED_EVAL="${RUN_LOCKED_EVAL:-1}"
RUN_LOCKED_GENERATION="${RUN_LOCKED_GENERATION:-1}"
# Native tofu_eval samples forget05 independently and therefore does not match
# the author-balanced selected 50.  Keep it optional/off by default; locked eval
# is the protocol-defining evaluation.
RUN_NATIVE_EVAL="${RUN_NATIVE_EVAL:-0}"
RUN_REFERENCE_LOCKED_EVAL="${RUN_REFERENCE_LOCKED_EVAL:-1}"
N_REAL_AUTHORS="${TOFU_N_REAL_AUTHORS_EVAL:-100}"
N_WORLD_FACTS="${TOFU_N_WORLD_FACTS_EVAL:-117}"
N_PERTURBED="${TOFU_N_PERTURBED_EVAL:-50}"

read -r -a SEEDS <<< "${SEEDS_TEXT}"
if [[ "${#SEEDS[@]}" -eq 0 ]]; then
  echo "TOFU_SEEDS resolved to an empty list." >&2
  exit 2
fi

if [[ ! -d "${FULL_TOFU_MODEL}" ]]; then
  cat >&2 <<EOF
ERROR: Full-TOFU starting checkpoint is missing:
  ${FULL_TOFU_MODEL}

Pass the validated epoch-5 Full-TOFU directory as the first argument or set
TOFU_FULL_MODEL_PATH.  No fallback to raw Llama is performed.
EOF
  exit 2
fi
if [[ ! -f "${FULL_TOFU_MODEL}/config.json" ]]; then
  echo "ERROR: ${FULL_TOFU_MODEL} is not a Hugging Face checkpoint (config.json missing)." >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}" "${PROTOCOL_DIR}"

echo "===== TOFU AUTHOR-BALANCED LOCKED PROTOCOL ====="
echo "Full-TOFU model: ${FULL_TOFU_MODEL}"
echo "Seeds: ${SEEDS[*]}"
echo "Selected authors/seed: ${FORGET_AUTHORS}"
echo "Train QAs/author: ${TRAIN_QAS_PER_AUTHOR}/${QAS_PER_AUTHOR}"
echo "Stage1/2 access: ${FORGET_NUM} direct forget QAs; retain/paraphrase/utility=0"
echo "Same-author heldout eval: ${HELDOUT_NUM} direct + ${HELDOUT_NUM} paraphrases"
echo "Final retain eval: ${RETAIN_EVAL_NUM} retain95 QAs"
echo "Repair: rank=${REPAIR_RANK} lr=${REPAIR_LR} max_delta_norm=${MAX_DELTA_NORM} rows=${ROW_SELECTION}"

BUILD_ARGS=(
  --output-dir "${PROTOCOL_DIR}"
  --seeds "${SEEDS[@]}"
  --forget-authors "${FORGET_AUTHORS}"
  --qas-per-author "${QAS_PER_AUTHOR}"
  --train-qas-per-author "${TRAIN_QAS_PER_AUTHOR}"
  --retain-num "${RETAIN_EVAL_NUM}"
)
if [[ -n "${DATASET_REVISION}" ]]; then
  BUILD_ARGS+=(--dataset-revision "${DATASET_REVISION}")
fi
"${PYTHON_BIN}" scripts/build_tofu_zerounlearn_locked_split.py "${BUILD_ARGS[@]}"

for SEED in "${SEEDS[@]}"; do
  SEED_PROTOCOL="${PROTOCOL_DIR}/seed${SEED}"
  TRAIN_FORGET="${SEED_PROTOCOL}/train_visible/forget.json"
  EVAL_ONLY_DIR="${SEED_PROTOCOL}/eval_only"
  SPLIT_MANIFEST="${SEED_PROTOCOL}/split_manifest.json"

  SEED_ROOT="${OUTPUT_ROOT}/seed${SEED}"
  STAGE1_DIR="${SEED_ROOT}/setting5e_forget_only"
  STAGE1_CKPT="${STAGE1_DIR}/checkpoint"
  REPAIR_DIR="${SEED_ROOT}/repair_forget_only"
  REPAIR_CKPT="${REPAIR_DIR}/checkpoint"
  LOCKED_EVAL="${SEED_ROOT}/locked_eval.json"
  NATIVE_EVAL_DIR="${SEED_ROOT}/native_eval"
  NATIVE_SUMMARY="${NATIVE_EVAL_DIR}/tofu_author_balanced_seed${SEED}_summary.json"
  RUN_MANIFEST="${SEED_ROOT}/run_manifest.json"

  mkdir -p "${SEED_ROOT}"
  test -f "${TRAIN_FORGET}"
  test -d "${EVAL_ONLY_DIR}"
  test -f "${SPLIT_MANIFEST}"

  if [[ "${SKIP_EXISTING}" == "1" && -f "${LOCKED_EVAL}" && ( "${RUN_NATIVE_EVAL}" != "1" || -f "${NATIVE_SUMMARY}" ) ]]; then
    echo "Seed ${SEED}: final evaluation artifacts already exist; skipping."
    continue
  fi

  echo
  echo "===== SEED ${SEED}: STAGE 1 — ${FORGET_AUTHORS} AUTHORS x ${TRAIN_QAS_PER_AUTHOR} QAs ====="
  if [[ ! -d "${STAGE1_CKPT}" ]]; then
    "${PYTHON_BIN}" scripts/tofu_forget_only_setting5e.py \
      --model-path "${FULL_TOFU_MODEL}" \
      --forget-json "${TRAIN_FORGET}" \
      --output-dir "${STAGE1_DIR}" \
      --seed "${SEED}" \
      --forget-num "${FORGET_NUM}" \
      --steps "${STAGE1_STEPS}" \
      --batch-size "${STAGE1_BATCH_SIZE}" \
      --emb-lm-lr "${STAGE1_LR}" \
      --forget-weight "${STAGE1_FORGET_WEIGHT}" \
      --optimizer adamw \
      --dtype "${DTYPE}" \
      --device-map "${DEVICE_MAP}"
  else
    echo "Seed ${SEED}: reusing Stage1 checkpoint ${STAGE1_CKPT}"
  fi
  test -d "${STAGE1_CKPT}"

  echo "===== SEED ${SEED}: STAGE 2 — BOUNDED FORGET-ONLY LM-HEAD REPAIR ====="
  if [[ ! -d "${REPAIR_CKPT}" ]]; then
    rm -rf "${REPAIR_DIR}"
    "${PYTHON_BIN}" scripts/tofu_forget_only_active_repair.py \
      --model-path "${STAGE1_CKPT}" \
      --forget-json "${TRAIN_FORGET}" \
      --output-dir "${REPAIR_DIR}" \
      --seed "${SEED}" \
      --forget-num "${FORGET_NUM}" \
      --target-forget-answer-probability "${TARGET_FORGET_PROB}" \
      --target-nll-buffer "${TARGET_NLL_BUFFER}" \
      --forget-hinge-weight "${FORGET_HINGE_WEIGHT}" \
      --hardest-forget-hinge-weight "${HARDEST_FORGET_HINGE_WEIGHT}" \
      --delta-l2-lambda "${DELTA_L2_LAMBDA}" \
      --repair-steps "${REPAIR_STEPS}" \
      --repair-lr "${REPAIR_LR}" \
      --repair-optimizer adamw \
      --repair-rank "${REPAIR_RANK}" \
      --basis-max-rows "${BASIS_MAX_ROWS}" \
      --max-delta-norm "${MAX_DELTA_NORM}" \
      --row-selection "${ROW_SELECTION}" \
      --rows-per-example "${ROWS_PER_EXAMPLE}" \
      --batch-size "${REPAIR_BATCH_SIZE}" \
      --max-length "${MAX_LENGTH}" \
      --dtype "${DTYPE}" \
      --device-map "${DEVICE_MAP}" \
      --save-best-effort
  else
    echo "Seed ${SEED}: reusing Stage2 checkpoint ${REPAIR_CKPT}"
  fi
  test -d "${REPAIR_CKPT}"

  if [[ "${RUN_LOCKED_EVAL}" == "1" ]]; then
    echo "===== SEED ${SEED}: AUTHOR-BALANCED FINAL LOCKED EVAL ====="
    LOCKED_ARGS=(
      --model-dir "${REPAIR_CKPT}"
      --eval-dir "${EVAL_ONLY_DIR}"
      --output "${LOCKED_EVAL}"
      --seed "${SEED}"
      --dtype "${DTYPE}"
      --max-length "${MAX_LENGTH}"
    )
    if [[ "${RUN_REFERENCE_LOCKED_EVAL}" == "1" ]]; then
      LOCKED_ARGS+=(--reference-model-dir "${FULL_TOFU_MODEL}")
    fi
    if [[ "${RUN_LOCKED_GENERATION}" != "1" ]]; then
      LOCKED_ARGS+=(--skip-generation)
    fi
    "${PYTHON_BIN}" scripts/tofu_zerounlearn_locked_eval.py "${LOCKED_ARGS[@]}"
  fi

  if [[ "${RUN_NATIVE_EVAL}" == "1" ]]; then
    echo "===== SEED ${SEED}: OPTIONAL STANDARD TOFU AXES ====="
    echo "NOTE: native forget05 sampling is independent of the author-balanced selected 50."
    rm -rf "${NATIVE_EVAL_DIR}"
    "${PYTHON_BIN}" scripts/tofu_eval.py \
      --model-dir "${REPAIR_CKPT}" \
      --method "tofu_author_balanced_seed${SEED}" \
      --forget-split forget05 \
      --retain-split retain95 \
      --output-dir "${NATIVE_EVAL_DIR}" \
      --seed "${SEED}" \
      --n-forget-eval "${FORGET_NUM}" \
      --n-retain-eval "${RETAIN_EVAL_NUM}" \
      --n-real-authors-eval "${N_REAL_AUTHORS}" \
      --n-world-facts-eval "${N_WORLD_FACTS}" \
      --n-perturbed-eval "${N_PERTURBED}"
  fi

  "${PYTHON_BIN}" - \
    "${RUN_MANIFEST}" "${SPLIT_MANIFEST}" "${FULL_TOFU_MODEL}" \
    "${STAGE1_CKPT}" "${REPAIR_CKPT}" "${LOCKED_EVAL}" \
    "${NATIVE_SUMMARY}" "${SEED}" <<PY
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = {
    "schema_version": 2,
    "protocol": "tofu_author_balanced_forget_only_locked_v1",
    "seed": int(sys.argv[8]),
    "split_manifest": str(pathlib.Path(sys.argv[2]).resolve()),
    "full_tofu_starting_model": str(pathlib.Path(sys.argv[3]).resolve()),
    "full_tofu_selection": {"learning_rate": 4e-5, "selected_epoch": 5},
    "stage1_checkpoint": str(pathlib.Path(sys.argv[4]).resolve()),
    "stage2_checkpoint": str(pathlib.Path(sys.argv[5]).resolve()),
    "locked_evaluation": str(pathlib.Path(sys.argv[6]).resolve()),
    "native_evaluation": str(pathlib.Path(sys.argv[7]).resolve()),
    "training_data_access": {
        "selected_forget_authors": ${FORGET_AUTHORS},
        "direct_qas_per_author": ${TRAIN_QAS_PER_AUTHOR},
        "direct_forget_qas": ${FORGET_NUM},
        "retain95": 0,
        "forget_paraphrases": 0,
        "perturbed_answers": 0,
        "real_authors": 0,
        "world_facts": 0,
    },
    "final_evaluation_data": {
        "seen_forget_direct": ${FORGET_NUM},
        "seen_forget_paraphrases": ${FORGET_NUM},
        "same_author_unseen_direct": ${HELDOUT_NUM},
        "same_author_unseen_paraphrases": ${HELDOUT_NUM},
        "retain95": ${RETAIN_EVAL_NUM},
    },
    "hyperparameters": {
        "stage1_steps": ${STAGE1_STEPS},
        "stage1_batch_size": ${STAGE1_BATCH_SIZE},
        "stage1_lr": ${STAGE1_LR},
        "target_forget_probability": ${TARGET_FORGET_PROB},
        "repair_steps": ${REPAIR_STEPS},
        "repair_lr": ${REPAIR_LR},
        "repair_rank": ${REPAIR_RANK},
        "max_delta_norm": ${MAX_DELTA_NORM},
        "row_selection": "${ROW_SELECTION}",
    },
    "final_selection_uses_heldout_data": False,
}
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

  echo "Seed ${SEED} complete."
done

echo
echo "TOFU author-balanced locked track complete."
echo "Start model: ${FULL_TOFU_MODEL}"
echo "Outputs: ${OUTPUT_ROOT}/seed*/"
