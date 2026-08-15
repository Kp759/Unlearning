#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_tofu_zerounlearn_locked_our_method.sh [FULL_TOFU_MODEL]

The default FULL_TOFU_MODEL is the previously selected Llama-3.2-3B TOFU
checkpoint trained with LR=4e-5 and selected at epoch 5:
  outputs/tofu_full_utility_sweep_v7/lr4e-5_epochs6_slurm/checkpoint_epoch_5

Protocol per seed:
  Stage 1: 50 forget05 direct QAs, 0 retain, no paraphrases.
  Stage 2: same 50 direct QAs, 0 retain/utility/paraphrase data.
  Final:   selected-50 direct + paired paraphrases + unseen-150 forget05
           + 1000 retain95; native TOFU real-authors/world-facts evaluation.
EOF
}

if [[ $# -gt 1 ]]; then
  usage
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
FULL_TOFU_MODEL="${1:-${TOFU_FULL_MODEL_PATH:-outputs/tofu_full_utility_sweep_v7/lr4e-5_epochs6_slurm/checkpoint_epoch_5}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/tofu_zerounlearn_forget_only_locked_3b}"
PROTOCOL_DIR="${TOFU_PROTOCOL_DIR:-${OUTPUT_ROOT}/protocol}"
SEEDS_TEXT="${TOFU_SEEDS:-1 2 3 4 5 6 7 8 9 10}"
FORGET_NUM="${TOFU_FORGET_NUM:-50}"
RETAIN_EVAL_NUM="${TOFU_RETAIN_EVAL_NUM:-1000}"
DATASET_REVISION="${TOFU_DATASET_REVISION:-}"
DTYPE="${DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-single}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

# Stage 1: one default pass over the 50 visible QAs, preserving the exposure
# pattern of the earlier 200-row / 200-step TOFU recipe.
STAGE1_STEPS="${TOFU_STAGE1_STEPS:-50}"
STAGE1_BATCH_SIZE="${TOFU_STAGE1_BATCH_SIZE:-1}"
STAGE1_LR="${TOFU_STAGE1_LR:-0.0002}"
STAGE1_FORGET_WEIGHT="${TOFU_STAGE1_FORGET_WEIGHT:-1.0}"

# Stage 2 starts from the registered successful F05 repair family but removes
# every retain/utility constraint because those data are evaluation-only here.
TARGET_FORGET_PROB="${TOFU_TARGET_FORGET_PROB:-0.0003}"
TARGET_NLL_BUFFER="${TOFU_TARGET_NLL_BUFFER:-0.25}"
REPAIR_STEPS="${TOFU_REPAIR_STEPS:-5000}"
REPAIR_LR="${TOFU_REPAIR_LR:-0.02}"
REPAIR_RANK="${TOFU_REPAIR_RANK:-64}"
BASIS_MAX_ROWS="${TOFU_BASIS_MAX_ROWS:-2048}"
FORGET_HINGE_WEIGHT="${TOFU_FORGET_HINGE_WEIGHT:-100.0}"
HARDEST_FORGET_HINGE_WEIGHT="${TOFU_HARDEST_FORGET_HINGE_WEIGHT:-25.0}"
DELTA_L2_LAMBDA="${TOFU_DELTA_L2_LAMBDA:-1e-5}"
REPAIR_BATCH_SIZE="${TOFU_REPAIR_BATCH_SIZE:-8}"
MAX_LENGTH="${TOFU_MAX_LENGTH:-256}"

RUN_LOCKED_EVAL="${RUN_LOCKED_EVAL:-1}"
RUN_LOCKED_GENERATION="${RUN_LOCKED_GENERATION:-1}"
RUN_NATIVE_EVAL="${RUN_NATIVE_EVAL:-1}"
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

TOFU unlearning must start from the TOFU-finetuned Full model, not raw Llama.
The selected project checkpoint is the LR=4e-5, epoch-5 model above.  If Wulver
has it at another location, pass that directory as the first argument or set
TOFU_FULL_MODEL_PATH.  No fallback to a raw base model is performed.
EOF
  exit 2
fi
if [[ ! -f "${FULL_TOFU_MODEL}/config.json" ]]; then
  echo "ERROR: ${FULL_TOFU_MODEL} is not a Hugging Face checkpoint (config.json missing)." >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}" "${PROTOCOL_DIR}"

echo "===== TOFU ZEROUnlearn-STYLE LOCKED PROTOCOL ====="
echo "Full-TOFU model: ${FULL_TOFU_MODEL}"
echo "Seeds: ${SEEDS[*]}"
echo "Stage1/2 access: ${FORGET_NUM} direct forget05 QAs; retain/paraphrase/utility=0"
echo "Final retain eval: ${RETAIN_EVAL_NUM} retain95 QAs"

BUILD_ARGS=(
  --output-dir "${PROTOCOL_DIR}"
  --seeds "${SEEDS[@]}"
  --forget-num "${FORGET_NUM}"
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
  NATIVE_SUMMARY="${NATIVE_EVAL_DIR}/tofu_locked_seed${SEED}_summary.json"
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
  echo "===== SEED ${SEED}: STAGE 1 — ${FORGET_NUM} DIRECT FORGET QAs ONLY ====="
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

  echo "===== SEED ${SEED}: STAGE 2 — FORGET-ONLY SPARSE LM-HEAD REPAIR ====="
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
      --batch-size "${REPAIR_BATCH_SIZE}" \
      --max-length "${MAX_LENGTH}" \
      --dtype "${DTYPE}" \
      --device-map "${DEVICE_MAP}"
  else
    echo "Seed ${SEED}: reusing Stage2 checkpoint ${REPAIR_CKPT}"
  fi
  test -d "${REPAIR_CKPT}"

  if [[ "${RUN_LOCKED_EVAL}" == "1" ]]; then
    echo "===== SEED ${SEED}: FINAL LOCKED EVAL — HELD-OUT DATA ENTERS HERE ====="
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
    echo "===== SEED ${SEED}: STANDARD TOFU UTILITY AXES (FINAL ONLY) ====="
    rm -rf "${NATIVE_EVAL_DIR}"
    "${PYTHON_BIN}" scripts/tofu_eval.py \
      --model-dir "${REPAIR_CKPT}" \
      --method "tofu_locked_seed${SEED}" \
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
    "schema_version": 1,
    "protocol": "tofu_zerounlearn_data_access_forget_only_locked",
    "seed": int(sys.argv[8]),
    "split_manifest": str(pathlib.Path(sys.argv[2]).resolve()),
    "full_tofu_starting_model": str(pathlib.Path(sys.argv[3]).resolve()),
    "full_tofu_selection": {"learning_rate": 4e-5, "selected_epoch": 5},
    "stage1_checkpoint": str(pathlib.Path(sys.argv[4]).resolve()),
    "stage2_checkpoint": str(pathlib.Path(sys.argv[5]).resolve()),
    "locked_evaluation": str(pathlib.Path(sys.argv[6]).resolve()),
    "native_evaluation": str(pathlib.Path(sys.argv[7]).resolve()),
    "training_data_access": {
        "forget05_direct_qas": ${FORGET_NUM},
        "retain95": 0,
        "forget_paraphrases": 0,
        "perturbed_answers": 0,
        "real_authors": 0,
        "world_facts": 0,
    },
    "final_evaluation_data": {
        "selected_forget_direct": ${FORGET_NUM},
        "selected_forget_paraphrases": ${FORGET_NUM},
        "remaining_forget05_facts": 200 - ${FORGET_NUM},
        "retain95": ${RETAIN_EVAL_NUM},
        "real_authors": ${N_REAL_AUTHORS},
        "world_facts": ${N_WORLD_FACTS},
    },
    "hyperparameters": {
        "stage1_steps": ${STAGE1_STEPS},
        "stage1_batch_size": ${STAGE1_BATCH_SIZE},
        "stage1_lr": ${STAGE1_LR},
        "target_forget_probability": ${TARGET_FORGET_PROB},
        "repair_steps": ${REPAIR_STEPS},
        "repair_lr": ${REPAIR_LR},
        "repair_rank": ${REPAIR_RANK},
    },
    "final_selection_uses_heldout_data": False,
}
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

  echo "Seed ${SEED} complete."
done

echo
echo "TOFU ZeroUnlearn-style locked track complete."
echo "Start model: ${FULL_TOFU_MODEL}"
echo "Outputs: ${OUTPUT_ROOT}/seed*/"
