#!/usr/bin/env bash
# Run an isolated grid of utility-preserving full-TOFU fine-tuning jobs.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAINER="${SCRIPT_DIR}/finetune_tofu_utility_preserving.py"

PYTHON_BIN="${PYTHON_BIN:-python}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/scratch/yl258/kp759/hf/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/../outputs/tofu_full_utility_sweep}"
LEARNING_RATES="${LEARNING_RATES:-1e-5 2e-5 5e-5}"
EPOCHS_LIST="${EPOCHS_LIST:-1 2 3 5}"
SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_RATIO="${WARMUP_RATIO:-0.10}"
MAX_LENGTH="${MAX_LENGTH:-256}"
DTYPE="${DTYPE:-bf16}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-1}"
SAVE_EVERY_EPOCH="${SAVE_EVERY_EPOCH:-1}"

mkdir -p "${OUTPUT_ROOT}"

if [[ "${GRADIENT_CHECKPOINTING}" == "1" ]]; then
  GRADIENT_CHECKPOINTING_FLAG="--gradient-checkpointing"
else
  GRADIENT_CHECKPOINTING_FLAG="--no-gradient-checkpointing"
fi

if [[ "${SAVE_EVERY_EPOCH}" == "1" ]]; then
  SAVE_EVERY_EPOCH_FLAG="--save-every-epoch"
else
  SAVE_EVERY_EPOCH_FLAG="--no-save-every-epoch"
fi

failures=0
for learning_rate in ${LEARNING_RATES}; do
  for epochs in ${EPOCHS_LIST}; do
    safe_lr="${learning_rate//./p}"
    safe_lr="${safe_lr//-/_}"
    run_dir="${OUTPUT_ROOT}/lr_${safe_lr}_epochs_${epochs}_seed_${SEED}"

    if [[ -d "${run_dir}" ]] && [[ -n "$(find "${run_dir}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
      if [[ -f "${run_dir}/finetune_metadata.json" ]] && \
         [[ -f "${run_dir}/epoch_${epochs}_probe.json" ]] && \
         [[ -d "${run_dir}/final" ]] && \
         grep -q '"status": "complete"' "${run_dir}/finetune_metadata.json"; then
        echo "Skipping completed existing run without overwriting: ${run_dir}"
        continue
      fi
      echo "ERROR: refusing to overwrite incomplete nonempty run: ${run_dir}" >&2
      failures=$((failures + 1))
      continue
    fi

    echo "Running LR=${learning_rate}, epochs=${epochs}, seed=${SEED}"
    if ! "${PYTHON_BIN}" "${TRAINER}" \
      --model-path "${BASE_MODEL_PATH}" \
      --output-dir "${run_dir}" \
      --learning-rate "${learning_rate}" \
      --epochs "${epochs}" \
      --batch-size "${BATCH_SIZE}" \
      --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}" \
      --weight-decay "${WEIGHT_DECAY}" \
      --warmup-ratio "${WARMUP_RATIO}" \
      --max-length "${MAX_LENGTH}" \
      --seed "${SEED}" \
      --dtype "${DTYPE}" \
      "${GRADIENT_CHECKPOINTING_FLAG}" \
      "${SAVE_EVERY_EPOCH_FLAG}"; then
      echo "ERROR: run failed: ${run_dir}" >&2
      failures=$((failures + 1))
    fi
  done
done

if ! "${PYTHON_BIN}" "${TRAINER}" \
  --summarize-sweep-root "${OUTPUT_ROOT}"; then
  echo "ERROR: could not produce sweep summaries" >&2
  failures=$((failures + 1))
fi

if [[ "${failures}" -ne 0 ]]; then
  echo "Sweep completed with ${failures} failed or protected run(s)." >&2
  exit 1
fi

echo "Sweep complete: ${OUTPUT_ROOT}/sweep_summary.csv"
echo "Sweep complete: ${OUTPUT_ROOT}/sweep_summary.md"
