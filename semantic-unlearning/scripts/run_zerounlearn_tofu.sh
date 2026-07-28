#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

MODEL_PATH="${MODEL_PATH:-outputs/finetuned_model_3B_instruct}"
FRAMEWORK_OUTPUT_ROOT="${FRAMEWORK_OUTPUT_ROOT:-outputs/tofu_gagd_targeted_2e5}"
FRAMEWORK_EVAL_DIR="${FRAMEWORK_EVAL_DIR:-${FRAMEWORK_OUTPUT_ROOT}/evaluation}"
REFERENCE_TRUTH_RATIOS="${REFERENCE_TRUTH_RATIOS:-${FRAMEWORK_EVAL_DIR}/base_forget05_reference_truth_ratios.json}"
BASE_SUMMARY="${BASE_SUMMARY:-${FRAMEWORK_EVAL_DIR}/base_summary.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/zerounlearn_tofu}"

cmd=(
  python scripts/run_zerounlearn_tofu.py
  --model-path "${MODEL_PATH}"
  --output-dir "${OUTPUT_ROOT}"
  --reference-truth-ratios "${REFERENCE_TRUTH_RATIOS}"
  --base-summary "${BASE_SUMMARY}"
)

if [[ -d "${FRAMEWORK_EVAL_DIR}" ]]; then
  cmd+=(--framework-eval-dir "${FRAMEWORK_EVAL_DIR}")
fi
if [[ "${SAVE_MODEL:-0}" == "1" ]]; then
  cmd+=(--save-model)
fi
if [[ -n "${N_REAL_AUTHORS_EVAL:-}" ]]; then
  cmd+=(--n-real-authors-eval "${N_REAL_AUTHORS_EVAL}")
fi
if [[ -n "${N_WORLD_FACTS_EVAL:-}" ]]; then
  cmd+=(--n-world-facts-eval "${N_WORLD_FACTS_EVAL}")
fi
if [[ -n "${N_PERTURBED_EVAL:-}" ]]; then
  cmd+=(--n-perturbed-eval "${N_PERTURBED_EVAL}")
fi

"${cmd[@]}"

echo "ZeroUnlearn TOFU summary: ${OUTPUT_ROOT}/evaluation/original_zerounlearn_summary.json"
echo "TOFU comparison: ${OUTPUT_ROOT}/comparison/comparison_tofu.md"
