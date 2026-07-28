#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_three_benchmark_experiments.sh DATASET [MODEL_PATH]

DATASET: mcf | tofu | zsre | all

Important environment overrides:
  OUTPUT_ROOT       outputs/three_benchmark
  PYTHON_BIN        python
  DRY_RUN           0 (set to 1 to print commands only)
  SKIP_EXISTING     1
  MCF_SEEDS         "0 1 2 3 4 5 6 7 8 9"
  TOFU_SEED         42 (the reviewed TOFU ZeroUnlearn protocol is fixed to 42)
  ZSRE_SEEDS        "1 2 3 4 5 6 7 8 9 10"
EOF
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
  exit 2
fi

DATASET="$1"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="${2:-${MODEL_PATH:-}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/three_benchmark}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
DTYPE="${DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-single}"

if [[ -z "${MODEL_PATH}" ]]; then
  echo "MODEL_PATH is required as the second argument or environment variable." >&2
  exit 2
fi

run_cmd() {
  printf '+ '
  printf '%q ' "$@"
  printf '\n'
  if [[ "${DRY_RUN}" != "1" ]]; then
    "$@"
  fi
}

contains_dataset() {
  [[ "${DATASET}" == "$1" || "${DATASET}" == "all" ]]
}

run_mcf() {
  local root="${OUTPUT_ROOT}/mcf"
  local zero_root="${root}/zerounlearn"
  local gagd_root="${root}/gagd"
  local repair_root="${root}/repair"
  local aggregate_root="${root}/aggregate"
  local seeds_text="${MCF_SEEDS:-0 1 2 3 4 5 6 7 8 9}"
  local forget_num="${MCF_FORGET_NUM:-50}"
  local retain_num="${MCF_RETAIN_NUM:-1000}"
  local mcf_path="${MCF_PATH:-data/multi_counterfact.json}"
  local wikidata_dir="${WIKIDATA_DIR:-data/wikidata}"
  local steps="${MCF_STEPS:-250}"
  local full_lr="${MCF_FULL_LR:-1e-3}"
  local emb_lm_lr="${MCF_EMB_LM_LR:-1e-4}"
  local forget_weight="${MCF_FORGET_WEIGHT:-2.0}"
  local retain_weight="${MCF_RETAIN_WEIGHT:-1.0}"
  local forget_margin="${MCF_FORGET_MARGIN:-1.0}"
  local retain_batch_size="${MCF_RETAIN_BATCH_SIZE:-4}"
  local -a seeds
  local -a modes=(
    full_all_tokens
    full_selective_tokens
    emb_lm_all_tokens
    emb_lm_selective_tokens
    emb_lm_all_restore_post_training_true
  )
  read -r -a seeds <<< "${seeds_text}"

  mkdir -p "${zero_root}" "${gagd_root}" "${repair_root}" "${aggregate_root}"

  local -a zero_args=(
    "${PYTHON_BIN}" scripts/run_zerounlearn_multiseed_mcf.py
    --seeds "${seeds[@]}"
    --model-path "${MODEL_PATH}"
    --mcf-path "${mcf_path}"
    --wikidata-dir "${wikidata_dir}"
    --output-root "${zero_root}"
    --forget-num "${forget_num}"
    --retain-num "${retain_num}"
    --sample-mode official
    --dtype bfloat16
  )
  if [[ "${SKIP_EXISTING}" == "1" ]]; then
    zero_args+=(--skip-completed)
  fi
  run_cmd "${zero_args[@]}"

  local seed mode seed_dir official_path setting5_checkpoint repair_result
  for seed in "${seeds[@]}"; do
    seed_dir="${gagd_root}/seed${seed}"
    mkdir -p "${seed_dir}"
    for mode in "${modes[@]}"; do
      official_path="${seed_dir}/official_eval/${mode}_official_eval.json"
      if [[ "${SKIP_EXISTING}" == "1" && -f "${official_path}" ]]; then
        echo "Skipping existing MCF ${mode} seed ${seed}: ${official_path}"
        continue
      fi
      local -a gagd_args=(
        "${PYTHON_BIN}" scripts/gagd_compare.py
        --dataset mcf
        --model-path "${MODEL_PATH}"
        --mcf-cache-path "${mcf_path}"
        --mcf-sample-mode official
        --official-sample-mode official
        --output-dir "${seed_dir}"
        --mode "${mode}"
        --forget-loss-type mcf_margin
        --forget-margin "${forget_margin}"
        --mcf-answer-field target_new
        --forget-num "${forget_num}"
        --retain-num "${retain_num}"
        --seed "${seed}"
        --steps "${steps}"
        --batch-size 1
        --retain-batch-size "${retain_batch_size}"
        --full-lr "${full_lr}"
        --emb-lm-lr "${emb_lm_lr}"
        --forget-weight "${forget_weight}"
        --retain-weight "${retain_weight}"
        --sampling-strategy epoch
        --post-training-new-true-alpha 0.75
        --post-training-new-retain-alpha 0.50
        --post-training-new-true-retain-alpha 0.25
        --dtype "${DTYPE}"
        --device-map "${DEVICE_MAP}"
        --wikidata-dir "${wikidata_dir}"
        --run-official-mcf-eval
      )
      if [[ "${mode}" == "emb_lm_all_restore_post_training_true" ]]; then
        gagd_args+=(--save-model)
      fi
      run_cmd "${gagd_args[@]}"
    done

    setting5_checkpoint="${seed_dir}/emb_lm_all_restore_post_training_true/checkpoint"
    repair_result="${repair_root}/seed${seed}/official_eval_selected.json"
    if [[ "${SKIP_EXISTING}" == "1" && -f "${repair_result}" ]]; then
      echo "Skipping existing MCF protected repair seed ${seed}: ${repair_result}"
    else
      if [[ "${DRY_RUN}" != "1" && ! -d "${setting5_checkpoint}" ]]; then
        echo "Missing Setting 5e checkpoint for MCF seed ${seed}: ${setting5_checkpoint}" >&2
        exit 2
      fi
      run_cmd env \
        OUT_ROOT="${repair_root}/seed${seed}" \
        MCF_PATH="${mcf_path}" \
        WIKIDATA_DIR="${wikidata_dir}" \
        SAMPLE_MODE=official \
        SEED="${seed}" \
        FORGET_NUM="${forget_num}" \
        RETAIN_NUM="${retain_num}" \
        DTYPE="${DTYPE}" \
        DEVICE_MAP="${DEVICE_MAP}" \
        bash scripts/run_gagd_active_case_repair.sh \
        "${setting5_checkpoint}" "${MODEL_PATH}"
    fi
  done

  run_cmd "${PYTHON_BIN}" scripts/aggregate_mcf_multimethod_results.py \
    --seeds "${seeds[@]}" \
    --base-pattern "${zero_root}/seed{seed}/base_seed{seed}_official_eval.json" \
    --zero-pattern "${zero_root}/seed{seed}/zerounlearn_seed{seed}_official_eval.json" \
    --gagd-pattern "${gagd_root}/seed{seed}/official_eval/{method}_official_eval.json" \
    --repair-pattern "${repair_root}/seed{seed}/official_eval_selected.json" \
    --output-dir "${aggregate_root}"
}

run_tofu() {
  local root="${OUTPUT_ROOT}/tofu"
  local pipeline_root="${root}/pipeline"
  local zero_root="${root}/zerounlearn"
  local seed="${TOFU_SEED:-42}"
  if [[ "${seed}" != "42" ]]; then
    echo "TOFU_SEED must remain 42 for the reviewed Original ZeroUnlearn comparison." >&2
    exit 2
  fi
  mkdir -p "${root}"

  run_cmd env \
    MODEL_PATH="${MODEL_PATH}" \
    OUTPUT_ROOT="${pipeline_root}" \
    SEED=42 \
    FORGET_SPLIT=forget05 \
    RETAIN_SPLIT=retain95 \
    FORGET_NUM=200 \
    RETAIN_NUM=1000 \
    DTYPE="${DTYPE}" \
    DEVICE_MAP="${DEVICE_MAP}" \
    RUN_NEIGHBORHOOD_REPAIR="${TOFU_RUN_NEIGHBORHOOD_REPAIR:-0}" \
    bash scripts/run_tofu_gagd_neighborhood_confidence.sh

  local reference="${pipeline_root}/evaluation/base_forget05_reference_truth_ratios.json"
  local base_summary="${pipeline_root}/evaluation/base_summary.json"
  if [[ "${DRY_RUN}" != "1" && ! -f "${reference}" ]]; then
    echo "Missing TOFU reference truth ratios: ${reference}" >&2
    exit 2
  fi
  run_cmd "${PYTHON_BIN}" scripts/run_zerounlearn_tofu.py \
    --model-path "${MODEL_PATH}" \
    --output-dir "${zero_root}" \
    --reference-truth-ratios "${reference}" \
    --base-summary "${base_summary}" \
    --framework-eval-dir "${pipeline_root}/evaluation"
}

run_zsre() {
  local root="${OUTPUT_ROOT}/zsre"
  local seeds_text="${ZSRE_SEEDS:-1 2 3 4 5 6 7 8 9 10}"
  mkdir -p "${root}"
  run_cmd env \
    OUT_ROOT="${root}" \
    SEEDS="${seeds_text}" \
    DTYPE="${DTYPE}" \
    DEVICE_MAP="${DEVICE_MAP}" \
    bash scripts/run_zsre_gagd_setting5e_active_repair.sh "${MODEL_PATH}"
}

case "${DATASET}" in
  mcf|tofu|zsre|all)
    ;;
  *)
    usage
    exit 2
    ;;
esac

if contains_dataset mcf; then
  run_mcf
fi
if contains_dataset tofu; then
  run_tofu
fi
if contains_dataset zsre; then
  run_zsre
fi
