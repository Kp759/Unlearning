#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: bash scripts/run_mquake_sure_v80_contextual_mlp.sh MODEL_PATH [MQUAKE_PATH]" >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="$1"
MQUAKE_PATH="${2:-${MQUAKE_PATH:-data/MQuAKE-CF-3k-v2.json}}"
WIKIDATA_DIR="${WIKIDATA_DIR:-data/wikidata}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/mquake_sure_v80_contextual_mlp}"
SEEDS_TEXT="${MQUAKE_SEEDS:-1}"
FORGET_NUM="${MQUAKE_FORGET_NUM:-50}"
RETAIN_NUM="${MQUAKE_RETAIN_EVAL_NUM:-1000}"
DTYPE="${DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-single}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
SKIP_PPL="${SKIP_PPL:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

V80_STEPS="${V80_STEPS:-1200}"
V80_LR="${V80_LR:-0.002}"
V80_RANK="${V80_RANK:-4}"
V80_ALPHA="${V80_ALPHA:-4}"
V80_LAYER_INDEX="${V80_LAYER_INDEX:--4}"
V80_FORGET_WEIGHT="${V80_FORGET_WEIGHT:-10}"
V80_HARDEST_WEIGHT="${V80_HARDEST_WEIGHT:-2.5}"
V80_SAFE_WEIGHT="${V80_SAFE_WEIGHT:-10}"
V80_KL_WEIGHT="${V80_KL_WEIGHT:-2}"
V80_PROMPT_WEIGHT="${V80_PROMPT_WEIGHT:-1}"
V80_FACTOR_L2="${V80_FACTOR_L2:-0.0001}"
V80_ACTIVE_MARGIN="${V80_ACTIVE_MARGIN:-0.05}"
V80_SAFE_FLOOR="${V80_SAFE_FLOOR:-0.0001}"
V80_GRAD_CLIP="${V80_GRAD_CLIP:-1}"
V80_AUDIT_EVERY="${V80_AUDIT_EVERY:-25}"
V80_BISECTION_STEPS="${V80_BISECTION_STEPS:-14}"

read -r -a SEEDS <<< "${SEEDS_TEXT}"
test -f "${MODEL_PATH}/config.json"
test -f "${MQUAKE_PATH}"
if [[ "${SKIP_PPL}" != "1" ]]; then
  test -d "${WIKIDATA_DIR}"
fi

run_locked_eval() {
  local model_dir="$1"
  local out_path="$2"
  local manifest_path="$3"
  local seed="$4"
  local args=(
    --model-dir "${model_dir}"
    --mquake-path "${MQUAKE_PATH}"
    --wikidata-dir "${WIKIDATA_DIR}"
    --out "${out_path}"
    --split-manifest "${manifest_path}"
    --method "SURE-MQuAKE V8.0 contextual MLP low-rank repair"
    --unlearn-num "${FORGET_NUM}"
    --retain-num "${RETAIN_NUM}"
    --seed "${seed}"
    --batch-size "${EVAL_BATCH_SIZE}"
    --dtype "${DTYPE}"
    --device-map "${DEVICE_MAP}"
    --skip-atomic-gen
  )
  if [[ "${SKIP_PPL}" == "1" ]]; then args+=(--skip-ppl); fi
  "${PYTHON_BIN}" scripts/mquake_zero_unlearn_official_eval.py "${args[@]}"
}

for SEED in "${SEEDS[@]}"; do
  ROOT="${OUTPUT_ROOT}/seed${SEED}"
  PROTOCOL_DIR="${ROOT}/protocol"
  VISIBLE="${PROTOCOL_DIR}/repair_visible_forget.json"
  MANIFEST="${PROTOCOL_DIR}/split_manifest.json"
  REPAIR_DIR="${ROOT}/v80_contextual_mlp"
  CKPT="${REPAIR_DIR}/checkpoint"
  SUMMARY="${REPAIR_DIR}/repair_summary.json"
  EVAL_OUT="${ROOT}/official_eval_v80.json"
  EVAL_MANIFEST="${ROOT}/final_eval_split_manifest_v80.json"

  mkdir -p "${PROTOCOL_DIR}"
  echo "===== SEED ${SEED}: BUILD LOCKED MQuAKE SPLIT ====="
  "${PYTHON_BIN}" scripts/build_mquake_zerounlearn_locked_split.py \
    --mquake-path "${MQUAKE_PATH}" \
    --output-dir "${PROTOCOL_DIR}" \
    --seed "${SEED}" \
    --forget-num "${FORGET_NUM}" \
    --retain-num "${RETAIN_NUM}"

  echo "===== SEED ${SEED}: V8.0 CONTEXTUAL MLP REPAIR ====="
  if [[ "${SKIP_EXISTING}" != "1" || ! -f "${CKPT}/config.json" || ! -f "${SUMMARY}" ]]; then
    rm -rf "${CKPT}"
    "${PYTHON_BIN}" scripts/mquake_sure_contextual_mlp_v80.py \
      --model-path "${MODEL_PATH}" \
      --repair-visible-path "${VISIBLE}" \
      --split-manifest "${MANIFEST}" \
      --output-dir "${REPAIR_DIR}" \
      --seed "${SEED}" \
      --forget-num "${FORGET_NUM}" \
      --steps "${V80_STEPS}" \
      --lr "${V80_LR}" \
      --rank "${V80_RANK}" \
      --lora-alpha "${V80_ALPHA}" \
      --layer-index "${V80_LAYER_INDEX}" \
      --forget-hinge-weight "${V80_FORGET_WEIGHT}" \
      --hardest-forget-hinge-weight "${V80_HARDEST_WEIGHT}" \
      --safe-hinge-weight "${V80_SAFE_WEIGHT}" \
      --non-target-kl-weight "${V80_KL_WEIGHT}" \
      --prompt-adapter-weight "${V80_PROMPT_WEIGHT}" \
      --factor-l2-lambda "${V80_FACTOR_L2}" \
      --active-target-margin "${V80_ACTIVE_MARGIN}" \
      --safe-margin-floor "${V80_SAFE_FLOOR}" \
      --grad-clip "${V80_GRAD_CLIP}" \
      --batch-size "${BATCH_SIZE}" \
      --audit-every "${V80_AUDIT_EVERY}" \
      --bisection-steps "${V80_BISECTION_STEPS}" \
      --dtype "${DTYPE}" \
      --device-map "${DEVICE_MAP}"
  else
    echo "Reusing ${CKPT}"
  fi

  AUDIT="$("${PYTHON_BIN}" -c 'import json,sys; d=json.load(open(sys.argv[1])); m=d["materialized_bf16_metrics"]; print(int(m["official_active_sensitive_token_count"]), int(m["base_active_buffer_unmet_count"]), int(m["base_safe_failed_count"]), d["edited_layer_index"], d["rank"], d["trainable_parameter_count"], d["merged_delta_weight_norm"], d["minimal_scale_multiplier"])' "${SUMMARY}")"
  read -r ACTIVE UNMET SAFE_FAILED LAYER RANK PARAMS NORM SCALE <<< "${AUDIT}"
  echo "V8 exact audit: active=${ACTIVE} active_buffer_unmet=${UNMET} safe_failed=${SAFE_FAILED} layer=${LAYER} rank=${RANK} params=${PARAMS} deltaW_norm=${NORM} scale=${SCALE}"
  if [[ "${ACTIVE}" != "0" || "${UNMET}" != "0" || "${SAFE_FAILED}" != "0" ]]; then
    echo "REFUSE final evaluation: V8 checkpoint failed forget-only audit." >&2
    exit 1
  fi

  echo "===== SEED ${SEED}: LOCKED FINAL EVAL V8.0 ====="
  if [[ "${SKIP_EXISTING}" != "1" || ! -f "${EVAL_OUT}" ]]; then
    run_locked_eval "${CKPT}" "${EVAL_OUT}" "${EVAL_MANIFEST}" "${SEED}"
  else
    echo "Reusing ${EVAL_OUT}"
    "${PYTHON_BIN}" -c 'import json,sys; d=json.load(open(sys.argv[1])); print("MQuAKE reused result: Eff={}, RetainEff={}, PPL={}".format(d["forget"]["Eff"], d["retain"]["Eff"], d.get("forget_PPL")))' "${EVAL_OUT}"
  fi
done

echo "SURE-MQuAKE V8.0 locked run complete: ${OUTPUT_ROOT}"
