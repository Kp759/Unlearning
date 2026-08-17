#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: bash scripts/run_mquake_sure_v74_prompt_protected.sh MODEL_PATH [MQUAKE_PATH]" >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="$1"
MQUAKE_PATH="${2:-${MQUAKE_PATH:-data/MQuAKE-CF-3k-v2.json}}"
WIKIDATA_DIR="${WIKIDATA_DIR:-data/wikidata}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/mquake_sure_v74_prompt_protected}"
SEEDS_TEXT="${MQUAKE_SEEDS:-1}"
FORGET_NUM="${MQUAKE_FORGET_NUM:-50}"
RETAIN_NUM="${MQUAKE_RETAIN_EVAL_NUM:-1000}"
DTYPE="${DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-single}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
SKIP_PPL="${SKIP_PPL:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

V74_STEPS="${V74_STEPS:-2000}"
V74_LR="${V74_LR:-0.01}"
V74_HINGE_WEIGHT="${V74_HINGE_WEIGHT:-10.0}"
V74_HARDEST_WEIGHT="${V74_HARDEST_WEIGHT:-2.5}"
V74_SAFE_WEIGHT="${V74_SAFE_WEIGHT:-10.0}"
V74_GD_WEIGHT="${V74_GD_WEIGHT:-10.0}"
V74_PROMPT_DRIFT_WEIGHT="${V74_PROMPT_DRIFT_WEIGHT:-1.0}"
V74_L2="${V74_L2:-0.001}"
V74_COEFF_L2="${V74_COEFF_L2:-0.0001}"
V74_TARGET_MARGIN="${V74_TARGET_MARGIN:-0.0}"
V74_BF16_BUFFER="${V74_BF16_BUFFER:-0.05}"
V74_SAFE_MARGIN_FLOOR="${V74_SAFE_MARGIN_FLOOR:-0.0001}"
V74_PROMPT_PROTECT_RANK="${V74_PROMPT_PROTECT_RANK:-1024}"
V74_BASIS_RANK_TOL="${V74_BASIS_RANK_TOL:-0.000001}"
V74_GRAD_CLIP="${V74_GRAD_CLIP:-1.0}"
V74_BISECTION_STEPS="${V74_BISECTION_STEPS:-14}"

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
    --method "SURE-MQuAKE V7.4 prompt-token protected minimal BF16 repair"
    --unlearn-num "${FORGET_NUM}"
    --retain-num "${RETAIN_NUM}"
    --seed "${seed}"
    --batch-size "${EVAL_BATCH_SIZE}"
    --dtype "${DTYPE}"
    --device-map "${DEVICE_MAP}"
    --skip-atomic-gen
  )
  if [[ "${SKIP_PPL}" == "1" ]]; then
    args+=(--skip-ppl)
  fi
  "${PYTHON_BIN}" scripts/mquake_zero_unlearn_official_eval.py "${args[@]}"
}

for SEED in "${SEEDS[@]}"; do
  ROOT="${OUTPUT_ROOT}/seed${SEED}"
  PROTOCOL_DIR="${ROOT}/protocol"
  VISIBLE="${PROTOCOL_DIR}/repair_visible_forget.json"
  MANIFEST="${PROTOCOL_DIR}/split_manifest.json"
  REPAIR_DIR="${ROOT}/v74_prompt_protected"
  CKPT="${REPAIR_DIR}/checkpoint"
  SUMMARY="${REPAIR_DIR}/repair_summary.json"
  EVAL_OUT="${ROOT}/official_eval_v74.json"
  EVAL_MANIFEST="${ROOT}/final_eval_split_manifest_v74.json"

  mkdir -p "${PROTOCOL_DIR}"

  echo "===== SEED ${SEED}: BUILD LOCKED MQuAKE SPLIT ====="
  "${PYTHON_BIN}" scripts/build_mquake_zerounlearn_locked_split.py \
    --mquake-path "${MQUAKE_PATH}" \
    --output-dir "${PROTOCOL_DIR}" \
    --seed "${SEED}" \
    --forget-num "${FORGET_NUM}" \
    --retain-num "${RETAIN_NUM}"

  echo "===== SEED ${SEED}: V7.4 PROMPT-TOKEN-PROTECTED REPAIR ====="
  if [[ "${SKIP_EXISTING}" != "1" || ! -f "${CKPT}/config.json" || ! -f "${SUMMARY}" ]]; then
    rm -rf "${CKPT}"
    "${PYTHON_BIN}" scripts/mquake_sure_prompt_protected_v74.py \
      --model-path "${MODEL_PATH}" \
      --repair-visible-path "${VISIBLE}" \
      --split-manifest "${MANIFEST}" \
      --output-dir "${REPAIR_DIR}" \
      --seed "${SEED}" \
      --forget-num "${FORGET_NUM}" \
      --steps "${V74_STEPS}" \
      --lr "${V74_LR}" \
      --forget-hinge-weight "${V74_HINGE_WEIGHT}" \
      --hardest-forget-hinge-weight "${V74_HARDEST_WEIGHT}" \
      --safe-hinge-weight "${V74_SAFE_WEIGHT}" \
      --gd-weight "${V74_GD_WEIGHT}" \
      --prompt-drift-weight "${V74_PROMPT_DRIFT_WEIGHT}" \
      --delta-l2-lambda "${V74_L2}" \
      --coeff-l2-lambda "${V74_COEFF_L2}" \
      --target-logit-margin "${V74_TARGET_MARGIN}" \
      --bf16-buffer-margin "${V74_BF16_BUFFER}" \
      --safe-margin-floor "${V74_SAFE_MARGIN_FLOOR}" \
      --prompt-protect-rank "${V74_PROMPT_PROTECT_RANK}" \
      --basis-rank-tol "${V74_BASIS_RANK_TOL}" \
      --grad-clip "${V74_GRAD_CLIP}" \
      --bf16-bisection-steps "${V74_BISECTION_STEPS}" \
      --batch-size "${BATCH_SIZE}" \
      --dtype "${DTYPE}" \
      --device-map "${DEVICE_MAP}"
  else
    echo "Reusing ${CKPT}"
  fi

  AUDIT="$("${PYTHON_BIN}" -c 'import json,sys; d=json.load(open(sys.argv[1])); m=d["materialized_bf16_metrics"]; p=d["prompt_protection"]; b=d["basis_diagnostics"]; c=d["final_prompt_context_drift"]; print(int(m["official_active_sensitive_token_count"]), int(m["buffered_margin_unmet_token_count"]), d["selected_base_active_target_row_count"], b["total_trainable_coefficients"], p["prompt_protection_hidden_count"], p["actual_prompt_protect_rank"], d["selected_lm_head_delta_norm"], d["bf16_minimal_ray_alpha"], c["prompt_context_abs_logit_drift_max"])' "${SUMMARY}")"
  read -r ACTIVE UNMET EDITABLE_ROWS COEFFS PROTECT_HIDDEN PROTECT_RANK DELTA_NORM ALPHA PROMPT_DRIFT <<< "${AUDIT}"
  echo "V7.4 BF16 audit: active=${ACTIVE} margin_unmet=${UNMET} editable_rows=${EDITABLE_ROWS} coeffs=${COEFFS} protect_hidden=${PROTECT_HIDDEN} protect_rank=${PROTECT_RANK} delta_norm=${DELTA_NORM} alpha=${ALPHA} prompt_drift_max=${PROMPT_DRIFT}"
  if [[ "${ACTIVE}" != "0" || "${UNMET}" != "0" ]]; then
    echo "REFUSE final evaluation: V7.4 checkpoint failed forget-only BF16 audit." >&2
    exit 1
  fi

  echo "===== SEED ${SEED}: LOCKED FINAL EVAL V7.4 ====="
  if [[ "${SKIP_EXISTING}" != "1" || ! -f "${EVAL_OUT}" ]]; then
    run_locked_eval "${CKPT}" "${EVAL_OUT}" "${EVAL_MANIFEST}" "${SEED}"
  else
    echo "Reusing ${EVAL_OUT}"
    "${PYTHON_BIN}" -c 'import json,sys; d=json.load(open(sys.argv[1])); print("MQuAKE reused result: Eff={}, RetainEff={}, PPL={}".format(d["forget"]["Eff"], d["retain"]["Eff"], d.get("forget_PPL")))' "${EVAL_OUT}"
  fi
done

echo "SURE-MQuAKE V7.4 locked run complete: ${OUTPUT_ROOT}"
