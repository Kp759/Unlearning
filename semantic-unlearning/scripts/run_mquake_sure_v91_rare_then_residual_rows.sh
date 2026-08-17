#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: bash scripts/run_mquake_sure_v91_rare_then_residual_rows.sh MODEL_PATH [MQUAKE_PATH]" >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="$1"
MQUAKE_PATH="${2:-${MQUAKE_PATH:-data/MQuAKE-CF-3k-v2.json}}"
WIKIDATA_DIR="${WIKIDATA_DIR:-data/wikidata}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/mquake_sure_v91_rare_then_residual_rows}"
SEEDS_TEXT="${MQUAKE_SEEDS:-1}"
FORGET_NUM="${MQUAKE_FORGET_NUM:-50}"
RETAIN_NUM="${MQUAKE_RETAIN_EVAL_NUM:-1000}"
DTYPE="${DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-single}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
SKIP_PPL="${SKIP_PPL:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

# Stage 1: unchanged V9 rare/easy selection.
V91_TOP_K="${V91_TOP_K:-3}"
V91_S1_STEPS="${V91_S1_STEPS:-800}"
V91_S1_LR="${V91_S1_LR:-0.0005}"
V91_S1_FORGET_WEIGHT="${V91_S1_FORGET_WEIGHT:-5}"
V91_S1_HARDEST_WEIGHT="${V91_S1_HARDEST_WEIGHT:-1}"
V91_S1_SAFE_WEIGHT="${V91_S1_SAFE_WEIGHT:-10}"
V91_S1_GD_WEIGHT="${V91_S1_GD_WEIGHT:-10}"
V91_S1_L2="${V91_S1_L2:-0.001}"
V91_S1_TARGET_MARGIN="${V91_S1_TARGET_MARGIN:-0.0}"
V91_S1_BF16_BUFFER="${V91_S1_BF16_BUFFER:-0.04}"
V91_S1_SAFE_FLOOR="${V91_S1_SAFE_FLOOR:-0.0}"
V91_S1_BISECTION_STEPS="${V91_S1_BISECTION_STEPS:-14}"

# Stage 2: deterministic residual sparse LM-head rows, starting from Stage 1.
# V7.2 recomputes activity on the supplied model path, so its Base-active rows
# are exactly the tokens that remain active after Stage 1.
V91_S2_STEPS="${V91_S2_STEPS:-1000}"
V91_S2_LR="${V91_S2_LR:-0.0005}"
V91_S2_FORGET_WEIGHT="${V91_S2_FORGET_WEIGHT:-1.0}"
V91_S2_HARDEST_WEIGHT="${V91_S2_HARDEST_WEIGHT:-0.25}"
V91_S2_GD_WEIGHT="${V91_S2_GD_WEIGHT:-10.0}"
V91_S2_L2="${V91_S2_L2:-0.001}"
V91_S2_TARGET_MARGIN="${V91_S2_TARGET_MARGIN:-0.0}"
V91_S2_BF16_BUFFER="${V91_S2_BF16_BUFFER:-0.02}"
V91_S2_BISECTION_STEPS="${V91_S2_BISECTION_STEPS:-14}"

read -r -a SEEDS <<< "${SEEDS_TEXT}"
test -f "${MODEL_PATH}/config.json"
test -f "${MQUAKE_PATH}"
if [[ "${SKIP_PPL}" != "1" ]]; then test -d "${WIKIDATA_DIR}"; fi

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
    --method "SURE-MQuAKE V9.1 rare-token Stage1 + residual sparse-row Stage2"
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
  S1_DIR="${ROOT}/stage1_rare_tokens"
  S1_CKPT="${S1_DIR}/checkpoint"
  S1_SUMMARY="${S1_DIR}/repair_summary.json"
  S2_DIR="${ROOT}/stage2_residual_sparse_rows"
  S2_CKPT="${S2_DIR}/checkpoint"
  S2_SUMMARY="${S2_DIR}/repair_summary.json"
  PIPELINE_SUMMARY="${ROOT}/pipeline_summary_v91.json"
  EVAL_OUT="${ROOT}/official_eval_v91.json"
  EVAL_MANIFEST="${ROOT}/final_eval_split_manifest_v91.json"

  mkdir -p "${PROTOCOL_DIR}"

  echo "===== SEED ${SEED}: BUILD LOCKED MQuAKE SPLIT ====="
  "${PYTHON_BIN}" scripts/build_mquake_zerounlearn_locked_split.py \
    --mquake-path "${MQUAKE_PATH}" \
    --output-dir "${PROTOCOL_DIR}" \
    --seed "${SEED}" \
    --forget-num "${FORGET_NUM}" \
    --retain-num "${RETAIN_NUM}"

  echo "===== SEED ${SEED}: V9.1 STAGE 1 — TOP-${V91_TOP_K} RARE/EASY TOKENS ====="
  if [[ "${SKIP_EXISTING}" != "1" || ! -f "${S1_CKPT}/config.json" || ! -f "${S1_SUMMARY}" ]]; then
    rm -rf "${S1_CKPT}"
    "${PYTHON_BIN}" scripts/mquake_sure_rare_token_stage1_v90.py \
      --model-path "${MODEL_PATH}" \
      --repair-visible-path "${VISIBLE}" \
      --split-manifest "${MANIFEST}" \
      --output-dir "${S1_DIR}" \
      --seed "${SEED}" \
      --forget-num "${FORGET_NUM}" \
      --top-k-per-instance "${V91_TOP_K}" \
      --steps "${V91_S1_STEPS}" \
      --lr "${V91_S1_LR}" \
      --forget-hinge-weight "${V91_S1_FORGET_WEIGHT}" \
      --hardest-forget-hinge-weight "${V91_S1_HARDEST_WEIGHT}" \
      --safe-hinge-weight "${V91_S1_SAFE_WEIGHT}" \
      --gd-weight "${V91_S1_GD_WEIGHT}" \
      --delta-l2-lambda "${V91_S1_L2}" \
      --target-logit-margin "${V91_S1_TARGET_MARGIN}" \
      --bf16-buffer-margin "${V91_S1_BF16_BUFFER}" \
      --safe-margin-floor "${V91_S1_SAFE_FLOOR}" \
      --bf16-bisection-steps "${V91_S1_BISECTION_STEPS}" \
      --batch-size "${BATCH_SIZE}" \
      --dtype "${DTYPE}" \
      --device-map "${DEVICE_MAP}"
  else
    echo "Reusing ${S1_CKPT}"
  fi

  S1_INFO="$("${PYTHON_BIN}" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["selected_position_count"], d["selected_unique_lm_head_row_count"], d["base_active_sensitive_token_count"], d["residual_active_sensitive_token_count"], d["selected_lm_head_delta_norm"])' "${S1_SUMMARY}")"
  read -r S1_POSITIONS S1_ROWS S1_BASE_ACTIVE S1_RESIDUAL S1_NORM <<< "${S1_INFO}"
  echo "V9.1 Stage1: selected_positions=${S1_POSITIONS} unique_rows=${S1_ROWS} base_active=${S1_BASE_ACTIVE} residual_active=${S1_RESIDUAL} delta_norm=${S1_NORM}"

  FINAL_CKPT="${S1_CKPT}"
  STAGE2_RAN=0

  if [[ "${S1_RESIDUAL}" != "0" ]]; then
    STAGE2_RAN=1
    echo "===== SEED ${SEED}: V9.1 STAGE 2 — RESIDUAL SPARSE LM-HEAD ROW REPAIR (${S1_RESIDUAL} ACTIVE TOKENS) ====="
    if [[ "${SKIP_EXISTING}" != "1" || ! -f "${S2_CKPT}/config.json" || ! -f "${S2_SUMMARY}" ]]; then
      rm -rf "${S2_CKPT}"
      "${PYTHON_BIN}" scripts/mquake_sure_active_rows_v72.py \
        --model-path "${S1_CKPT}" \
        --repair-visible-path "${VISIBLE}" \
        --split-manifest "${MANIFEST}" \
        --output-dir "${S2_DIR}" \
        --seed "${SEED}" \
        --forget-num "${FORGET_NUM}" \
        --steps "${V91_S2_STEPS}" \
        --lr "${V91_S2_LR}" \
        --forget-hinge-weight "${V91_S2_FORGET_WEIGHT}" \
        --hardest-forget-hinge-weight "${V91_S2_HARDEST_WEIGHT}" \
        --gd-weight "${V91_S2_GD_WEIGHT}" \
        --delta-l2-lambda "${V91_S2_L2}" \
        --target-logit-margin "${V91_S2_TARGET_MARGIN}" \
        --bf16-buffer-margin "${V91_S2_BF16_BUFFER}" \
        --grad-clip 1 \
        --bf16-bisection-steps "${V91_S2_BISECTION_STEPS}" \
        --batch-size "${BATCH_SIZE}" \
        --dtype "${DTYPE}" \
        --device-map "${DEVICE_MAP}"
    else
      echo "Reusing ${S2_CKPT}"
    fi

    S2_INFO="$("${PYTHON_BIN}" -c 'import json,sys; d=json.load(open(sys.argv[1])); m=d["materialized_bf16_metrics"]; print(int(m["official_active_sensitive_token_count"]), int(m["buffered_margin_unmet_token_count"]), d["base_active_sensitive_token_case_count"], d["selected_base_active_target_row_count"], d["selected_lm_head_delta_norm"], d["bf16_minimal_ray_alpha"])' "${S2_SUMMARY}")"
    read -r S2_ACTIVE S2_UNMET S2_START_ACTIVE S2_ROWS S2_NORM S2_ALPHA <<< "${S2_INFO}"
    echo "V9.1 Stage2: start_active=${S2_START_ACTIVE} editable_rows=${S2_ROWS} active=${S2_ACTIVE} margin_unmet=${S2_UNMET} delta_norm=${S2_NORM} alpha=${S2_ALPHA}"
    if [[ "${S2_ACTIVE}" != "0" || "${S2_UNMET}" != "0" ]]; then
      echo "REFUSE final evaluation: V9.1 residual sparse-row Stage2 failed exact audit." >&2
      exit 1
    fi
    FINAL_CKPT="${S2_CKPT}"
  else
    echo "V9.1 Stage1 already achieved zero residual active tokens; Stage2 skipped."
  fi

  "${PYTHON_BIN}" - "${S1_SUMMARY}" "${S2_SUMMARY}" "${PIPELINE_SUMMARY}" "${FINAL_CKPT}" "${STAGE2_RAN}" <<'PY'
import json, sys
s1_path, s2_path, out_path, final_ckpt, stage2_ran = sys.argv[1:]
s1 = json.load(open(s1_path))
s2 = json.load(open(s2_path)) if stage2_ran == "1" else None
out = {
    "status": "PASS_V91_PIPELINE_FORGET_ONLY",
    "method": "SURE-MQuAKE V9.1 rare-token Stage1 + residual sparse-row Stage2",
    "seed": s1["seed"],
    "stage1": {
        "selected_position_count": s1["selected_position_count"],
        "selected_unique_lm_head_row_count": s1["selected_unique_lm_head_row_count"],
        "base_active_sensitive_token_count": s1["base_active_sensitive_token_count"],
        "residual_active_sensitive_token_count": s1["residual_active_sensitive_token_count"],
        "selected_lm_head_delta_norm": s1["selected_lm_head_delta_norm"],
        "checkpoint": s1["checkpoint"],
    },
    "stage2_ran": stage2_ran == "1",
    "stage2": None if s2 is None else {
        "residual_active_at_stage2_start": s2["base_active_sensitive_token_case_count"],
        "selected_residual_lm_head_row_count": s2["selected_base_active_target_row_count"],
        "selected_lm_head_delta_norm": s2["selected_lm_head_delta_norm"],
        "bf16_minimal_ray_alpha": s2["bf16_minimal_ray_alpha"],
        "materialized_bf16_metrics": s2["materialized_bf16_metrics"],
        "checkpoint": s2["checkpoint"],
    },
    "final_checkpoint": final_ckpt,
    "checkpoint_selection_uses_retain_or_heldout": False,
}
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
    f.write("\n")
PY

  echo "===== SEED ${SEED}: LOCKED FINAL EVAL V9.1 ====="
  if [[ "${SKIP_EXISTING}" != "1" || ! -f "${EVAL_OUT}" ]]; then
    run_locked_eval "${FINAL_CKPT}" "${EVAL_OUT}" "${EVAL_MANIFEST}" "${SEED}"
  else
    echo "Reusing ${EVAL_OUT}"
    "${PYTHON_BIN}" -c 'import json,sys; d=json.load(open(sys.argv[1])); print("MQuAKE reused result: Eff={}, RetainEff={}, PPL={}".format(d["forget"]["Eff"], d["retain"]["Eff"], d.get("forget_PPL")))' "${EVAL_OUT}"
  fi
done

echo "SURE-MQuAKE V9.1 locked run complete: ${OUTPUT_ROOT}"
