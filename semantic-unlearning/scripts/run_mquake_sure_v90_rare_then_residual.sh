#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: bash scripts/run_mquake_sure_v90_rare_then_residual.sh MODEL_PATH [MQUAKE_PATH]" >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="$1"
MQUAKE_PATH="${2:-${MQUAKE_PATH:-data/MQuAKE-CF-3k-v2.json}}"
WIKIDATA_DIR="${WIKIDATA_DIR:-data/wikidata}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/mquake_sure_v90_rare_then_residual}"
SEEDS_TEXT="${MQUAKE_SEEDS:-1}"
FORGET_NUM="${MQUAKE_FORGET_NUM:-50}"
RETAIN_NUM="${MQUAKE_RETAIN_EVAL_NUM:-1000}"
DTYPE="${DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-single}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
SKIP_PPL="${SKIP_PPL:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

# Stage 1: rare/easy LM-head tokens.
V90_TOP_K="${V90_TOP_K:-3}"
V90_S1_STEPS="${V90_S1_STEPS:-800}"
V90_S1_LR="${V90_S1_LR:-0.0005}"
V90_S1_FORGET_WEIGHT="${V90_S1_FORGET_WEIGHT:-5}"
V90_S1_HARDEST_WEIGHT="${V90_S1_HARDEST_WEIGHT:-1}"
V90_S1_SAFE_WEIGHT="${V90_S1_SAFE_WEIGHT:-10}"
V90_S1_GD_WEIGHT="${V90_S1_GD_WEIGHT:-10}"
V90_S1_L2="${V90_S1_L2:-0.001}"
V90_S1_TARGET_MARGIN="${V90_S1_TARGET_MARGIN:-0.0}"
V90_S1_BF16_BUFFER="${V90_S1_BF16_BUFFER:-0.04}"
V90_S1_SAFE_FLOOR="${V90_S1_SAFE_FLOOR:-0.0}"
V90_S1_BISECTION_STEPS="${V90_S1_BISECTION_STEPS:-14}"

# Stage 2: contextual MLP repair only for positions still active after Stage 1.
V90_S2_STEPS="${V90_S2_STEPS:-2000}"
V90_S2_LR="${V90_S2_LR:-0.002}"
V90_S2_RANK="${V90_S2_RANK:-8}"
V90_S2_ALPHA="${V90_S2_ALPHA:-8}"
V90_S2_LAYER_INDEX="${V90_S2_LAYER_INDEX:--4}"
V90_S2_FORGET_WEIGHT="${V90_S2_FORGET_WEIGHT:-20}"
V90_S2_HARDEST_WEIGHT="${V90_S2_HARDEST_WEIGHT:-5}"
V90_S2_SAFE_WEIGHT="${V90_S2_SAFE_WEIGHT:-10}"
V90_S2_KL_WEIGHT="${V90_S2_KL_WEIGHT:-2}"
V90_S2_PROMPT_WEIGHT="${V90_S2_PROMPT_WEIGHT:-0.5}"
V90_S2_FACTOR_L2="${V90_S2_FACTOR_L2:-0.0001}"
V90_S2_ACTIVE_MARGIN="${V90_S2_ACTIVE_MARGIN:-0.01}"
V90_S2_SAFE_FLOOR="${V90_S2_SAFE_FLOOR:-0.0}"
V90_S2_AUDIT_EVERY="${V90_S2_AUDIT_EVERY:-25}"
V90_S2_BISECTION_STEPS="${V90_S2_BISECTION_STEPS:-14}"

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
    --method "SURE-MQuAKE V9 rare-token Stage1 + residual contextual Stage2"
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
  S2_DIR="${ROOT}/stage2_residual_contextual_mlp"
  S2_CKPT="${S2_DIR}/checkpoint"
  S2_SUMMARY="${S2_DIR}/repair_summary.json"
  PIPELINE_SUMMARY="${ROOT}/pipeline_summary_v90.json"
  EVAL_OUT="${ROOT}/official_eval_v90.json"
  EVAL_MANIFEST="${ROOT}/final_eval_split_manifest_v90.json"

  mkdir -p "${PROTOCOL_DIR}"

  echo "===== SEED ${SEED}: BUILD LOCKED MQuAKE SPLIT ====="
  "${PYTHON_BIN}" scripts/build_mquake_zerounlearn_locked_split.py \
    --mquake-path "${MQUAKE_PATH}" \
    --output-dir "${PROTOCOL_DIR}" \
    --seed "${SEED}" \
    --forget-num "${FORGET_NUM}" \
    --retain-num "${RETAIN_NUM}"

  echo "===== SEED ${SEED}: V9 STAGE 1 — TOP-${V90_TOP_K} RARE/EASY ACTIVE TOKENS PER INSTANCE ====="
  if [[ "${SKIP_EXISTING}" != "1" || ! -f "${S1_CKPT}/config.json" || ! -f "${S1_SUMMARY}" ]]; then
    rm -rf "${S1_CKPT}"
    "${PYTHON_BIN}" scripts/mquake_sure_rare_token_stage1_v90.py \
      --model-path "${MODEL_PATH}" \
      --repair-visible-path "${VISIBLE}" \
      --split-manifest "${MANIFEST}" \
      --output-dir "${S1_DIR}" \
      --seed "${SEED}" \
      --forget-num "${FORGET_NUM}" \
      --top-k-per-instance "${V90_TOP_K}" \
      --steps "${V90_S1_STEPS}" \
      --lr "${V90_S1_LR}" \
      --forget-hinge-weight "${V90_S1_FORGET_WEIGHT}" \
      --hardest-forget-hinge-weight "${V90_S1_HARDEST_WEIGHT}" \
      --safe-hinge-weight "${V90_S1_SAFE_WEIGHT}" \
      --gd-weight "${V90_S1_GD_WEIGHT}" \
      --delta-l2-lambda "${V90_S1_L2}" \
      --target-logit-margin "${V90_S1_TARGET_MARGIN}" \
      --bf16-buffer-margin "${V90_S1_BF16_BUFFER}" \
      --safe-margin-floor "${V90_S1_SAFE_FLOOR}" \
      --bf16-bisection-steps "${V90_S1_BISECTION_STEPS}" \
      --batch-size "${BATCH_SIZE}" \
      --dtype "${DTYPE}" \
      --device-map "${DEVICE_MAP}"
  else
    echo "Reusing ${S1_CKPT}"
  fi

  S1_INFO="$("${PYTHON_BIN}" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["selected_position_count"], d["selected_unique_lm_head_row_count"], d["base_active_sensitive_token_count"], d["residual_active_sensitive_token_count"], d["selected_lm_head_delta_norm"])' "${S1_SUMMARY}")"
  read -r S1_POSITIONS S1_ROWS S1_BASE_ACTIVE S1_RESIDUAL S1_NORM <<< "${S1_INFO}"
  echo "V9 Stage1: selected_positions=${S1_POSITIONS} unique_rows=${S1_ROWS} base_active=${S1_BASE_ACTIVE} residual_active=${S1_RESIDUAL} delta_norm=${S1_NORM}"

  FINAL_CKPT="${S1_CKPT}"
  STAGE2_RAN=0

  if [[ "${S1_RESIDUAL}" != "0" ]]; then
    STAGE2_RAN=1
    echo "===== SEED ${SEED}: V9 STAGE 2 — RESIDUAL CONTEXTUAL MLP REPAIR (${S1_RESIDUAL} ACTIVE TOKENS) ====="
    if [[ "${SKIP_EXISTING}" != "1" || ! -f "${S2_CKPT}/config.json" || ! -f "${S2_SUMMARY}" ]]; then
      rm -rf "${S2_CKPT}"
      "${PYTHON_BIN}" scripts/mquake_sure_contextual_mlp_v80.py \
        --model-path "${S1_CKPT}" \
        --repair-visible-path "${VISIBLE}" \
        --split-manifest "${MANIFEST}" \
        --output-dir "${S2_DIR}" \
        --seed "${SEED}" \
        --forget-num "${FORGET_NUM}" \
        --steps "${V90_S2_STEPS}" \
        --lr "${V90_S2_LR}" \
        --rank "${V90_S2_RANK}" \
        --lora-alpha "${V90_S2_ALPHA}" \
        --layer-index "${V90_S2_LAYER_INDEX}" \
        --forget-hinge-weight "${V90_S2_FORGET_WEIGHT}" \
        --hardest-forget-hinge-weight "${V90_S2_HARDEST_WEIGHT}" \
        --safe-hinge-weight "${V90_S2_SAFE_WEIGHT}" \
        --non-target-kl-weight "${V90_S2_KL_WEIGHT}" \
        --prompt-adapter-weight "${V90_S2_PROMPT_WEIGHT}" \
        --factor-l2-lambda "${V90_S2_FACTOR_L2}" \
        --active-target-margin "${V90_S2_ACTIVE_MARGIN}" \
        --safe-margin-floor "${V90_S2_SAFE_FLOOR}" \
        --grad-clip 1 \
        --batch-size "${BATCH_SIZE}" \
        --audit-every "${V90_S2_AUDIT_EVERY}" \
        --bisection-steps "${V90_S2_BISECTION_STEPS}" \
        --dtype "${DTYPE}" \
        --device-map "${DEVICE_MAP}"
    else
      echo "Reusing ${S2_CKPT}"
    fi

    S2_INFO="$("${PYTHON_BIN}" -c 'import json,sys; d=json.load(open(sys.argv[1])); m=d["materialized_bf16_metrics"]; print(int(m["official_active_sensitive_token_count"]), int(m["base_active_buffer_unmet_count"]), int(m["base_safe_failed_count"]), d["rank"], d["merged_delta_weight_norm"], d["minimal_scale_multiplier"])' "${S2_SUMMARY}")"
    read -r S2_ACTIVE S2_UNMET S2_SAFE_FAILED S2_RANK S2_NORM S2_SCALE <<< "${S2_INFO}"
    echo "V9 Stage2: active=${S2_ACTIVE} active_buffer_unmet=${S2_UNMET} safe_failed=${S2_SAFE_FAILED} rank=${S2_RANK} deltaW_norm=${S2_NORM} scale=${S2_SCALE}"
    if [[ "${S2_ACTIVE}" != "0" || "${S2_UNMET}" != "0" || "${S2_SAFE_FAILED}" != "0" ]]; then
      echo "REFUSE final evaluation: V9 Stage2 failed residual audit." >&2
      exit 1
    fi
    FINAL_CKPT="${S2_CKPT}"
  else
    echo "V9 Stage1 already achieved zero residual active tokens; Stage2 is identity/skipped."
  fi

  "${PYTHON_BIN}" - "${S1_SUMMARY}" "${S2_SUMMARY}" "${PIPELINE_SUMMARY}" "${FINAL_CKPT}" "${STAGE2_RAN}" <<'PY'
import json, sys
s1_path, s2_path, out_path, final_ckpt, stage2_ran = sys.argv[1:]
s1 = json.load(open(s1_path))
s2 = json.load(open(s2_path)) if stage2_ran == "1" else None
out = {
    "status": "PASS_V90_PIPELINE_FORGET_ONLY",
    "method": "SURE-MQuAKE V9 rare-token Stage1 + residual contextual Stage2",
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
        "residual_active_at_stage2_start": s2["base_active_sensitive_token_count"],
        "edited_layer_index": s2["edited_layer_index"],
        "rank": s2["rank"],
        "merged_delta_weight_norm": s2["merged_delta_weight_norm"],
        "minimal_scale_multiplier": s2["minimal_scale_multiplier"],
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

  echo "===== SEED ${SEED}: LOCKED FINAL EVAL V9 ====="
  if [[ "${SKIP_EXISTING}" != "1" || ! -f "${EVAL_OUT}" ]]; then
    run_locked_eval "${FINAL_CKPT}" "${EVAL_OUT}" "${EVAL_MANIFEST}" "${SEED}"
  else
    echo "Reusing ${EVAL_OUT}"
    "${PYTHON_BIN}" -c 'import json,sys; d=json.load(open(sys.argv[1])); print("MQuAKE reused result: Eff={}, RetainEff={}, PPL={}".format(d["forget"]["Eff"], d["retain"]["Eff"], d.get("forget_PPL")))' "${EVAL_OUT}"
  fi

done

echo "SURE-MQuAKE V9 locked run complete: ${OUTPUT_ROOT}"
