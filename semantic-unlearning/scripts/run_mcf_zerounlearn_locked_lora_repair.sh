#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_mcf_zerounlearn_locked_lora_repair.sh MODEL_PATH [MCF_PATH]

Purpose:
  Fair Stage-2 architecture ablation against the registered locked SURE-LM run.
  This runner REUSES the exact existing forget-only Setting-5e Stage-1
  checkpoints and changes only Stage 2:

      previous: DeltaW_A = C @ B_fixed(SVD)
      this run: DeltaW_A = B_lora @ A_lora * (alpha / rank)

  Data access, active-margin logic, selected rows, repair objective, final
  evaluation, and seeds stay otherwise identical.

Defaults:
  MCF_SEEDS="1 2 3 4 5 6 7 8 9 10"
  STAGE1_ROOT=outputs/mcf_zerounlearn_forget_only_locked_3b
  OUTPUT_ROOT=outputs/mcf_zerounlearn_forget_only_locked_3b_lora_r1
  LORA_RANK=1
  LORA_ALPHA=LORA_RANK  (scaling = 1)
EOF
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="$1"
ORIGINAL_MCF="${2:-${MCF_PATH:-data/multi_counterfact.json}}"
WIKIDATA_DIR="${WIKIDATA_DIR:-data/wikidata}"
STAGE1_ROOT="${STAGE1_ROOT:-outputs/mcf_zerounlearn_forget_only_locked_3b}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/mcf_zerounlearn_forget_only_locked_3b_lora_r1}"
PROTOCOL_DIR="${MCF_PROTOCOL_DIR:-${OUTPUT_ROOT}/protocol}"
REPAIR_MCF="${PROTOCOL_DIR}/repair_visible_mcf.json"
SPLIT_MANIFEST="${PROTOCOL_DIR}/split_manifest.json"

SEEDS_TEXT="${MCF_SEEDS:-1 2 3 4 5 6 7 8 9 10}"
FORGET_NUM="${MCF_FORGET_NUM:-50}"
EVAL_RETAIN_NUM="${MCF_RETAIN_EVAL_NUM:-1000}"
DTYPE="${DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-single}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

ACTIVE_MARGIN="${REPAIR_ACTIVE_MARGIN:-0.25}"
REPAIR_STEPS="${REPAIR_STEPS:-100}"
REPAIR_LR="${REPAIR_LR:-0.005}"
REPAIR_OPTIMIZER="${REPAIR_OPTIMIZER:-adamw}"
HINGE_WEIGHT="${HINGE_WEIGHT:-2.0}"
DELTA_L2_LAMBDA="${DELTA_L2_LAMBDA:-0.0001}"
LORA_RANK="${LORA_RANK:-1}"
LORA_ALPHA="${LORA_ALPHA:-${LORA_RANK}}"
MARGIN_BATCH_SIZE="${MARGIN_BATCH_SIZE:-4}"
SKIP_PPL="${SKIP_PPL:-0}"

read -r -a SEEDS <<< "${SEEDS_TEXT}"
if [[ "${#SEEDS[@]}" -eq 0 ]]; then
  echo "MCF_SEEDS resolved to an empty list." >&2
  exit 2
fi

test -f "${ORIGINAL_MCF}"
test -d "${WIKIDATA_DIR}"
mkdir -p "${PROTOCOL_DIR}"

echo "===== BUILD/VERIFY LOCKED MCF VIEW ====="
"${PYTHON_BIN}" scripts/build_mcf_zerounlearn_locked_split.py \
  --mcf-path "${ORIGINAL_MCF}" \
  --output-dir "${PROTOCOL_DIR}" \
  --seeds "${SEEDS[@]}" \
  --forget-num "${FORGET_NUM}" \
  --retain-num "${EVAL_RETAIN_NUM}"

test -f "${REPAIR_MCF}"
test -f "${SPLIT_MANIFEST}"

for SEED in "${SEEDS[@]}"; do
  SOURCE_SEED_ROOT="${STAGE1_ROOT}/seed${SEED}"
  SETTING_DIR="${SOURCE_SEED_ROOT}/setting5e_forget_only"
  SETTING_CKPT="${SETTING_DIR}/emb_lm_all_restore_post_training_true/checkpoint"
  SETTING_CONFIG="${SETTING_DIR}/config_used.json"

  SEED_ROOT="${OUTPUT_ROOT}/seed${SEED}"
  REPAIR_DIR="${SEED_ROOT}/repair_forget_only_lora"
  REPAIR_CKPT="${REPAIR_DIR}/checkpoint"
  FINAL_EVAL="${SEED_ROOT}/official_eval_locked.json"
  RUN_MANIFEST="${SEED_ROOT}/run_manifest.json"
  mkdir -p "${SEED_ROOT}"

  if [[ "${SKIP_EXISTING}" == "1" && -f "${FINAL_EVAL}" ]]; then
    echo "Seed ${SEED}: final LoRA evaluation exists; skipping."
    continue
  fi

  if [[ ! -d "${SETTING_CKPT}" || ! -f "${SETTING_CONFIG}" ]]; then
    echo "ERROR: exact Stage-1 checkpoint missing for seed ${SEED}." >&2
    echo "Expected checkpoint: ${SETTING_CKPT}" >&2
    echo "Expected config:     ${SETTING_CONFIG}" >&2
    echo "Run the locked previous architecture first; this ablation intentionally does not retrain Stage 1." >&2
    exit 1
  fi

  echo
  echo "===== SEED ${SEED}: REUSE EXACT STAGE-1 CHECKPOINT ====="
  echo "${SETTING_CKPT}"

  echo "===== SEED ${SEED}: STAGE 2 — SPARSE LORA ON ACTIVE LM-HEAD ROWS ====="
  rm -rf "${REPAIR_DIR}"
  "${PYTHON_BIN}" scripts/mcf_forget_only_active_repair_lora.py \
    --model-path "${SETTING_CKPT}" \
    --base-model-path "${MODEL_PATH}" \
    --experiment-config-path "${SETTING_CONFIG}" \
    --output-dir "${REPAIR_DIR}" \
    --mcf-cache-path "${REPAIR_MCF}" \
    --sample-mode official \
    --seed "${SEED}" \
    --forget-num "${FORGET_NUM}" \
    --retain-num 0 \
    --repair-mode minimal_optimize \
    --active-margin "${ACTIVE_MARGIN}" \
    --repair-steps "${REPAIR_STEPS}" \
    --repair-lr "${REPAIR_LR}" \
    --repair-optimizer "${REPAIR_OPTIMIZER}" \
    --hinge-weight "${HINGE_WEIGHT}" \
    --delta-l2-lambda "${DELTA_L2_LAMBDA}" \
    --retain-kl-mu 0 \
    --retain-calibration-num 0 \
    --repair-rank "${LORA_RANK}" \
    --lora-alpha "${LORA_ALPHA}" \
    --no-project-away-retain-hidden \
    --stop-when-all-satisfied \
    --dtype "${DTYPE}" \
    --device-map "${DEVICE_MAP}" \
    --margin-batch-size "${MARGIN_BATCH_SIZE}" \
    --save-model

  test -d "${REPAIR_CKPT}"
  test -f "${REPAIR_DIR}/repair_summary.json"

  echo "===== SEED ${SEED}: FINAL LOCKED EVALUATION ====="
  echo "Forget paraphrases, neighborhoods, and 1000 retain records first enter here."
  EVAL_ARGS=(
    --model-dir "${REPAIR_CKPT}"
    --mcf-path "${ORIGINAL_MCF}"
    --wikidata-dir "${WIKIDATA_DIR}"
    --out "${FINAL_EVAL}"
    --unlearn-num "${FORGET_NUM}"
    --retain-num "${EVAL_RETAIN_NUM}"
    --seed "${SEED}"
    --sample-mode official
    --dtype "${DTYPE}"
    --device-map "${DEVICE_MAP}"
  )
  if [[ "${SKIP_PPL}" == "1" ]]; then
    EVAL_ARGS+=(--skip-ppl)
  fi
  "${PYTHON_BIN}" scripts/mcf_zero_unlearn_official_eval.py "${EVAL_ARGS[@]}"

  "${PYTHON_BIN}" - \
    "${RUN_MANIFEST}" "${SPLIT_MANIFEST}" "${SETTING_CKPT}" \
    "${REPAIR_CKPT}" "${FINAL_EVAL}" "${SEED}" "${REPAIR_DIR}/repair_summary.json" <<PY
import json
import pathlib
import sys

summary = json.loads(pathlib.Path(sys.argv[7]).read_text(encoding="utf-8"))
payload = {
    "schema_version": 1,
    "protocol": "zerounlearn_data_access_forget_only_locked_probes",
    "architecture_ablation": "sparse_lora_selected_rows",
    "comparison_control": "same_exact_stage1_checkpoint_as_previous_fixed_svd_repair",
    "seed": int(sys.argv[6]),
    "split_manifest": str(pathlib.Path(sys.argv[2]).resolve()),
    "setting5e_checkpoint_reused": str(pathlib.Path(sys.argv[3]).resolve()),
    "repair_checkpoint": str(pathlib.Path(sys.argv[4]).resolve()),
    "final_official_evaluation": str(pathlib.Path(sys.argv[5]).resolve()),
    "training_data_access": {
        "forget_records": ${FORGET_NUM},
        "mcf_retain_records": 0,
        "forget_prompt_types": ["requested_rewrite"],
        "paraphrases": 0,
        "neighborhood_prompts": 0
    },
    "final_evaluation_data": {
        "forget_records": ${FORGET_NUM},
        "retain_records": ${EVAL_RETAIN_NUM},
        "forget_paraphrases_enabled": True,
        "neighborhood_prompts_enabled": True
    },
    "final_selection_uses_heldout_gen": False,
    "stage2": {
        "parameterization": "sparse_lora_selected_rows",
        "lora_rank": ${LORA_RANK},
        "lora_alpha": ${LORA_ALPHA},
        "active_margin": ${ACTIVE_MARGIN},
        "repair_steps": ${REPAIR_STEPS},
        "repair_lr": ${REPAIR_LR},
        "hinge_weight": ${HINGE_WEIGHT},
        "delta_l2_lambda": ${DELTA_L2_LAMBDA},
        "selected_lm_head_rows": summary.get("selected_lm_head_rows"),
        "stage2_trainable_parameters": summary.get("stage2_trainable_parameters"),
        "lora_shapes": summary.get("lora_shapes")
    }
}
path = pathlib.Path(sys.argv[1])
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

  echo "Seed ${SEED} LoRA complete: ${FINAL_EVAL}"
done

echo
echo "Locked sparse-LoRA Stage-2 ablation complete."
echo "Stage 1 reused from: ${STAGE1_ROOT}"
echo "LoRA output root:  ${OUTPUT_ROOT}"
echo "LoRA rank/alpha:   ${LORA_RANK}/${LORA_ALPHA}"
echo "Final results:     ${OUTPUT_ROOT}/seed*/official_eval_locked.json"
