#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

FULL_TOFU_MODEL="${1:-${TOFU_FULL_MODEL_PATH:-/scratch/yl258/kp759/Unlearning/semantic-unlearning/outputs/tofu_full_utility_sweep_v7_repro_20260815/lr4e-5_epochs6/checkpoint_epoch_5}}"
SEEDS="${TOFU_SEEDS:-1}"
PROTOCOL_ROOT="${TOFU_PROTOCOL_ROOT:-outputs/tofu_author_balanced_locked_3b_test/protocol}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/tofu_sure_author_balanced_v5_r64_active_3b_test}"

FORGET_AUTHORS="${TOFU_FORGET_AUTHORS:-5}"
QAS_PER_AUTHOR="${TOFU_QAS_PER_AUTHOR:-20}"
TRAIN_QAS_PER_AUTHOR="${TOFU_TRAIN_QAS_PER_AUTHOR:-10}"
RETAIN_EVAL_NUM="${TOFU_RETAIN_EVAL_NUM:-1000}"
FORGET_NUM=$((FORGET_AUTHORS * TRAIN_QAS_PER_AUTHOR))

# Identical Stage1A to the R512 V5 run.
STAGE1A_STEPS="${STAGE1A_STEPS:-600}"
STAGE1A_LR="${STAGE1A_LR:-0.0001}"
STAGE1A_GA_WEIGHT="${STAGE1A_GA_WEIGHT:-2.0}"
STAGE1A_GD_WEIGHT="${STAGE1A_GD_WEIGHT:-1.0}"
STAGE1A_RESTORATION_MODE="${STAGE1A_RESTORATION_MODE:-sensitive_both}"

TARGET_FORGET_PROB="${TARGET_FORGET_PROB:-0.0003}"
INITIAL_ROWS_PER_EXAMPLE="${INITIAL_ROWS_PER_EXAMPLE:-3}"

# Only controlled change relative to the successful V5 run: primary hidden rank 512 -> 64.
R64_RANK=64
R64_STEPS="${R64_STEPS:-10000}"
R64_LR="${R64_LR:-0.005}"
R64_OPTIMIZER="${R64_OPTIMIZER:-adamw}"
R64_L2="${R64_L2:-0.000001}"
R64_BOUNDARY_BISECTION_STEPS="${R64_BOUNDARY_BISECTION_STEPS:-30}"
R64_BOUNDARY_SAFETY_FRACTION="${R64_BOUNDARY_SAFETY_FRACTION:-0.002}"

# Stage2 remains identical: unrestricted correction only on residual deficient pairs.
STAGE2_STEPS="${STAGE2_STEPS:-10000}"
STAGE2_LR="${STAGE2_LR:-0.005}"
STAGE2_OPTIMIZER="${STAGE2_OPTIMIZER:-adamw}"
STAGE2_L2="${STAGE2_L2:-0.000001}"
STAGE2_BOUNDARY_BISECTION_STEPS="${STAGE2_BOUNDARY_BISECTION_STEPS:-30}"
STAGE2_MATERIALIZATION_SAFETY_FRACTIONS="${STAGE2_MATERIALIZATION_SAFETY_FRACTIONS:-0.002,0.01,0.02,0.05,0.10}"

DTYPE="${DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-single}"
MAX_LENGTH="${MAX_LENGTH:-256}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
RUN_LOCKED_GENERATION="${RUN_LOCKED_GENERATION:-0}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
CLEANUP_PREVIOUS_SURE_WEIGHTS="${CLEANUP_PREVIOUS_SURE_WEIGHTS:-1}"
CLEANUP_AFTER_EVAL="${CLEANUP_AFTER_EVAL:-1}"

if [[ ! -f "${FULL_TOFU_MODEL}/model.safetensors" || ! -f "${FULL_TOFU_MODEL}/config.json" ]]; then
  echo "Missing protected Full-TOFU model: ${FULL_TOFU_MODEL}" >&2
  exit 2
fi
PROTECTED_FULL_ABS="$(realpath "${FULL_TOFU_MODEL}")"

safe_delete_sure_checkpoint() {
  local target="$1"
  [[ -d "${target}" ]] || return 0
  [[ "$(basename "${target}")" == "checkpoint" ]] || {
    echo "REFUSE delete: not a checkpoint directory: ${target}" >&2
    exit 90
  }
  local target_abs
  target_abs="$(realpath "${target}")"
  if [[ "${target_abs}" == "${PROTECTED_FULL_ABS}" || "${target_abs}" == "${PROTECTED_FULL_ABS}"/* || "${PROTECTED_FULL_ABS}" == "${target_abs}"/* ]]; then
    echo "REFUSE delete: protected Full-TOFU overlap: ${target_abs}" >&2
    exit 91
  fi
  case "${target_abs}" in
    */outputs/tofu_sure*/checkpoint) ;;
    *)
      echo "REFUSE delete outside SURE output checkpoint: ${target_abs}" >&2
      exit 92
      ;;
  esac
  echo "DELETE SURE CHECKPOINT WEIGHTS, KEEP JSON/REPORTS: ${target_abs}"
  rm -rf -- "${target_abs}"
}

if [[ "${CLEANUP_PREVIOUS_SURE_WEIGHTS}" == "1" ]]; then
  echo "===== PRE-RUN CLEANUP: DELETE ALL PREVIOUS SURE CHECKPOINT WEIGHTS ====="
  while IFS= read -r -d '' ckpt; do
    safe_delete_sure_checkpoint "${ckpt}"
  done < <(find outputs -type d -name checkpoint -path '*/tofu_sure*/*' -print0 2>/dev/null || true)
  echo "Previous SURE checkpoints removed. JSON/JSONL/log/eval artifacts preserved."
fi

MISSING_SEEDS=()
for SEED in ${SEEDS}; do
  if [[ ! -f "${PROTOCOL_ROOT}/seed${SEED}/split_manifest.json" ]]; then
    MISSING_SEEDS+=("${SEED}")
  fi
done
if (( ${#MISSING_SEEDS[@]} > 0 )); then
  python scripts/build_tofu_zerounlearn_locked_split.py \
    --output-dir "${PROTOCOL_ROOT}" \
    --seeds "${MISSING_SEEDS[@]}" \
    --forget-authors "${FORGET_AUTHORS}" \
    --qas-per-author "${QAS_PER_AUTHOR}" \
    --train-qas-per-author "${TRAIN_QAS_PER_AUTHOR}" \
    --retain-num "${RETAIN_EVAL_NUM}"
fi

for SEED in ${SEEDS}; do
  PROTOCOL_SEED="${PROTOCOL_ROOT}/seed${SEED}"
  TRAIN_FORGET="${PROTOCOL_SEED}/train_visible/forget.json"
  EVAL_DIR="${PROTOCOL_SEED}/eval_only"
  ROOT="${OUTPUT_ROOT}/seed${SEED}"
  STAGE1A="${ROOT}/stage1a_gagd"
  STAGE1B="${ROOT}/stage1b_r64_activepair_v5"
  PRIMARY_CKPT="${STAGE1B}/r512/checkpoint"  # historical internal V5 directory name
  FINAL_CKPT="${STAGE1B}/checkpoint"
  mkdir -p "${ROOT}"

  test -f "${TRAIN_FORGET}"
  test -d "${EVAL_DIR}"

  echo
  echo "===== SURE-TOFU V5-R64 SEED ${SEED}: STAGE1A SAME-PROMPT GA/GD ====="
  if [[ "${SKIP_EXISTING}" != "1" || ! -f "${STAGE1A}/checkpoint/model.safetensors" ]]; then
    rm -rf "${STAGE1A}/checkpoint"
    python scripts/tofu_sure_stage1_gagd.py \
      --model-path "${FULL_TOFU_MODEL}" \
      --forget-json "${TRAIN_FORGET}" \
      --output-dir "${STAGE1A}" \
      --seed "${SEED}" --forget-num "${FORGET_NUM}" \
      --steps "${STAGE1A_STEPS}" --batch-size 1 \
      --emb-lm-lr "${STAGE1A_LR}" \
      --ga-weight "${STAGE1A_GA_WEIGHT}" \
      --gd-weight "${STAGE1A_GD_WEIGHT}" \
      --restoration-mode "${STAGE1A_RESTORATION_MODE}" \
      --dtype "${DTYPE}" --device-map "${DEVICE_MAP}"
  else
    echo "Reusing ${STAGE1A}/checkpoint"
  fi

  echo "===== V5-R64: R64 FORGET-HIDDEN REPAIR -> RESIDUAL ACTIVE-PAIR LM REPAIR ====="
  if [[ "${SKIP_EXISTING}" != "1" || ! -f "${FINAL_CKPT}/model.safetensors" ]]; then
    rm -rf "${FINAL_CKPT}" "${PRIMARY_CKPT}"
    python scripts/tofu_sure_r64_activepair_v5.py \
      --model-path "${STAGE1A}/checkpoint" \
      --reference-model-path "${FULL_TOFU_MODEL}" \
      --forget-json "${TRAIN_FORGET}" \
      --output-dir "${STAGE1B}" \
      --seed "${SEED}" --forget-num "${FORGET_NUM}" \
      --initial-rows-per-example "${INITIAL_ROWS_PER_EXAMPLE}" \
      --target-forget-answer-probability "${TARGET_FORGET_PROB}" \
      --target-nll-buffer 0 \
      --r512-rank "${R64_RANK}" \
      --r512-steps "${R64_STEPS}" \
      --r512-lr "${R64_LR}" \
      --r512-optimizer "${R64_OPTIMIZER}" \
      --r512-l2 "${R64_L2}" \
      --r512-boundary-bisection-steps "${R64_BOUNDARY_BISECTION_STEPS}" \
      --r512-boundary-safety-fraction "${R64_BOUNDARY_SAFETY_FRACTION}" \
      --stage2-steps "${STAGE2_STEPS}" \
      --stage2-lr "${STAGE2_LR}" \
      --stage2-optimizer "${STAGE2_OPTIMIZER}" \
      --stage2-l2 "${STAGE2_L2}" \
      --stage2-boundary-bisection-steps "${STAGE2_BOUNDARY_BISECTION_STEPS}" \
      --stage2-materialization-safety-fractions "${STAGE2_MATERIALIZATION_SAFETY_FRACTIONS}" \
      --batch-size "${EVAL_BATCH_SIZE}" --max-length "${MAX_LENGTH}" \
      --dtype "${DTYPE}" --device-map "${DEVICE_MAP}"
  else
    echo "Reusing ${FINAL_CKPT}"
  fi

  test -f "${PRIMARY_CKPT}/model.safetensors"
  test -f "${FINAL_CKPT}/model.safetensors"

  echo "===== V5-R64 MODELS FROZEN; LOCKED HELD-OUT EVALUATION STARTS ====="
  PRIMARY_EVAL_ARGS=(
    --model-dir "${PRIMARY_CKPT}"
    --eval-dir "${EVAL_DIR}"
    --reference-model-dir "${FULL_TOFU_MODEL}"
    --output "${ROOT}/locked_eval_v5_r64.json"
    --seed "${SEED}"
    --dtype "${DTYPE}"
    --max-length "${MAX_LENGTH}"
  )
  FINAL_EVAL_ARGS=(
    --model-dir "${FINAL_CKPT}"
    --eval-dir "${EVAL_DIR}"
    --reference-model-dir "${FULL_TOFU_MODEL}"
    --output "${ROOT}/locked_eval_v5_r64_final.json"
    --seed "${SEED}"
    --dtype "${DTYPE}"
    --max-length "${MAX_LENGTH}"
  )
  if [[ "${RUN_LOCKED_GENERATION}" != "1" ]]; then
    PRIMARY_EVAL_ARGS+=(--skip-generation)
    FINAL_EVAL_ARGS+=(--skip-generation)
  fi
  python scripts/tofu_zerounlearn_locked_eval.py "${PRIMARY_EVAL_ARGS[@]}"
  python scripts/tofu_zerounlearn_locked_eval.py "${FINAL_EVAL_ARGS[@]}"

  python - "${ROOT}" <<'PY'
import json, pathlib, sys
root=pathlib.Path(sys.argv[1])
repair=json.loads((root/"stage1b_r64_activepair_v5"/"repair_summary.json").read_text())
r64=json.loads((root/"locked_eval_v5_r64.json").read_text())["summary"]
final=json.loads((root/"locked_eval_v5_r64_final.json").read_text())["summary"]

def brief(x):
    return {
        "seen_mean": x["seen_deletion_efficacy"]["answer_probability_mean"],
        "seen_max": x["seen_deletion_efficacy"]["answer_probability_max"],
        "paraphrase": x["prompt_generalization"]["answer_probability_mean"],
        "same_author_direct": x["same_author_fact_generalization"]["direct_answer_probability_mean"],
        "same_author_para": x["same_author_fact_generalization"]["paraphrase_answer_probability_mean"],
        "retain": x["groups"]["retain"]["answer_probability_mean"],
        "retain_ratio": x.get("retain_answer_probability_ratio_to_reference"),
    }
out={
    "primary_rank": 64,
    "r64_geometry": repair["r512"],
    "stage2_geometry": repair["stage2"],
    "r64_locked_eval": brief(r64),
    "final_locked_eval": brief(final),
}
(root/"comparison_v5_r64_vs_final.json").write_text(json.dumps(out,indent=2)+"\n")
print("===== V5 R64 VS FINAL =====")
print("R64 rank",repair["r512"]["actual_rank"],"rows",repair["r512"]["selected_row_count"],"norm",repair["r512"]["delta_norm"],"residual_qas",repair["r512"]["residual_violating_sequence_count"])
print("Stage2 active_pairs",repair["stage2"].get("active_pair_count"),"unique_rows",repair["stage2"].get("active_pair_unique_row_count"),"norm",repair["stage2"].get("stage2_delta_norm"))
for label,x in (("R64",r64),("FINAL",final)):
    print(label,"seen",x["seen_deletion_efficacy"]["answer_probability_mean"],"seen_max",x["seen_deletion_efficacy"]["answer_probability_max"],"para",x["prompt_generalization"]["answer_probability_mean"],"retain",x["groups"]["retain"]["answer_probability_mean"],"ratio",x.get("retain_answer_probability_ratio_to_reference"))
print("wrote",root/"comparison_v5_r64_vs_final.json")
PY

  if [[ "${CLEANUP_AFTER_EVAL}" == "1" ]]; then
    echo "===== V5-R64 CLEANUP: WEIGHTS ONLY, KEEP JSON ====="
    safe_delete_sure_checkpoint "${PRIMARY_CKPT}"
    safe_delete_sure_checkpoint "${FINAL_CKPT}"
    safe_delete_sure_checkpoint "${STAGE1A}/checkpoint"
    printf '%s\n' \
      "seed=${SEED}" \
      "primary_rank=64" \
      "cleanup_after_eval=1" \
      "deleted=Stage1A, V5 R64 primary, and V5 R64 final checkpoint directories only" \
      "kept=all JSON, JSONL, logs, split manifests, locked evaluations" \
      "protected_full_tofu=${PROTECTED_FULL_ABS}" \
      > "${ROOT}/checkpoint_cleanup.txt"
  fi
done

if [[ ! -f "${FULL_TOFU_MODEL}/model.safetensors" || ! -f "${FULL_TOFU_MODEL}/config.json" ]]; then
  echo "FATAL: protected Full-TOFU checkpoint missing after V5-R64 cleanup" >&2
  exit 93
fi

echo
echo "SURE-TOFU V5 R64 + active-pair repair complete."
echo "R64 and final locked evaluations retained; temporary SURE weights removed."
echo "Protected Full-TOFU remains: ${FULL_TOFU_MODEL}"
echo "Outputs/reports: ${OUTPUT_ROOT}/seed*/"
