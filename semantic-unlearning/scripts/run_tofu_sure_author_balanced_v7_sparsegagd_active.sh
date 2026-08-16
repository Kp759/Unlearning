#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

FULL_TOFU_MODEL="${1:-${TOFU_FULL_MODEL_PATH:-/scratch/yl258/kp759/Unlearning/semantic-unlearning/outputs/tofu_full_utility_sweep_v7_repro_20260815/lr4e-5_epochs6/checkpoint_epoch_5}}"
SEEDS="${TOFU_SEEDS:-1}"
REPAIR_RANKS="${REPAIR_RANKS:-0 256}"
PROTOCOL_ROOT="${TOFU_PROTOCOL_ROOT:-outputs/tofu_author_balanced_locked_3b_test/protocol}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/tofu_sure_author_balanced_v7_sparsegagd_active_3b_test}"

FORGET_AUTHORS="${TOFU_FORGET_AUTHORS:-5}"
QAS_PER_AUTHOR="${TOFU_QAS_PER_AUTHOR:-20}"
TRAIN_QAS_PER_AUTHOR="${TOFU_TRAIN_QAS_PER_AUTHOR:-10}"
RETAIN_EVAL_NUM="${TOFU_RETAIN_EVAL_NUM:-1000}"
FORGET_NUM=$((FORGET_AUTHORS * TRAIN_QAS_PER_AUTHOR))

# Stage 1: sparse LM-head-only GA/GD on all answer-token rows from the 50 QAs.
STAGE1_STEPS="${STAGE1_STEPS:-600}"
STAGE1_LR="${STAGE1_LR:-0.0001}"
STAGE1_GA_WEIGHT="${STAGE1_GA_WEIGHT:-2.0}"
STAGE1_GD_WEIGHT="${STAGE1_GD_WEIGHT:-1.0}"
STAGE1_L2="${STAGE1_L2:-0.000001}"
STAGE1_GRAD_CLIP="${STAGE1_GRAD_CLIP:-1.0}"

# Stage 2: residual active cases only. Rank 0 = unrestricted; rank 256 = all-50 hidden basis.
TARGET_FORGET_PROB="${TARGET_FORGET_PROB:-0.0003}"
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

cleanup_v7_temp_weights() {
  if [[ "${CLEANUP_AFTER_EVAL}" != "1" || ! -d "${OUTPUT_ROOT}" ]]; then
    return 0
  fi
  while IFS= read -r -d '' ckpt; do
    safe_delete_sure_checkpoint "${ckpt}"
  done < <(find "${OUTPUT_ROOT}" -type d -name checkpoint -print0 2>/dev/null || true)
}
trap cleanup_v7_temp_weights EXIT

if [[ "${CLEANUP_PREVIOUS_SURE_WEIGHTS}" == "1" ]]; then
  echo "===== PRE-RUN CLEANUP: DELETE ALL PREVIOUS SURE CHECKPOINT WEIGHTS ====="
  while IFS= read -r -d '' ckpt; do
    safe_delete_sure_checkpoint "${ckpt}"
  done < <(find outputs -type d -name checkpoint -path '*/tofu_sure*/*' -print0 2>/dev/null || true)
  echo "Previous SURE checkpoint weights removed; JSON/JSONL/log/eval artifacts preserved."
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
  STAGE1="${ROOT}/stage1_sparse_lm_gagd"
  mkdir -p "${ROOT}"

  test -f "${TRAIN_FORGET}"
  test -d "${EVAL_DIR}"

  echo
  echo "===== SURE-TOFU V7 SEED ${SEED}: STAGE1 SPARSE LM-HEAD GA/GD ====="
  if [[ "${SKIP_EXISTING}" != "1" || ! -f "${STAGE1}/checkpoint/model.safetensors" ]]; then
    rm -rf "${STAGE1}/checkpoint"
    python scripts/tofu_sure_sparse_lm_gagd_v7.py \
      --model-path "${FULL_TOFU_MODEL}" \
      --forget-json "${TRAIN_FORGET}" \
      --output-dir "${STAGE1}" \
      --seed "${SEED}" --forget-num "${FORGET_NUM}" \
      --steps "${STAGE1_STEPS}" --lr "${STAGE1_LR}" \
      --ga-weight "${STAGE1_GA_WEIGHT}" --gd-weight "${STAGE1_GD_WEIGHT}" \
      --delta-l2-lambda "${STAGE1_L2}" --grad-clip "${STAGE1_GRAD_CLIP}" \
      --target-forget-answer-probability "${TARGET_FORGET_PROB}" \
      --batch-size "${EVAL_BATCH_SIZE}" --max-length "${MAX_LENGTH}" \
      --dtype "${DTYPE}" --device-map "${DEVICE_MAP}"
  else
    echo "Reusing ${STAGE1}/checkpoint"
  fi

  echo "===== V7 STAGE1 FROZEN; LOCKED EVALUATION ====="
  STAGE1_EVAL_ARGS=(
    --model-dir "${STAGE1}/checkpoint"
    --eval-dir "${EVAL_DIR}"
    --reference-model-dir "${FULL_TOFU_MODEL}"
    --output "${ROOT}/locked_eval_v7_stage1.json"
    --seed "${SEED}"
    --dtype "${DTYPE}"
    --max-length "${MAX_LENGTH}"
  )
  if [[ "${RUN_LOCKED_GENERATION}" != "1" ]]; then
    STAGE1_EVAL_ARGS+=(--skip-generation)
  fi
  python scripts/tofu_zerounlearn_locked_eval.py "${STAGE1_EVAL_ARGS[@]}"

  for RANK in ${REPAIR_RANKS}; do
    STAGE2="${ROOT}/stage2_active_r${RANK}"
    echo
    echo "===== V7 STAGE2 RANK ${RANK}: ACTIVE CASES ONLY ====="
    if [[ "${SKIP_EXISTING}" != "1" || ! -f "${STAGE2}/checkpoint/model.safetensors" ]]; then
      rm -rf "${STAGE2}/checkpoint"
      python scripts/tofu_sure_active_hidden_repair_v7.py \
        --model-path "${STAGE1}/checkpoint" \
        --reference-model-path "${FULL_TOFU_MODEL}" \
        --forget-json "${TRAIN_FORGET}" \
        --output-dir "${STAGE2}" \
        --seed "${SEED}" --forget-num "${FORGET_NUM}" \
        --target-forget-answer-probability "${TARGET_FORGET_PROB}" \
        --repair-rank "${RANK}" \
        --repair-steps "${STAGE2_STEPS}" --repair-lr "${STAGE2_LR}" \
        --repair-optimizer "${STAGE2_OPTIMIZER}" \
        --delta-l2-lambda "${STAGE2_L2}" \
        --boundary-bisection-steps "${STAGE2_BOUNDARY_BISECTION_STEPS}" \
        --materialization-safety-fractions "${STAGE2_MATERIALIZATION_SAFETY_FRACTIONS}" \
        --batch-size "${EVAL_BATCH_SIZE}" --max-length "${MAX_LENGTH}" \
        --dtype "${DTYPE}" --device-map "${DEVICE_MAP}"
    else
      echo "Reusing ${STAGE2}/checkpoint"
    fi

    echo "===== V7 RANK ${RANK} FROZEN; LOCKED EVALUATION ====="
    EVAL_ARGS=(
      --model-dir "${STAGE2}/checkpoint"
      --eval-dir "${EVAL_DIR}"
      --reference-model-dir "${FULL_TOFU_MODEL}"
      --output "${ROOT}/locked_eval_v7_r${RANK}.json"
      --seed "${SEED}"
      --dtype "${DTYPE}"
      --max-length "${MAX_LENGTH}"
    )
    if [[ "${RUN_LOCKED_GENERATION}" != "1" ]]; then
      EVAL_ARGS+=(--skip-generation)
    fi
    python scripts/tofu_zerounlearn_locked_eval.py "${EVAL_ARGS[@]}"

    # Delete each Stage2 model immediately after its locked evaluation so rank0
    # and rank256 can be compared without holding two 3B checkpoints at once.
    if [[ "${CLEANUP_AFTER_EVAL}" == "1" ]]; then
      safe_delete_sure_checkpoint "${STAGE2}/checkpoint"
    fi
  done

  python - "${ROOT}" ${REPAIR_RANKS} <<'PY'
import json, pathlib, sys
root=pathlib.Path(sys.argv[1])
ranks=[int(x) for x in sys.argv[2:]]

def load_summary(path):
    payload=json.loads(path.read_text())
    return payload.get("summary",payload)

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

stage1_geom=json.loads((root/"stage1_sparse_lm_gagd"/"repair_summary.json").read_text())
stage1_eval=load_summary(root/"locked_eval_v7_stage1.json")
out={
    "stage1_geometry": stage1_geom,
    "stage1_locked_eval": brief(stage1_eval),
    "repairs": {},
}
print("===== V7 STAGE1 / ACTIVE REPAIR COMPARISON =====")
print(
    "STAGE1 sensitive_rows",stage1_geom["sensitive_lm_head_row_count"],
    "norm",stage1_geom["selected_lm_head_delta_norm"],
    "KL",stage1_geom["same_prompt_non_target_kl"],
    "active_after",stage1_geom["materialized_metrics"]["active_forget_instance_count"],
    "seen",stage1_eval["seen_deletion_efficacy"]["answer_probability_mean"],
    "seen_max",stage1_eval["seen_deletion_efficacy"]["answer_probability_max"],
    "retain",stage1_eval["groups"]["retain"]["answer_probability_mean"],
    "ratio",stage1_eval.get("retain_answer_probability_ratio_to_reference"),
)
for rank in ranks:
    geom=json.loads((root/f"stage2_active_r{rank}"/"repair_summary.json").read_text())
    ev=load_summary(root/f"locked_eval_v7_r{rank}.json")
    out["repairs"][str(rank)]={"geometry":geom,"locked_eval":brief(ev)}
    print(
        f"R{rank}",
        "active_qas",geom["initially_active_forget_instance_count"],
        "active_rows",geom["selected_active_lm_head_row_count"],
        "actual_rank",geom["repair_rank_actual"],
        "stage2_norm",geom["incremental_stage2_delta_norm"],
        "total_base_norm",geom["total_visible_answer_lm_head_delta_norm_from_base"],
        "seen",ev["seen_deletion_efficacy"]["answer_probability_mean"],
        "seen_max",ev["seen_deletion_efficacy"]["answer_probability_max"],
        "para",ev["prompt_generalization"]["answer_probability_mean"],
        "unseen",ev["same_author_fact_generalization"]["direct_answer_probability_mean"],
        "retain",ev["groups"]["retain"]["answer_probability_mean"],
        "ratio",ev.get("retain_answer_probability_ratio_to_reference"),
    )
(root/"comparison_v7_stage1_active_repairs.json").write_text(json.dumps(out,indent=2)+"\n")
print("wrote",root/"comparison_v7_stage1_active_repairs.json")
PY

  if [[ "${CLEANUP_AFTER_EVAL}" == "1" ]]; then
    echo "===== V7 CLEANUP: STAGE1 WEIGHTS ONLY, KEEP ALL REPORTS ====="
    safe_delete_sure_checkpoint "${STAGE1}/checkpoint"
    printf '%s\n' \
      "seed=${SEED}" \
      "repair_ranks=${REPAIR_RANKS}" \
      "cleanup_after_eval=1" \
      "deleted=Stage1 and each Stage2 checkpoint directory after locked evaluation" \
      "kept=all JSON, JSONL, logs, split manifests, locked evaluations" \
      "protected_full_tofu=${PROTECTED_FULL_ABS}" \
      > "${ROOT}/checkpoint_cleanup.txt"
  fi
done

if [[ ! -f "${FULL_TOFU_MODEL}/model.safetensors" || ! -f "${FULL_TOFU_MODEL}/config.json" ]]; then
  echo "FATAL: protected Full-TOFU checkpoint missing after V7 cleanup" >&2
  exit 93
fi

echo
echo "SURE-TOFU V7 sparse GA/GD + active-case rank0/rank256 comparison complete."
echo "Temporary SURE checkpoints removed; JSON/evaluation artifacts retained."
echo "Protected Full-TOFU remains: ${FULL_TOFU_MODEL}"
echo "Outputs/reports: ${OUTPUT_ROOT}/seed*/"
