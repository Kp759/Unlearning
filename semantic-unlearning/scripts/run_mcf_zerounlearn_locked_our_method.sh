#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_mcf_zerounlearn_locked_our_method.sh MODEL_PATH [MCF_PATH]

Purpose:
  Apply Setting 5e + protected sparse LM-head repair under a ZeroUnlearn-style
  MCF protocol where benchmark paraphrases/neighborhood prompts stay locked
  until the final frozen-checkpoint evaluation.

Defaults:
  MCF_SEEDS="1 2 3 4 5 6 7 8 9 10"
  MCF_FORGET_NUM=50
  MCF_RETAIN_NUM=1000

The runner intentionally does NOT run official MCF evaluation after Stage 1.
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
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/mcf_zerounlearn_locked_our_method}"
PROTOCOL_DIR="${MCF_PROTOCOL_DIR:-${OUTPUT_ROOT}/protocol}"
REPAIR_MCF="${PROTOCOL_DIR}/repair_visible_mcf.json"
SPLIT_MANIFEST="${PROTOCOL_DIR}/split_manifest.json"

SEEDS_TEXT="${MCF_SEEDS:-1 2 3 4 5 6 7 8 9 10}"
FORGET_NUM="${MCF_FORGET_NUM:-50}"
RETAIN_NUM="${MCF_RETAIN_NUM:-1000}"
DTYPE="${DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-single}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

# Frozen method defaults are taken from the already-registered controlled
# margin025/rank2 configuration, not tuned on the locked paraphrases.
STEPS="${MCF_STEPS:-600}"
BATCH_SIZE="${MCF_BATCH_SIZE:-1}"
RETAIN_BATCH_SIZE="${MCF_RETAIN_BATCH_SIZE:-4}"
EMB_LM_LR="${MCF_EMB_LM_LR:-0.0001}"
FORGET_WEIGHT="${MCF_FORGET_WEIGHT:-2.0}"
RETAIN_WEIGHT="${MCF_RETAIN_WEIGHT:-1.0}"
FORGET_MARGIN="${MCF_FORGET_MARGIN:-1.0}"

ACTIVE_MARGIN="${REPAIR_ACTIVE_MARGIN:-0.25}"
REPAIR_STEPS="${REPAIR_STEPS:-100}"
REPAIR_LR="${REPAIR_LR:-0.005}"
REPAIR_OPTIMIZER="${REPAIR_OPTIMIZER:-adamw}"
HINGE_WEIGHT="${HINGE_WEIGHT:-2.0}"
DELTA_L2_LAMBDA="${DELTA_L2_LAMBDA:-0.0001}"
RETAIN_KL_MU="${RETAIN_KL_MU:-0.1}"
RETAIN_CALIBRATION_NUM="${RETAIN_CALIBRATION_NUM:-200}"
RETAIN_CALIBRATION_SEED="${RETAIN_CALIBRATION_SEED:-1729}"
REPAIR_RANK="${REPAIR_RANK:-2}"
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

echo "===== BUILD LOCKED ZEROUnlearn-STYLE MCF VIEW ====="
"${PYTHON_BIN}" scripts/build_mcf_zerounlearn_locked_split.py \
  --mcf-path "${ORIGINAL_MCF}" \
  --output-dir "${PROTOCOL_DIR}" \
  --seeds "${SEEDS[@]}" \
  --forget-num "${FORGET_NUM}" \
  --retain-num "${RETAIN_NUM}"

test -f "${REPAIR_MCF}"
test -f "${SPLIT_MANIFEST}"

for SEED in "${SEEDS[@]}"; do
  SEED_ROOT="${OUTPUT_ROOT}/seed${SEED}"
  SETTING_DIR="${SEED_ROOT}/setting5e"
  SETTING_CKPT="${SETTING_DIR}/emb_lm_all_restore_post_training_true/checkpoint"
  SETTING_CONFIG="${SETTING_DIR}/config_used.json"
  REPAIR_DIR="${SEED_ROOT}/repair_locked"
  REPAIR_CKPT="${REPAIR_DIR}/checkpoint"
  FINAL_EVAL="${SEED_ROOT}/official_eval_locked.json"
  RUN_MANIFEST="${SEED_ROOT}/run_manifest.json"

  mkdir -p "${SEED_ROOT}"

  if [[ "${SKIP_EXISTING}" == "1" && -f "${FINAL_EVAL}" ]]; then
    echo "Seed ${SEED}: final locked evaluation already exists; skipping."
    continue
  fi

  echo
  echo "===== SEED ${SEED}: STAGE 1 SETTING 5e ====="
  echo "Repair-visible MCF has requested_rewrite only; official probes are locked."
  if [[ ! -d "${SETTING_CKPT}" ]]; then
    "${PYTHON_BIN}" scripts/gagd_compare.py \
      --dataset mcf \
      --model-path "${MODEL_PATH}" \
      --mcf-cache-path "${REPAIR_MCF}" \
      --mcf-sample-mode official \
      --official-sample-mode official \
      --output-dir "${SETTING_DIR}" \
      --mode emb_lm_all_restore_post_training_true \
      --forget-loss-type mcf_margin \
      --forget-margin "${FORGET_MARGIN}" \
      --mcf-answer-field target_new \
      --forget-num "${FORGET_NUM}" \
      --retain-num "${RETAIN_NUM}" \
      --seed "${SEED}" \
      --steps "${STEPS}" \
      --batch-size "${BATCH_SIZE}" \
      --retain-batch-size "${RETAIN_BATCH_SIZE}" \
      --emb-lm-lr "${EMB_LM_LR}" \
      --forget-weight "${FORGET_WEIGHT}" \
      --retain-weight "${RETAIN_WEIGHT}" \
      --sampling-strategy epoch \
      --post-training-new-true-alpha 0.75 \
      --post-training-new-retain-alpha 0.50 \
      --post-training-new-true-retain-alpha 0.25 \
      --dtype "${DTYPE}" \
      --device-map "${DEVICE_MAP}" \
      --wikidata-dir "${WIKIDATA_DIR}" \
      --save-model
  else
    echo "Seed ${SEED}: reusing existing Setting 5e checkpoint."
  fi

  test -d "${SETTING_CKPT}"
  test -f "${SETTING_CONFIG}"

  echo "===== SEED ${SEED}: STAGE 2 LOCKED-PROBE LM-HEAD REPAIR ====="
  rm -rf "${REPAIR_DIR}"
  "${PYTHON_BIN}" scripts/gagd_active_case_repair.py \
    --model-path "${SETTING_CKPT}" \
    --base-model-path "${MODEL_PATH}" \
    --reference-model-path "${SETTING_CKPT}" \
    --experiment-config-path "${SETTING_CONFIG}" \
    --output-dir "${REPAIR_DIR}" \
    --mcf-cache-path "${REPAIR_MCF}" \
    --sample-mode official \
    --seed "${SEED}" \
    --forget-num "${FORGET_NUM}" \
    --retain-num "${RETAIN_NUM}" \
    --repair-mode minimal_optimize \
    --active-margin "${ACTIVE_MARGIN}" \
    --repair-steps "${REPAIR_STEPS}" \
    --repair-lr "${REPAIR_LR}" \
    --repair-optimizer "${REPAIR_OPTIMIZER}" \
    --hinge-weight "${HINGE_WEIGHT}" \
    --delta-l2-lambda "${DELTA_L2_LAMBDA}" \
    --retain-kl-mu "${RETAIN_KL_MU}" \
    --retain-calibration-num "${RETAIN_CALIBRATION_NUM}" \
    --retain-calibration-seed "${RETAIN_CALIBRATION_SEED}" \
    --repair-rank "${REPAIR_RANK}" \
    --project-away-retain-hidden \
    --stop-when-all-satisfied \
    --dtype "${DTYPE}" \
    --device-map "${DEVICE_MAP}" \
    --margin-batch-size "${MARGIN_BATCH_SIZE}" \
    --save-model

  test -d "${REPAIR_CKPT}"

  # The generic repair script predates this locked-probe path and labels all MCF
  # repairs as paraphrase-conditioned. Correct only the generated provenance
  # metadata; no weights or metrics are changed here.
  "${PYTHON_BIN}" - "${REPAIR_DIR}" "${ORIGINAL_MCF}" "${REPAIR_MCF}" <<'PY'
import json
import pathlib
import sys

repair_dir = pathlib.Path(sys.argv[1])
original_mcf = str(pathlib.Path(sys.argv[2]).resolve())
repair_mcf = str(pathlib.Path(sys.argv[3]).resolve())

for config_path in (
    repair_dir / "config_used.json",
    repair_dir / "checkpoint" / "repair_experiment_config.json",
):
    if not config_path.exists():
        continue
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["repair_uses_official_paraphrases"] = False
    config["repair_prompt_scope"] = "requested_rewrite_only"
    config["evaluation_probes_locked_during_repair"] = True
    config["repair_dataset_path"] = repair_mcf
    config["final_evaluation_dataset_path"] = original_mcf
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

summary_path = repair_dir / "repair_summary.json"
summary = json.loads(summary_path.read_text(encoding="utf-8"))
summary["protocol_status"] = "native_data_and_metrics_with_locked_evaluation_probes"
summary["protocol_status_reason"] = (
    "Stage 2 saw only sampled requested_rewrite prompts. MCF paraphrase, "
    "neighborhood, and generation prompts were removed from the repair-visible "
    "dataset and remained locked until final evaluation."
)
summary["repair_uses_official_paraphrases"] = False
summary["repair_prompt_scope"] = "requested_rewrite_only"
summary["evaluation_probes_locked_during_repair"] = True
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
PY

  echo "===== SEED ${SEED}: FINAL FROZEN-CHECKPOINT EVALUATION ====="
  echo "This is the first model evaluation that receives original MCF paraphrases."
  EVAL_ARGS=(
    --model-dir "${REPAIR_CKPT}"
    --mcf-path "${ORIGINAL_MCF}"
    --wikidata-dir "${WIKIDATA_DIR}"
    --out "${FINAL_EVAL}"
    --unlearn-num "${FORGET_NUM}"
    --retain-num "${RETAIN_NUM}"
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
    "${REPAIR_CKPT}" "${FINAL_EVAL}" "${SEED}" <<PY
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "protocol": "mcf_zerounlearn_locked_paraphrase",
    "seed": int(sys.argv[6]),
    "split_manifest": str(pathlib.Path(sys.argv[2]).resolve()),
    "setting5e_checkpoint": str(pathlib.Path(sys.argv[3]).resolve()),
    "repair_checkpoint": str(pathlib.Path(sys.argv[4]).resolve()),
    "final_official_evaluation": str(pathlib.Path(sys.argv[5]).resolve()),
    "method_visible_forget_prompts": "requested_rewrite_only",
    "paraphrase_and_neighborhood_access_before_freeze": False,
    "final_selection_uses_heldout_gen": False,
    "hyperparameter_source": "config/best_runs/mcf/controlled_fivefold_margin025_rank2.json",
    "hyperparameters": {
        "setting5e_steps": ${STEPS},
        "emb_lm_lr": ${EMB_LM_LR},
        "forget_weight": ${FORGET_WEIGHT},
        "retain_weight": ${RETAIN_WEIGHT},
        "forget_margin": ${FORGET_MARGIN},
        "active_margin": ${ACTIVE_MARGIN},
        "repair_steps": ${REPAIR_STEPS},
        "repair_lr": ${REPAIR_LR},
        "repair_rank": ${REPAIR_RANK},
        "retain_kl_mu": ${RETAIN_KL_MU},
        "retain_calibration_num": ${RETAIN_CALIBRATION_NUM},
    },
}
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

  echo "Seed ${SEED} complete: ${FINAL_EVAL}"
done

echo
echo "Locked ZeroUnlearn-style MCF track complete."
echo "Protocol manifest: ${SPLIT_MANIFEST}"
echo "Final results: ${OUTPUT_ROOT}/seed*/official_eval_locked.json"
