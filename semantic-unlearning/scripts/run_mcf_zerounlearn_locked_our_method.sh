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
  Fair ZeroUnlearn-style MCF data-access experiment:
    * 50 sampled forget records are the only MCF records used by Stage 1/2;
    * paraphrase/neighborhood/generation probes stay locked until final eval;
    * 1000 sampled MCF retain records are evaluation-only utility records.

Defaults:
  MCF_SEEDS="1 2 3 4 5 6 7 8 9 10"
  MCF_FORGET_NUM=50
  MCF_RETAIN_EVAL_NUM=1000
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
TRAIN_RETAIN_NUM=0
EVAL_RETAIN_NUM="${MCF_RETAIN_EVAL_NUM:-1000}"
DTYPE="${DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-single}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

# Forget-side hyperparameters are frozen from the registered margin025/rank2
# configuration. Benchmark-retain terms are intentionally disabled here.
STEPS="${MCF_STEPS:-600}"
BATCH_SIZE="${MCF_BATCH_SIZE:-1}"
EMB_LM_LR="${MCF_EMB_LM_LR:-0.0001}"
FORGET_WEIGHT="${MCF_FORGET_WEIGHT:-2.0}"
FORGET_MARGIN="${MCF_FORGET_MARGIN:-1.0}"

ACTIVE_MARGIN="${REPAIR_ACTIVE_MARGIN:-0.25}"
REPAIR_STEPS="${REPAIR_STEPS:-100}"
REPAIR_LR="${REPAIR_LR:-0.005}"
REPAIR_OPTIMIZER="${REPAIR_OPTIMIZER:-adamw}"
HINGE_WEIGHT="${HINGE_WEIGHT:-2.0}"
DELTA_L2_LAMBDA="${DELTA_L2_LAMBDA:-0.0001}"
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

echo "===== BUILD ZEROUnlearn-STYLE LOCKED MCF VIEW ====="
"${PYTHON_BIN}" scripts/build_mcf_zerounlearn_locked_split.py \
  --mcf-path "${ORIGINAL_MCF}" \
  --output-dir "${PROTOCOL_DIR}" \
  --seeds "${SEEDS[@]}" \
  --forget-num "${FORGET_NUM}" \
  --retain-num "${EVAL_RETAIN_NUM}"

test -f "${REPAIR_MCF}"
test -f "${SPLIT_MANIFEST}"

for SEED in "${SEEDS[@]}"; do
  SEED_ROOT="${OUTPUT_ROOT}/seed${SEED}"
  SETTING_DIR="${SEED_ROOT}/setting5e_forget_only"
  SETTING_CKPT="${SETTING_DIR}/emb_lm_all_restore_post_training_true/checkpoint"
  SETTING_CONFIG="${SETTING_DIR}/config_used.json"
  REPAIR_DIR="${SEED_ROOT}/repair_forget_only"
  REPAIR_CKPT="${REPAIR_DIR}/checkpoint"
  FINAL_EVAL="${SEED_ROOT}/official_eval_locked.json"
  RUN_MANIFEST="${SEED_ROOT}/run_manifest.json"

  mkdir -p "${SEED_ROOT}"

  if [[ "${SKIP_EXISTING}" == "1" && -f "${FINAL_EVAL}" ]]; then
    echo "Seed ${SEED}: final evaluation already exists; skipping."
    continue
  fi

  echo
  echo "===== SEED ${SEED}: STAGE 1 — 50 FORGET RECORDS ONLY ====="
  if [[ ! -d "${SETTING_CKPT}" ]]; then
    "${PYTHON_BIN}" scripts/mcf_forget_only_setting5e.py \
      --model-path "${MODEL_PATH}" \
      --mcf-cache-path "${REPAIR_MCF}" \
      --output-dir "${SETTING_DIR}" \
      --seed "${SEED}" \
      --forget-num "${FORGET_NUM}" \
      --steps "${STEPS}" \
      --batch-size "${BATCH_SIZE}" \
      --emb-lm-lr "${EMB_LM_LR}" \
      --forget-weight "${FORGET_WEIGHT}" \
      --forget-margin "${FORGET_MARGIN}" \
      --optimizer adamw \
      --dtype "${DTYPE}" \
      --device-map "${DEVICE_MAP}" \
      --post-training-new-true-alpha 0.75 \
      --post-training-new-retain-alpha 0.50 \
      --post-training-new-true-retain-alpha 0.25
  else
    echo "Seed ${SEED}: reusing forget-only Setting 5e checkpoint."
  fi

  test -d "${SETTING_CKPT}"
  test -f "${SETTING_CONFIG}"

  echo "===== SEED ${SEED}: STAGE 2 — FORGET-ONLY LM-HEAD REPAIR ====="
  rm -rf "${REPAIR_DIR}"
  "${PYTHON_BIN}" scripts/mcf_forget_only_active_repair.py \
    --model-path "${SETTING_CKPT}" \
    --base-model-path "${MODEL_PATH}" \
    --experiment-config-path "${SETTING_CONFIG}" \
    --output-dir "${REPAIR_DIR}" \
    --mcf-cache-path "${REPAIR_MCF}" \
    --sample-mode official \
    --seed "${SEED}" \
    --forget-num "${FORGET_NUM}" \
    --retain-num "${TRAIN_RETAIN_NUM}" \
    --repair-mode minimal_optimize \
    --active-margin "${ACTIVE_MARGIN}" \
    --repair-steps "${REPAIR_STEPS}" \
    --repair-lr "${REPAIR_LR}" \
    --repair-optimizer "${REPAIR_OPTIMIZER}" \
    --hinge-weight "${HINGE_WEIGHT}" \
    --delta-l2-lambda "${DELTA_L2_LAMBDA}" \
    --retain-kl-mu 0 \
    --retain-calibration-num 0 \
    --repair-rank "${REPAIR_RANK}" \
    --no-project-away-retain-hidden \
    --stop-when-all-satisfied \
    --dtype "${DTYPE}" \
    --device-map "${DEVICE_MAP}" \
    --margin-batch-size "${MARGIN_BATCH_SIZE}" \
    --save-model

  test -d "${REPAIR_CKPT}"

  # Correct generic provenance labels. The sanitized repair-visible dataset has
  # no paraphrases, and this variant uses zero benchmark-retain records.
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
    config["protocol"] = "zerounlearn_data_access_forget_only"
    config["repair_uses_official_paraphrases"] = False
    config["repair_prompt_scope"] = "requested_rewrite_only"
    config["evaluation_probes_locked_during_repair"] = True
    config["benchmark_retain_examples_used_during_repair"] = 0
    config["retain_kl_mu"] = 0.0
    config["project_away_retain_hidden"] = False
    config["repair_dataset_path"] = repair_mcf
    config["final_evaluation_dataset_path"] = original_mcf
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

summary_path = repair_dir / "repair_summary.json"
summary = json.loads(summary_path.read_text(encoding="utf-8"))
summary["protocol_status"] = "zerounlearn_data_access_forget_only_locked_probes"
summary["protocol_status_reason"] = (
    "Stage 1 and Stage 2 used only the sampled forget requested_rewrite records. "
    "No MCF retain records or official paraphrase/neighborhood probes were used "
    "before the final frozen-checkpoint evaluation."
)
summary["repair_uses_official_paraphrases"] = False
summary["repair_prompt_scope"] = "requested_rewrite_only"
summary["evaluation_probes_locked_during_repair"] = True
summary["benchmark_retain_examples_used_during_repair"] = 0
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
PY

  echo "===== SEED ${SEED}: FINAL EVALUATION — 50 FORGET + 1000 RETAIN ====="
  echo "The 1000 retain records and forget paraphrases first enter here."
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
    "${REPAIR_CKPT}" "${FINAL_EVAL}" "${SEED}" <<PY
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = {
    "schema_version": 2,
    "protocol": "zerounlearn_data_access_forget_only_locked_probes",
    "seed": int(sys.argv[6]),
    "split_manifest": str(pathlib.Path(sys.argv[2]).resolve()),
    "setting5e_checkpoint": str(pathlib.Path(sys.argv[3]).resolve()),
    "repair_checkpoint": str(pathlib.Path(sys.argv[4]).resolve()),
    "final_official_evaluation": str(pathlib.Path(sys.argv[5]).resolve()),
    "training_data_access": {
        "forget_records": ${FORGET_NUM},
        "mcf_retain_records": 0,
        "forget_prompt_types": ["requested_rewrite"],
        "paraphrases": 0,
        "neighborhood_prompts": 0,
    },
    "final_evaluation_data": {
        "forget_records": ${FORGET_NUM},
        "retain_records": ${EVAL_RETAIN_NUM},
        "forget_paraphrases_enabled": True,
        "neighborhood_prompts_enabled": True,
    },
    "final_selection_uses_heldout_gen": False,
    "retain_kl_mu": 0.0,
    "project_away_retain_hidden": False,
    "hyperparameters": {
        "setting5e_steps": ${STEPS},
        "emb_lm_lr": ${EMB_LM_LR},
        "forget_weight": ${FORGET_WEIGHT},
        "forget_margin": ${FORGET_MARGIN},
        "active_margin": ${ACTIVE_MARGIN},
        "repair_steps": ${REPAIR_STEPS},
        "repair_lr": ${REPAIR_LR},
        "repair_rank": ${REPAIR_RANK},
    },
}
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

  echo "Seed ${SEED} complete: ${FINAL_EVAL}"
done

echo
echo "ZeroUnlearn-style forget-only MCF track complete."
echo "Training: ${FORGET_NUM} forget, 0 MCF retain."
echo "Evaluation: ${FORGET_NUM} forget + ${EVAL_RETAIN_NUM} retain."
echo "Final results: ${OUTPUT_ROOT}/seed*/official_eval_locked.json"
