#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_mcf_zerounlearn_locked_target_true_sensitive_rank2.sh MODEL_PATH [MCF_PATH]

Purpose:
  Exact semantic mirror of the registered 20260810 MCF best-run protocol:
    config/best_runs/by_model/llama_3b_instruct_model/mcf/
      zerounlearn_locked_forget_only_rank2_seeds1_10_20260810.md

  The ONLY intended semantic change is which original MCF answer is sensitive:
    * ORIGINAL target_true = sensitive / forget answer
    * ORIGINAL target_new  = non-sensitive / reference answer

  Training-visible mapping:
    * training target_new  <- ORIGINAL target_true   (sensitive)
    * training target_true <- ORIGINAL target_new    (reference)

  This mapping lets the exact historical Setting-5e + active-repair code path,
  which suppresses its training target_new, suppress ORIGINAL target_true
  without changing the old optimization implementation.

  Everything else matches the registered protocol:
    * seeds 1..10
    * 50 sampled forget records
    * 0 MCF retain records in Stage 1/2
    * requested_rewrite only during Stage 1/2
    * official paraphrases/neighborhood/generation probes locked until final eval
    * 1000 MCF retain records evaluation-only
    * Stage 1: 600 steps, batch 1, lr 1e-4, margin 1.0, weight 2.0
    * Stage 2: rank 2, margin .25, max 100 steps, lr .005,
               hinge 2.0, L2 1e-4
    * no retain KL/calibration/projection

Final evaluation always uses ORIGINAL UNSWAPPED MCF. Two views are saved:
  1) raw mcf_zero_unlearn_official_eval.py output for provenance;
  2) target-true-sensitive canonical/ZeroUnlearn-style mirrored summaries.
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
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/mcf_zerounlearn_locked_target_true_sensitive_rank2}"
PROTOCOL_DIR="${MCF_PROTOCOL_DIR:-${OUTPUT_ROOT}/protocol}"
REPAIR_MCF="${PROTOCOL_DIR}/repair_visible_mcf_target_true_sensitive.json"

SEEDS_TEXT="${MCF_SEEDS:-1 2 3 4 5 6 7 8 9 10}"
FORGET_NUM="${MCF_FORGET_NUM:-50}"
TRAIN_RETAIN_NUM=0
EVAL_RETAIN_NUM="${MCF_RETAIN_EVAL_NUM:-1000}"
DTYPE="${DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-single}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

# EXACT registered 20260810 Stage-1 hyperparameters.
STEPS="${MCF_STEPS:-600}"
BATCH_SIZE="${MCF_BATCH_SIZE:-1}"
EMB_LM_LR="${MCF_EMB_LM_LR:-0.0001}"
FORGET_WEIGHT="${MCF_FORGET_WEIGHT:-2.0}"
FORGET_MARGIN="${MCF_FORGET_MARGIN:-1.0}"

# EXACT registered 20260810 Stage-2 hyperparameters.
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

test -d "${MODEL_PATH}"
test -f "${ORIGINAL_MCF}"
test -d "${WIKIDATA_DIR}"
mkdir -p "${PROTOCOL_DIR}"

echo "===== BUILD TARGET-TRUE-SENSITIVE MIRRORED LOCKED MCF VIEW ====="
echo "ORIGINAL target_true = sensitive"
echo "ORIGINAL target_new  = non-sensitive/reference"
"${PYTHON_BIN}" scripts/build_mcf_best_run_target_true_locked_split.py \
  --mcf-path "${ORIGINAL_MCF}" \
  --output-dir "${PROTOCOL_DIR}" \
  --seeds "${SEEDS[@]}" \
  --forget-num "${FORGET_NUM}" \
  --retain-num "${EVAL_RETAIN_NUM}"

test -f "${REPAIR_MCF}"

for SEED in "${SEEDS[@]}"; do
  SEED_ROOT="${OUTPUT_ROOT}/seed${SEED}"
  SETTING_DIR="${SEED_ROOT}/setting5e_forget_only"
  SETTING_CKPT="${SETTING_DIR}/emb_lm_all_restore_post_training_true/checkpoint"
  SETTING_CONFIG="${SETTING_DIR}/config_used.json"
  REPAIR_DIR="${SEED_ROOT}/repair_forget_only"
  REPAIR_CKPT="${REPAIR_DIR}/checkpoint"
  BASE_EVAL="${SEED_ROOT}/base_original_mcf_eval.json"
  POST_EVAL="${SEED_ROOT}/post_original_mcf_eval.json"
  PAPER_EVAL="${SEED_ROOT}/target_true_sensitive_eval.json"
  RUN_MANIFEST="${SEED_ROOT}/run_manifest.json"
  SEED_MANIFEST="${PROTOCOL_DIR}/seed${SEED}_manifest.json"

  mkdir -p "${SEED_ROOT}"

  if [[ "${SKIP_EXISTING}" == "1" && -f "${PAPER_EVAL}" ]]; then
    echo "Seed ${SEED}: final target-true-sensitive evaluation already exists; skipping."
    continue
  fi

  echo
  echo "===== SEED ${SEED}: STAGE 1 — EXACT 20260810 LOGIC, MIRRORED ANSWERS ====="
  echo "training target_new  = ORIGINAL target_true (SENSITIVE)"
  echo "training target_true = ORIGINAL target_new  (REFERENCE)"
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
    echo "Seed ${SEED}: reusing mirrored forget-only Setting 5e checkpoint."
  fi

  test -d "${SETTING_CKPT}"
  test -f "${SETTING_CONFIG}"

  echo "===== SEED ${SEED}: STAGE 2 — EXACT 20260810 RANK-2 REPAIR ====="
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
  test -f "${REPAIR_DIR}/repair_summary.json"

  # Record semantic inversion explicitly without changing optimization outputs.
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
    config["protocol"] = "zerounlearn_locked_forget_only_target_true_sensitive_mirror"
    config["semantic_mirror_of"] = "zerounlearn_locked_forget_only_rank2_seeds1_10_20260810"
    config["repair_uses_official_paraphrases"] = False
    config["repair_prompt_scope"] = "requested_rewrite_only"
    config["evaluation_probes_locked_during_repair"] = True
    config["benchmark_retain_examples_used_during_repair"] = 0
    config["retain_kl_mu"] = 0.0
    config["project_away_retain_hidden"] = False
    config["original_sensitive_field"] = "target_true"
    config["original_reference_field"] = "target_new"
    config["training_target_new"] = "ORIGINAL target_true"
    config["training_target_true"] = "ORIGINAL target_new"
    config["repair_dataset_path"] = repair_mcf
    config["final_evaluation_dataset_path"] = original_mcf
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

summary_path = repair_dir / "repair_summary.json"
summary = json.loads(summary_path.read_text(encoding="utf-8"))
summary["protocol_status"] = "zerounlearn_locked_forget_only_target_true_sensitive_mirror"
summary["semantic_mirror_of"] = "zerounlearn_locked_forget_only_rank2_seeds1_10_20260810"
summary["protocol_status_reason"] = (
    "Exact registered 20260810 Stage-1/Stage-2 logic and hyperparameters, with only "
    "the repair-visible answer semantics mirrored so ORIGINAL target_true is sensitive. "
    "No MCF retain records or official paraphrase/neighborhood/generation probes were "
    "used before final frozen-checkpoint evaluation."
)
summary["repair_uses_official_paraphrases"] = False
summary["repair_prompt_scope"] = "requested_rewrite_only"
summary["evaluation_probes_locked_during_repair"] = True
summary["benchmark_retain_examples_used_during_repair"] = 0
summary["original_sensitive_field"] = "target_true"
summary["original_reference_field"] = "target_new"
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
PY

  echo "===== SEED ${SEED}: FINAL EVALUATION ON ORIGINAL UNSWAPPED MCF ====="
  echo "The 1000 retain records and held-out forget probes first enter here."

  EVAL_COMMON=(
    --mcf-path "${ORIGINAL_MCF}"
    --wikidata-dir "${WIKIDATA_DIR}"
    --unlearn-num "${FORGET_NUM}"
    --retain-num "${EVAL_RETAIN_NUM}"
    --seed "${SEED}"
    --sample-mode official
    --dtype "${DTYPE}"
    --device-map "${DEVICE_MAP}"
  )
  if [[ "${SKIP_PPL}" == "1" ]]; then
    EVAL_COMMON+=(--skip-ppl)
  fi

  # Base is required for paired sensitive-NLL deltas and canonical audit.
  if [[ ! -f "${BASE_EVAL}" ]]; then
    "${PYTHON_BIN}" scripts/mcf_zero_unlearn_official_eval.py \
      --model-dir "${MODEL_PATH}" \
      --out "${BASE_EVAL}" \
      "${EVAL_COMMON[@]}"
  fi

  "${PYTHON_BIN}" scripts/mcf_zero_unlearn_official_eval.py \
    --model-dir "${REPAIR_CKPT}" \
    --out "${POST_EVAL}" \
    "${EVAL_COMMON[@]}"

  # Canonical target-true-sensitive interpretation on ORIGINAL field names.
  "${PYTHON_BIN}" scripts/evaluate_mcf_target_true_sensitive.py \
    --base-eval-json "${BASE_EVAL}" \
    --post-eval-json "${POST_EVAL}" \
    --split-manifest "${SEED_MANIFEST}" \
    --out "${PAPER_EVAL}"

  "${PYTHON_BIN}" - \
    "${RUN_MANIFEST}" "${SEED_MANIFEST}" "${SETTING_CKPT}" \
    "${REPAIR_CKPT}" "${BASE_EVAL}" "${POST_EVAL}" "${PAPER_EVAL}" "${SEED}" <<PY
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "protocol": "zerounlearn_locked_forget_only_target_true_sensitive_mirror",
    "semantic_mirror_of": "zerounlearn_locked_forget_only_rank2_seeds1_10_20260810",
    "seed": int(sys.argv[8]),
    "split_manifest": str(pathlib.Path(sys.argv[2]).resolve()),
    "setting5e_checkpoint": str(pathlib.Path(sys.argv[3]).resolve()),
    "repair_checkpoint": str(pathlib.Path(sys.argv[4]).resolve()),
    "base_original_mcf_eval": str(pathlib.Path(sys.argv[5]).resolve()),
    "post_original_mcf_eval": str(pathlib.Path(sys.argv[6]).resolve()),
    "target_true_sensitive_eval": str(pathlib.Path(sys.argv[7]).resolve()),
    "target_semantics": {
        "original_sensitive_field": "target_true",
        "original_reference_field": "target_new",
        "training_target_new": "ORIGINAL target_true",
        "training_target_true": "ORIGINAL target_new",
    },
    "training_data_access": {
        "forget_records": ${FORGET_NUM},
        "mcf_retain_records": 0,
        "forget_prompt_types": ["requested_rewrite"],
        "paraphrases": 0,
        "neighborhood_prompts": 0,
        "generation_prompts": 0,
    },
    "final_evaluation_data": {
        "forget_records": ${FORGET_NUM},
        "retain_records": ${EVAL_RETAIN_NUM},
        "forget_paraphrases_enabled": True,
        "neighborhood_prompts_enabled": True,
        "original_unswapped_mcf": True,
    },
    "final_selection_uses_heldout_gen": False,
    "retain_kl_mu": 0.0,
    "project_away_retain_hidden": False,
    "hyperparameters": {
        "setting5e_steps": ${STEPS},
        "setting5e_batch_size": ${BATCH_SIZE},
        "emb_lm_lr": ${EMB_LM_LR},
        "forget_weight": ${FORGET_WEIGHT},
        "forget_margin": ${FORGET_MARGIN},
        "active_margin": ${ACTIVE_MARGIN},
        "repair_steps": ${REPAIR_STEPS},
        "repair_lr": ${REPAIR_LR},
        "hinge_weight": ${HINGE_WEIGHT},
        "delta_l2_lambda": ${DELTA_L2_LAMBDA},
        "repair_rank": ${REPAIR_RANK},
    },
}
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

  echo "Seed ${SEED} complete: ${PAPER_EVAL}"
done

# Canonical higher-is-better ROME view.
"${PYTHON_BIN}" scripts/aggregate_mcf_target_true_sensitive.py \
  --root "${OUTPUT_ROOT}" \
  --seeds "${SEEDS[@]}" \
  --out-prefix target_true_sensitive_canonical_aggregate

# ZeroUnlearn-style lower-is-better semantic mirror.
"${PYTHON_BIN}" scripts/aggregate_mcf_target_true_sensitive_zerounlearn_style.py \
  --root "${OUTPUT_ROOT}" \
  --seeds "${SEEDS[@]}" \
  --out-prefix zerounlearn_style_target_true_sensitive_aggregate

echo
echo "Target-true-sensitive exact mirrored MCF track complete."
echo "Semantic mirror of: zerounlearn_locked_forget_only_rank2_seeds1_10_20260810"
echo "Sensitive: ORIGINAL target_true"
echo "Reference: ORIGINAL target_new"
echo "Training: ${FORGET_NUM} forget, 0 MCF retain, requested_rewrite only."
echo "Stage 2: rank=${REPAIR_RANK}, margin=${ACTIVE_MARGIN}, steps=${REPAIR_STEPS}."
echo "Evaluation: ${FORGET_NUM} forget + ${EVAL_RETAIN_NUM} retain on ORIGINAL MCF."
echo "Output root: ${OUTPUT_ROOT}"
