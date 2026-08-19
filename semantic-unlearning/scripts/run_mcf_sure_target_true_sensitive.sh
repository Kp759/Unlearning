#!/usr/bin/env bash
set -euo pipefail

# SURE-LM MCF variant with the standard unlearning target convention:
#   sensitive      = original requested_rewrite.target_true
#   safe/reference = original requested_rewrite.target_new
#
# IMPORTANT:
# - Existing accepted SURE scripts/checkpoints are not modified.
# - We keep the existing SURE Stage-1/Stage-2 implementation and hyperparameters.
# - For TRAINING ONLY, we swap target_new <-> target_true in the locked
#   repair-visible MCF copy. Therefore the unchanged SURE objective suppresses
#   original target_true and favors original target_new.
# - Final evaluation always reopens the ORIGINAL, UNSWAPPED MCF file.
# - Eff-Pref/Gen-Pref below mean residual preference for the sensitive
#   ORIGINAL target_true; lower is better.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_mcf_sure_target_true_sensitive.sh MODEL_PATH [MCF_PATH]

Default run:
  seed 1 only
  50 forget facts
  0 benchmark-retain facts during Stage 1/2
  1000 retain facts at final evaluation

Semantic convention:
  sensitive      = original target_true
  safe/reference = original target_new

Override seed(s), if needed:
  MCF_SEEDS="1 2" bash scripts/run_mcf_sure_target_true_sensitive.sh MODEL_PATH
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
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/mcf_sure_target_true_sensitive}"
PROTOCOL_DIR="${MCF_PROTOCOL_DIR:-${OUTPUT_ROOT}/protocol}"
LOCKED_MCF="${PROTOCOL_DIR}/repair_visible_mcf.json"
SWAPPED_TRAIN_MCF="${PROTOCOL_DIR}/repair_visible_mcf_target_true_sensitive.json"
SPLIT_MANIFEST="${PROTOCOL_DIR}/split_manifest.json"

SEEDS_TEXT="${MCF_SEEDS:-1}"
FORGET_NUM="${MCF_FORGET_NUM:-50}"
TRAIN_RETAIN_NUM=0
EVAL_RETAIN_NUM="${MCF_RETAIN_EVAL_NUM:-1000}"
DTYPE="${DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-single}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"

# Keep the accepted SURE MCF hyperparameters unchanged.
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

echo "===== BUILD LOCKED MCF VIEW (same split/probe holdout as accepted SURE) ====="
"${PYTHON_BIN}" scripts/build_mcf_zerounlearn_locked_split.py \
  --mcf-path "${ORIGINAL_MCF}" \
  --output-dir "${PROTOCOL_DIR}" \
  --seeds "${SEEDS[@]}" \
  --forget-num "${FORGET_NUM}" \
  --retain-num "${EVAL_RETAIN_NUM}"

test -f "${LOCKED_MCF}"
test -f "${SPLIT_MANIFEST}"

echo "===== CREATE TRAINING-ONLY TARGET-SWAPPED MCF VIEW ====="
"${PYTHON_BIN}" - "${LOCKED_MCF}" "${SWAPPED_TRAIN_MCF}" <<'PY'
import copy
import json
import pathlib
import sys

src = pathlib.Path(sys.argv[1])
out = pathlib.Path(sys.argv[2])
data = json.loads(src.read_text(encoding="utf-8"))

if not isinstance(data, list):
    raise TypeError("Locked MCF must be a JSON list")

swapped = []
for idx, record in enumerate(data):
    rec = copy.deepcopy(record)
    rr = rec.get("requested_rewrite")
    rr_list = rr if isinstance(rr, list) else [rr]
    if not rr_list or any(not isinstance(x, dict) for x in rr_list):
        raise ValueError(f"Malformed requested_rewrite at record {idx}")
    for rewrite in rr_list:
        if "target_true" not in rewrite or "target_new" not in rewrite:
            raise ValueError(f"Missing target_true/target_new at record {idx}")
        original_true = copy.deepcopy(rewrite["target_true"])
        original_new = copy.deepcopy(rewrite["target_new"])
        # Existing SURE suppresses target_new and favors target_true.
        # Swap only in the TRAINING view so it suppresses ORIGINAL target_true.
        rewrite["target_new"] = original_true
        rewrite["target_true"] = original_new
    rec["requested_rewrite"] = rr_list if isinstance(rr, list) else rr_list[0]
    swapped.append(rec)

out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(swapped, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

# Sanity check: locked probes must remain empty.
for idx, rec in enumerate(swapped):
    for field in ("paraphrase_prompts", "neighborhood_prompts", "generation_prompts"):
        if rec.get(field):
            raise AssertionError(f"Training view leaked {field} at record {idx}")

print(f"Wrote target-swapped TRAINING view: {out}")
print("TRAINING semantics: target_new=ORIGINAL target_true (sensitive)")
print("TRAINING semantics: target_true=ORIGINAL target_new (safe/reference)")
PY

test -f "${SWAPPED_TRAIN_MCF}"

for SEED in "${SEEDS[@]}"; do
  SEED_ROOT="${OUTPUT_ROOT}/seed${SEED}"
  SETTING_DIR="${SEED_ROOT}/setting5e_forget_only_target_true_sensitive"
  SETTING_CKPT="${SETTING_DIR}/emb_lm_all_restore_post_training_true/checkpoint"
  SETTING_CONFIG="${SETTING_DIR}/config_used.json"
  REPAIR_DIR="${SEED_ROOT}/repair_forget_only_target_true_sensitive"
  REPAIR_CKPT="${REPAIR_DIR}/checkpoint"
  RAW_EVAL="${SEED_ROOT}/raw_eval_original_mcf.json"
  FINAL_REPORT="${SEED_ROOT}/target_true_sensitive_eval.json"
  RUN_MANIFEST="${SEED_ROOT}/run_manifest.json"

  mkdir -p "${SEED_ROOT}"

  if [[ "${SKIP_EXISTING}" == "1" && -f "${FINAL_REPORT}" ]]; then
    echo "Seed ${SEED}: target-true-sensitive final report already exists; skipping."
    continue
  fi

  echo
  echo "===== SEED ${SEED}: STAGE 1 — ORIGINAL target_true IS SENSITIVE ====="
  if [[ ! -d "${SETTING_CKPT}" ]]; then
    "${PYTHON_BIN}" scripts/mcf_forget_only_setting5e.py \
      --model-path "${MODEL_PATH}" \
      --mcf-cache-path "${SWAPPED_TRAIN_MCF}" \
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
    echo "Seed ${SEED}: reusing target-true-sensitive Stage-1 checkpoint."
  fi

  test -d "${SETTING_CKPT}"
  test -f "${SETTING_CONFIG}"

  echo "===== SEED ${SEED}: STAGE 2 — SAME RANK-2 SPARSE REPAIR, REVERSED SEMANTICS ====="
  rm -rf "${REPAIR_DIR}"
  "${PYTHON_BIN}" scripts/mcf_forget_only_active_repair.py \
    --model-path "${SETTING_CKPT}" \
    --base-model-path "${MODEL_PATH}" \
    --experiment-config-path "${SETTING_CONFIG}" \
    --output-dir "${REPAIR_DIR}" \
    --mcf-cache-path "${SWAPPED_TRAIN_MCF}" \
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

  echo "===== SEED ${SEED}: FINAL EVAL ON ORIGINAL UNSWAPPED MCF ====="
  EVAL_ARGS=(
    --model-dir "${REPAIR_CKPT}"
    --mcf-path "${ORIGINAL_MCF}"
    --wikidata-dir "${WIKIDATA_DIR}"
    --out "${RAW_EVAL}"
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

  echo "===== SEED ${SEED}: BUILD TARGET-TRUE-SENSITIVE PAPER METRICS ====="
  "${PYTHON_BIN}" - "${RAW_EVAL}" "${FINAL_REPORT}" <<'PY'
import json
import math
import pathlib
import statistics
import sys

raw_path = pathlib.Path(sys.argv[1])
out_path = pathlib.Path(sys.argv[2])
raw = json.loads(raw_path.read_text(encoding="utf-8"))


def per_record_sensitive_preference(metric_rows, prompt_key):
    vals = []
    prompt_count = 0
    sensitive_wins = 0
    ties = 0
    for row in metric_rows:
        xs = row.get("post", {}).get(prompt_key, [])
        if not xs:
            continue
        outcomes = []
        for x in xs:
            # ORIGINAL MCF semantics at final evaluation:
            # target_true = sensitive; target_new = safe/reference.
            sens_nll = float(x["target_true"])
            safe_nll = float(x["target_new"])
            prompt_count += 1
            if sens_nll < safe_nll:
                outcomes.append(1.0)
                sensitive_wins += 1
            elif sens_nll > safe_nll:
                outcomes.append(0.0)
            else:
                outcomes.append(0.5)
                ties += 1
        vals.append(sum(outcomes) / len(outcomes))
    mean = 100.0 * (sum(vals) / len(vals)) if vals else None
    # Match existing evaluator convention: population SD over per-record means.
    sd = 100.0 * statistics.pstdev(vals) if len(vals) > 1 else (0.0 if vals else None)
    return {
        "mean": mean,
        "population_sd": sd,
        "per_record_count": len(vals),
        "prompt_instance_count": prompt_count,
        "sensitive_preferred_prompt_instances": sensitive_wins,
        "exact_nll_ties": ties,
    }

forget_raw = raw["forget_raw"]
eff = per_record_sensitive_preference(forget_raw, "rewrite_prompts_probs")
gen = per_record_sensitive_preference(forget_raw, "paraphrase_prompts_probs")

forget = raw["forget"]
report = {
    "schema_version": 1,
    "method": "SURE-LM-target-true-sensitive",
    "dataset": "MCF",
    "seed": raw["seed"],
    "target_semantics": {
        "sensitive": "original requested_rewrite.target_true",
        "safe_reference": "original requested_rewrite.target_new",
        "training_transform": "swap target_true and target_new only in locked training view",
        "final_evaluation_dataset": "original unswapped MCF",
    },
    "metrics": {
        "Eff_Pref": eff,
        "Gen_Pref": gen,
        "Spe_margin": {
            "mean": forget["post_neighborhood_diff"][0],
            "population_sd": forget["post_neighborhood_diff"][1],
        },
        "Spe_success": {
            "mean": forget["post_neighborhood_success"][0],
            "population_sd": forget["post_neighborhood_success"][1],
        },
        "PPL": raw.get("forget_PPL"),
    },
    "directions": {
        "Eff_Pref": "lower_is_better",
        "Gen_Pref": "lower_is_better",
        "Spe_margin": "higher_is_better",
        "Spe_success": "higher_is_better",
        "PPL": "lower_or_stable_is_better",
    },
    "provenance": {
        "raw_eval": str(raw_path.resolve()),
        "unlearn_num": raw["unlearn_num"],
        "retain_num": raw["retain_num"],
        "sample_mode": raw["sample_mode"],
    },
}

out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

print("\n============================================================")
print("SURE-LM MCF — ORIGINAL target_true IS SENSITIVE")
print("============================================================")
print(f"{'Metric':<18}{'Value':>12}{'Direction':>14}")
print("-" * 44)
print(f"{'Eff-Pref':<18}{eff['mean']:>12.2f}{'↓':>14}")
print(f"{'Gen-Pref':<18}{gen['mean']:>12.2f}{'↓':>14}")
print(f"{'Spe-Margin':<18}{forget['post_neighborhood_diff'][0]:>12.2f}{'↑':>14}")
print(f"{'Spe-Success':<18}{forget['post_neighborhood_success'][0]:>12.2f}{'↑':>14}")
ppl = raw.get("forget_PPL")
print(f"{'PPL':<18}{('null' if ppl is None else f'{ppl:.4f}'):>12}{'↓':>14}")
print("============================================================")
print(
    f"Direct sensitive preference: {eff['sensitive_preferred_prompt_instances']}/"
    f"{eff['prompt_instance_count']} prompt instances"
)
print(
    f"Paraphrase sensitive preference: {gen['sensitive_preferred_prompt_instances']}/"
    f"{gen['prompt_instance_count']} prompt instances"
)
print(f"Wrote: {out_path.resolve()}")
PY

  "${PYTHON_BIN}" - \
    "${RUN_MANIFEST}" "${SPLIT_MANIFEST}" "${SWAPPED_TRAIN_MCF}" \
    "${SETTING_CKPT}" "${REPAIR_CKPT}" "${RAW_EVAL}" "${FINAL_REPORT}" "${SEED}" <<PY
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "protocol": "sure_mcf_target_true_sensitive_locked_probes",
    "seed": int(sys.argv[8]),
    "target_semantics": {
        "sensitive": "original target_true",
        "safe_reference": "original target_new",
    },
    "split_manifest": str(pathlib.Path(sys.argv[2]).resolve()),
    "training_dataset_swapped": str(pathlib.Path(sys.argv[3]).resolve()),
    "setting5e_checkpoint": str(pathlib.Path(sys.argv[4]).resolve()),
    "repair_checkpoint": str(pathlib.Path(sys.argv[5]).resolve()),
    "raw_original_mcf_eval": str(pathlib.Path(sys.argv[6]).resolve()),
    "target_true_sensitive_eval": str(pathlib.Path(sys.argv[7]).resolve()),
    "training_data_access": {
        "forget_records": ${FORGET_NUM},
        "mcf_retain_records": 0,
        "forget_prompt_types": ["requested_rewrite"],
        "paraphrases": 0,
        "neighborhood_prompts": 0
    },
    "hyperparameters": {
        "setting5e_steps": ${STEPS},
        "batch_size": ${BATCH_SIZE},
        "emb_lm_lr": ${EMB_LM_LR},
        "forget_weight": ${FORGET_WEIGHT},
        "forget_margin": ${FORGET_MARGIN},
        "active_margin": ${ACTIVE_MARGIN},
        "repair_steps": ${REPAIR_STEPS},
        "repair_lr": ${REPAIR_LR},
        "hinge_weight": ${HINGE_WEIGHT},
        "delta_l2_lambda": ${DELTA_L2_LAMBDA},
        "repair_rank": ${REPAIR_RANK}
    }
}
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

  echo "Seed ${SEED} complete: ${FINAL_REPORT}"
done

echo
echo "Target-true-sensitive SURE MCF run complete."
echo "Training target: ORIGINAL target_true is sensitive; ORIGINAL target_new is safe/reference."
echo "Final evaluation uses ORIGINAL UNSWAPPED MCF."
echo "Results: ${OUTPUT_ROOT}/seed*/target_true_sensitive_eval.json"
