#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_mquake_zerounlearn_locked_our_method.sh MODEL_PATH [MQUAKE_PATH]

Locked ZeroUnlearn-style MQuAKE SURE protocol:
  Stage 1: atomic rewrites from 50 sampled forget instances; 0 benchmark retain.
  Stage 2: same direct rewrites only; no atomic/multi-hop questions or retain.
  Final eval: original MQuAKE file, same 50 forget + 1000 sampled retain.
              Official comparison output is Eff + PPL; optional AtomicGen is
              evaluated only after checkpoint selection.
EOF
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL_PATH="$1"
ORIGINAL_MQUAKE="${2:-${MQUAKE_PATH:-data/MQuAKE-CF-3k-v2.json}}"
WIKIDATA_DIR="${WIKIDATA_DIR:-data/wikidata}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/mquake_zerounlearn_forget_only_locked_3b}"
SEEDS_TEXT="${MQUAKE_SEEDS:-1 2 3 4 5 6 7 8 9 10}"
FORGET_NUM="${MQUAKE_FORGET_NUM:-50}"
EVAL_RETAIN_NUM="${MQUAKE_RETAIN_EVAL_NUM:-1000}"
DTYPE="${DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-single}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
SKIP_PPL="${SKIP_PPL:-0}"
RUN_ATOMIC_GEN_EXTENSION="${RUN_ATOMIC_GEN_EXTENSION:-0}"

# Stage 1: same SURE Setting-5e forget hyperparameters as the locked ZsRE track.
STEPS="${MQUAKE_STEPS:-600}"
BATCH_SIZE="${MQUAKE_BATCH_SIZE:-1}"
EMB_LM_LR="${MQUAKE_EMB_LM_LR:-0.0001}"
FORGET_WEIGHT="${MQUAKE_FORGET_WEIGHT:-2.0}"
FORGET_MARGIN="${MQUAKE_FORGET_MARGIN:-1.0}"

# Stage 2: direct-rewrite Unknown-row repair only.
REPAIR_STEPS="${REPAIR_STEPS:-800}"
REPAIR_LR="${REPAIR_LR:-0.005}"
REPAIR_OPTIMIZER="${REPAIR_OPTIMIZER:-adamw}"
ACTIVE_LOGIT_MARGIN="${ACTIVE_LOGIT_MARGIN:-0.25}"
SELECTION_LOGIT_MARGIN="${SELECTION_LOGIT_MARGIN:-0.05}"
REPAIR_RANK="${REPAIR_RANK:-0}"
REPAIR_L2_LAMBDA="${REPAIR_L2_LAMBDA:-0.000001}"
CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
CANDIDATE_SCALES="${CANDIDATE_SCALES:-1.0,0.875,0.75,0.625,0.5,0.375,0.25,0.1875,0.125,0.09375,0.0625,0.046875,0.03125,0.015625,0.0078125,0.0}"

read -r -a SEEDS <<< "${SEEDS_TEXT}"
if [[ "${#SEEDS[@]}" -eq 0 ]]; then
  echo "MQUAKE_SEEDS resolved to an empty list" >&2
  exit 2
fi

if [[ "${SKIP_PPL}" != "1" ]]; then
  test -d "${WIKIDATA_DIR}"
fi

for SEED in "${SEEDS[@]}"; do
  SEED_ROOT="${OUTPUT_ROOT}/seed${SEED}"
  PROTOCOL_DIR="${MQUAKE_PROTOCOL_DIR:-${SEED_ROOT}/protocol}"
  REPAIR_VISIBLE="${PROTOCOL_DIR}/repair_visible_forget.json"
  SPLIT_MANIFEST="${PROTOCOL_DIR}/split_manifest.json"

  SETTING_DIR="${SEED_ROOT}/setting5e_forget_only"
  SETTING_CKPT="${SETTING_DIR}/emb_lm_all_restore_post_training_true/checkpoint"
  SETTING_CONFIG="${SETTING_DIR}/config_used.json"

  REPAIR_DIR="${SEED_ROOT}/repair_forget_only"
  REPAIR_CKPT="${REPAIR_DIR}/checkpoint"
  FINAL_EVAL="${SEED_ROOT}/official_eval_locked.json"
  ATOMIC_GEN_EVAL="${SEED_ROOT}/atomic_gen_postselection.json"
  RUN_MANIFEST="${SEED_ROOT}/run_manifest.json"

  mkdir -p "${SEED_ROOT}" "${PROTOCOL_DIR}"

  if [[ "${SKIP_EXISTING}" == "1" && -f "${FINAL_EVAL}" ]]; then
    echo "Seed ${SEED}: final locked MQuAKE evaluation exists; skipping."
    continue
  fi

  echo
  echo "===== SEED ${SEED}: BUILD EXACT ZEROUnlearn MQuAKE SPLIT ====="
  "${PYTHON_BIN}" scripts/build_mquake_zerounlearn_locked_split.py \
    --mquake-path "${ORIGINAL_MQUAKE}" \
    --output-dir "${PROTOCOL_DIR}" \
    --seed "${SEED}" \
    --forget-num "${FORGET_NUM}" \
    --retain-num "${EVAL_RETAIN_NUM}"

  test -f "${REPAIR_VISIBLE}"
  test -f "${SPLIT_MANIFEST}"

  echo "===== SEED ${SEED}: STAGE 1 — 50 FORGET INSTANCES / 0 RETAIN ====="
  if [[ ! -d "${SETTING_CKPT}" ]]; then
    "${PYTHON_BIN}" scripts/mquake_forget_only_setting5e.py \
      --model-path "${MODEL_PATH}" \
      --repair-visible-path "${REPAIR_VISIBLE}" \
      --split-manifest "${SPLIT_MANIFEST}" \
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
    echo "Seed ${SEED}: reusing locked forget-only MQuAKE Setting-5e checkpoint."
  fi
  test -d "${SETTING_CKPT}"
  test -f "${SETTING_CONFIG}"

  echo "===== SEED ${SEED}: STAGE 2 — DIRECT-REWRITE UNKNOWN-ROW REPAIR ====="
  rm -rf "${REPAIR_DIR}"
  "${PYTHON_BIN}" scripts/mquake_forget_only_active_repair.py \
    --model-path "${SETTING_CKPT}" \
    --base-model-path "${MODEL_PATH}" \
    --repair-visible-path "${REPAIR_VISIBLE}" \
    --split-manifest "${SPLIT_MANIFEST}" \
    --output-dir "${REPAIR_DIR}" \
    --seed "${SEED}" \
    --forget-num "${FORGET_NUM}" \
    --repair-steps "${REPAIR_STEPS}" \
    --repair-lr "${REPAIR_LR}" \
    --repair-optimizer "${REPAIR_OPTIMIZER}" \
    --active-logit-margin "${ACTIVE_LOGIT_MARGIN}" \
    --selection-logit-margin "${SELECTION_LOGIT_MARGIN}" \
    --repair-rank "${REPAIR_RANK}" \
    --repair-l2-lambda "${REPAIR_L2_LAMBDA}" \
    --candidate-scales "${CANDIDATE_SCALES}" \
    --cache-batch-size "${CACHE_BATCH_SIZE}" \
    --dtype "${DTYPE}" \
    --device-map "${DEVICE_MAP}"

  test -d "${REPAIR_CKPT}"

  echo "===== SEED ${SEED}: FINAL ZEROUnlearn-COMPATIBLE EVAL ====="
  echo "The source file is reopened here; 1000 retain instances first enter here."
  EVAL_ARGS=(
    --model-dir "${REPAIR_CKPT}"
    --mquake-path "${ORIGINAL_MQUAKE}"
    --wikidata-dir "${WIKIDATA_DIR}"
    --out "${FINAL_EVAL}"
    --split-manifest "${SEED_ROOT}/final_eval_split_manifest.json"
    --method "SURE-LM locked MQuAKE"
    --unlearn-num "${FORGET_NUM}"
    --retain-num "${EVAL_RETAIN_NUM}"
    --seed "${SEED}"
    --batch-size "${EVAL_BATCH_SIZE}"
    --dtype "${DTYPE}"
    --device-map "${DEVICE_MAP}"
    --skip-atomic-gen
  )
  if [[ "${SKIP_PPL}" == "1" ]]; then
    EVAL_ARGS+=(--skip-ppl)
  fi
  "${PYTHON_BIN}" scripts/mquake_zero_unlearn_official_eval.py "${EVAL_ARGS[@]}"

  if [[ "${RUN_ATOMIC_GEN_EXTENSION}" == "1" ]]; then
    echo "===== SEED ${SEED}: OPTIONAL POST-SELECTION ATOMICGEN EXTENSION ====="
    EXT_ARGS=(
      --model-dir "${REPAIR_CKPT}"
      --mquake-path "${ORIGINAL_MQUAKE}"
      --wikidata-dir "${WIKIDATA_DIR}"
      --out "${ATOMIC_GEN_EVAL}"
      --method "SURE-LM locked MQuAKE post-selection AtomicGen"
      --unlearn-num "${FORGET_NUM}"
      --retain-num "${EVAL_RETAIN_NUM}"
      --seed "${SEED}"
      --batch-size "${EVAL_BATCH_SIZE}"
      --dtype "${DTYPE}"
      --device-map "${DEVICE_MAP}"
      --skip-ppl
    )
    "${PYTHON_BIN}" scripts/mquake_zero_unlearn_official_eval.py "${EXT_ARGS[@]}"
  fi

  "${PYTHON_BIN}" - \
    "${RUN_MANIFEST}" "${SPLIT_MANIFEST}" "${SETTING_CKPT}" \
    "${REPAIR_CKPT}" "${FINAL_EVAL}" "${SEED}" <<PY
import json
import pathlib
import sys

split = json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
path = pathlib.Path(sys.argv[1])
payload = {
    "schema_version": 1,
    "protocol": "mquake_zerounlearn_forget_only_locked_probes",
    "seed": int(sys.argv[6]),
    "split_manifest": str(pathlib.Path(sys.argv[2]).resolve()),
    "setting5e_checkpoint": str(pathlib.Path(sys.argv[3]).resolve()),
    "repair_checkpoint": str(pathlib.Path(sys.argv[4]).resolve()),
    "final_official_evaluation": str(pathlib.Path(sys.argv[5]).resolve()),
    "training_data_access": {
        "forget_instances": ${FORGET_NUM},
        "forget_atomic_facts": split["sampling"]["forget_atomic_fact_count"],
        "benchmark_retain_instances": 0,
        "prompt_types": ["requested_rewrite"],
        "atomic_questions": 0,
        "multihop_questions": 0,
        "benchmark_counterfactual_targets": 0,
    },
    "repair_data_access": {
        "forget_instances": ${FORGET_NUM},
        "forget_atomic_facts": split["sampling"]["forget_atomic_fact_count"],
        "benchmark_retain_instances": 0,
        "prompt_types": ["requested_rewrite"],
        "atomic_questions": 0,
        "multihop_questions": 0,
    },
    "final_evaluation_data": {
        "forget_instances": ${FORGET_NUM},
        "retain_instances": ${EVAL_RETAIN_NUM},
        "same_forget_instances_as_deletion_requests": True,
        "zero_unlearn_native_metrics": ["Eff", "PPL"],
    },
    "checkpoint_selection_uses_atomic_or_multihop_questions": False,
    "checkpoint_selection_uses_benchmark_retain": False,
    "hyperparameters": {
        "setting5e_steps": ${STEPS},
        "emb_lm_lr": ${EMB_LM_LR},
        "forget_weight": ${FORGET_WEIGHT},
        "forget_margin": ${FORGET_MARGIN},
        "active_logit_margin": ${ACTIVE_LOGIT_MARGIN},
        "selection_logit_margin": ${SELECTION_LOGIT_MARGIN},
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
echo "Locked ZeroUnlearn-style MQuAKE SURE track complete."
echo "Stage 1/2: ${FORGET_NUM} sampled forget instances, 0 benchmark retain/questions."
echo "Final eval: same ${FORGET_NUM} forget + ${EVAL_RETAIN_NUM} retain instances."
echo "Results: ${OUTPUT_ROOT}/seed*/official_eval_locked.json"
