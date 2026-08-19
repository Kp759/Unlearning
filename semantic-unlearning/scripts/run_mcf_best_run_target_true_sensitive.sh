#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL="${1:?Usage: bash scripts/run_mcf_best_run_target_true_sensitive.sh MODEL [MCF_JSON]}"
MCF="${2:-data/multi_counterfact.json}"
WIKIDATA_DIR="${WIKIDATA_DIR:-data/wikidata}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/mcf_best_run_target_true_sensitive}"
PROTOCOL_DIR="${OUTPUT_ROOT}/protocol"
VISIBLE="${PROTOCOL_DIR}/repair_visible_mcf_target_true_sensitive.json"
SEEDS_TEXT="${MCF_SEEDS:-1}"
FORGET_NUM="${MCF_FORGET_NUM:-50}"
RETAIN_EVAL_NUM="${MCF_RETAIN_EVAL_NUM:-1000}"
DTYPE="${DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-single}"

# Exact registered clean MCF best-run mechanics/hyperparameters.
STEPS="${MCF_STEPS:-600}"
BATCH_SIZE="${MCF_BATCH_SIZE:-1}"
EMB_LM_LR="${MCF_EMB_LM_LR:-0.0001}"
FORGET_WEIGHT="${MCF_FORGET_WEIGHT:-2.0}"
FORGET_MARGIN="${MCF_FORGET_MARGIN:-1.0}"
ACTIVE_MARGIN="${REPAIR_ACTIVE_MARGIN:-0.25}"
REPAIR_STEPS="${REPAIR_STEPS:-100}"
REPAIR_LR="${REPAIR_LR:-0.005}"
HINGE_WEIGHT="${HINGE_WEIGHT:-2.0}"
DELTA_L2_LAMBDA="${DELTA_L2_LAMBDA:-0.0001}"
REPAIR_RANK="${REPAIR_RANK:-2}"
MARGIN_BATCH_SIZE="${MARGIN_BATCH_SIZE:-4}"

read -r -a SEEDS <<< "${SEEDS_TEXT}"
test -f "${MCF}"
test -d "${MODEL}"
test -d "${WIKIDATA_DIR}"
mkdir -p "${OUTPUT_ROOT}"

rm -rf "${PROTOCOL_DIR}"
mkdir -p "${PROTOCOL_DIR}"
python scripts/build_mcf_best_run_target_true_locked_split.py \
  --mcf-path "${MCF}" --output-dir "${PROTOCOL_DIR}" \
  --seeds "${SEEDS[@]}" --forget-num "${FORGET_NUM}" \
  --retain-num "${RETAIN_EVAL_NUM}"

test -f "${VISIBLE}"

for SEED in "${SEEDS[@]}"; do
  ROOT="${OUTPUT_ROOT}/seed${SEED}"
  STAGE1="${ROOT}/setting5e_best_run_mirrored"
  STAGE1_CKPT="${STAGE1}/emb_lm_all_restore_post_training_true/checkpoint"
  STAGE1_CONFIG="${STAGE1}/config_used.json"
  STAGE2="${ROOT}/repair_best_run_mirrored"
  STAGE2_CKPT="${STAGE2}/checkpoint"
  SEED_MANIFEST="${PROTOCOL_DIR}/seed${SEED}_manifest.json"
  BASE_EVAL="${ROOT}/base_original_mcf_eval.json"
  POST_EVAL="${ROOT}/post_original_mcf_eval.json"
  PAPER_EVAL="${ROOT}/target_true_sensitive_eval.json"

  rm -rf "${ROOT}"
  mkdir -p "${ROOT}"

  echo "===== MCF SEED ${SEED}: BEST-RUN MIRROR STAGE 1 ====="
  echo "training target_new = ORIGINAL target_true (sensitive)"
  echo "training target_true = ORIGINAL target_new (reference)"
  python scripts/mcf_forget_only_setting5e.py \
    --model-path "${MODEL}" \
    --mcf-cache-path "${VISIBLE}" \
    --output-dir "${STAGE1}" \
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

  test -d "${STAGE1_CKPT}"
  test -f "${STAGE1_CONFIG}"

  echo "===== MCF SEED ${SEED}: BEST-RUN MIRROR STAGE 2 ====="
  python scripts/mcf_forget_only_active_repair.py \
    --model-path "${STAGE1_CKPT}" \
    --base-model-path "${MODEL}" \
    --experiment-config-path "${STAGE1_CONFIG}" \
    --output-dir "${STAGE2}" \
    --mcf-cache-path "${VISIBLE}" \
    --sample-mode official \
    --seed "${SEED}" \
    --forget-num "${FORGET_NUM}" \
    --retain-num 0 \
    --repair-mode minimal_optimize \
    --active-margin "${ACTIVE_MARGIN}" \
    --repair-steps "${REPAIR_STEPS}" \
    --repair-lr "${REPAIR_LR}" \
    --repair-optimizer adamw \
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

  test -d "${STAGE2_CKPT}"
  test -f "${STAGE2}/repair_summary.json"

  echo "===== MCF SEED ${SEED}: ORIGINAL UNSWAPPED BASE/POST EVAL ====="
  python scripts/mcf_zero_unlearn_official_eval.py \
    --model-dir "${MODEL}" --mcf-path "${MCF}" --wikidata-dir "${WIKIDATA_DIR}" \
    --out "${BASE_EVAL}" --unlearn-num "${FORGET_NUM}" --retain-num "${RETAIN_EVAL_NUM}" \
    --seed "${SEED}" --sample-mode official --dtype "${DTYPE}" --device-map "${DEVICE_MAP}"

  python scripts/mcf_zero_unlearn_official_eval.py \
    --model-dir "${STAGE2_CKPT}" --mcf-path "${MCF}" --wikidata-dir "${WIKIDATA_DIR}" \
    --out "${POST_EVAL}" --unlearn-num "${FORGET_NUM}" --retain-num "${RETAIN_EVAL_NUM}" \
    --seed "${SEED}" --sample-mode official --dtype "${DTYPE}" --device-map "${DEVICE_MAP}"

  python scripts/annotate_ppl_provenance.py \
    --eval-json "${BASE_EVAL}" --model-dir "${MODEL}" --wikidata-dir "${WIKIDATA_DIR}"
  python scripts/annotate_ppl_provenance.py \
    --eval-json "${POST_EVAL}" --model-dir "${STAGE2_CKPT}" --wikidata-dir "${WIKIDATA_DIR}"

  python scripts/evaluate_mcf_target_true_sensitive.py \
    --base-eval-json "${BASE_EVAL}" --post-eval-json "${POST_EVAL}" \
    --split-manifest "${SEED_MANIFEST}" --out "${PAPER_EVAL}"

done

python scripts/aggregate_mcf_target_true_sensitive.py \
  --root "${OUTPUT_ROOT}" --seeds "${SEEDS[@]}"

echo "Best-run-mirrored target-true-sensitive MCF complete: ${OUTPUT_ROOT}"
