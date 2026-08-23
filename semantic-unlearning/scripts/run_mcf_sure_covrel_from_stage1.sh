#!/usr/bin/env bash
set -euo pipefail

# Run SURE-MCF CovRel v7 from an already frozen directional Stage-1 checkpoint.
#
# Usage:
#   bash scripts/run_mcf_sure_covrel_from_stage1.sh \
#     /path/to/stage1/checkpoint \
#     /path/to/training_visible_mcf_target_true.json \
#     /path/to/split_manifest.json \
#     /path/to/data/wikidata
#
# The Wikipedia covariance cache excludes train docs 0..19, which are reserved
# by mcf_zero_unlearn_official_eval.py for PPL evaluation.

STAGE1=${1:?Stage-1 checkpoint required}
VISIBLE=${2:?training-visible JSON required}
MANIFEST=${3:?split manifest required}
WIKIDATA=${4:?Wikipedia load_from_disk directory required}

SEED=${SEED:-1}
FORGET_NUM=${FORGET_NUM:-50}
OUT_ROOT=${OUT_ROOT:-outputs/mcf_covrel_fullrow_seed${SEED}}
COV_CACHE=${COV_CACHE:-$OUT_ROOT/wiki_lmhead_cov_1000x100_seed${SEED}.pt}
STAGE2_OUT=${STAGE2_OUT:-$OUT_ROOT/stage2_covrel_v7}

NUM_WIKI_DOCS=${NUM_WIKI_DOCS:-1000}
PROMPTS_PER_WIKI_DOC=${PROMPTS_PER_WIKI_DOC:-100}
WIKI_SKIP_FIRST_DOCS=${WIKI_SKIP_FIRST_DOCS:-20}
WIKI_MAX_DOC_TOKENS=${WIKI_MAX_DOC_TOKENS:-1024}
WIKI_BATCH_SIZE=${WIKI_BATCH_SIZE:-2}

REPAIR_STEPS=${REPAIR_STEPS:-800}
REPAIR_LR=${REPAIR_LR:-0.005}
TRAIN_MCF_MARGIN=${TRAIN_MCF_MARGIN:-0.10}
FINAL_MCF_MARGIN=${FINAL_MCF_MARGIN:-0.05}
PROTECTED_MCF_MARGIN_FLOOR=${PROTECTED_MCF_MARGIN_FLOOR:-0.0}
RELATION_CONTROLS_PER_RECORD=${RELATION_CONTROLS_PER_RECORD:-4}
RELATION_KL_WEIGHT=${RELATION_KL_WEIGHT:-1.0}
RELATION_KL_MAX=${RELATION_KL_MAX:-0.01}
COV_RIDGE=${COV_RIDGE:-0.10}
REPAIR_L2=${REPAIR_L2:-1e-6}
BATCH_SIZE=${BATCH_SIZE:-8}
CHECK_EVERY=${CHECK_EVERY:-25}
DTYPE=${DTYPE:-bf16}
DEVICE_MAP=${DEVICE_MAP:-single}

mkdir -p "$OUT_ROOT"

if [[ ! -f "$COV_CACHE" ]]; then
  python -u scripts/build_wiki_lmhead_covariance.py \
    --model-path "$STAGE1" \
    --wikidata-dir "$WIKIDATA" \
    --output "$COV_CACHE" \
    --seed "$SEED" \
    --num-docs "$NUM_WIKI_DOCS" \
    --prompts-per-doc "$PROMPTS_PER_WIKI_DOC" \
    --skip-first-docs "$WIKI_SKIP_FIRST_DOCS" \
    --max-doc-tokens "$WIKI_MAX_DOC_TOKENS" \
    --batch-size "$WIKI_BATCH_SIZE" \
    --dtype "$DTYPE" \
    --device-map "$DEVICE_MAP" \
    2>&1 | tee "$OUT_ROOT/wiki_covariance.log"
else
  echo "Reusing covariance cache: $COV_CACHE"
fi

python -u scripts/mcf_sure_covrel_fullrow_stage2_v7.py \
  --model-path "$STAGE1" \
  --training-visible-path "$VISIBLE" \
  --split-manifest "$MANIFEST" \
  --wiki-covariance "$COV_CACHE" \
  --output-dir "$STAGE2_OUT" \
  --seed "$SEED" \
  --forget-num "$FORGET_NUM" \
  --repair-steps "$REPAIR_STEPS" \
  --repair-lr "$REPAIR_LR" \
  --train-mcf-margin "$TRAIN_MCF_MARGIN" \
  --final-mcf-margin "$FINAL_MCF_MARGIN" \
  --protected-mcf-margin-floor "$PROTECTED_MCF_MARGIN_FLOOR" \
  --relation-controls-per-record "$RELATION_CONTROLS_PER_RECORD" \
  --relation-kl-weight "$RELATION_KL_WEIGHT" \
  --relation-kl-max "$RELATION_KL_MAX" \
  --cov-ridge "$COV_RIDGE" \
  --repair-l2 "$REPAIR_L2" \
  --batch-size "$BATCH_SIZE" \
  --check-every "$CHECK_EVERY" \
  --dtype "$DTYPE" \
  --device-map "$DEVICE_MAP" \
  2>&1 | tee "$OUT_ROOT/stage2_covrel_v7.log"

printf '\nCovRel v7 complete.\n'
printf 'Covariance: %s\n' "$COV_CACHE"
printf 'Final checkpoint: %s\n' "$STAGE2_OUT/checkpoint"
printf 'Summary: %s\n' "$STAGE2_OUT/repair_summary.json"
