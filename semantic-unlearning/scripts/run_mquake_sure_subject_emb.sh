#!/usr/bin/env bash
set -euo pipefail

# MQuAKE SURE subject-keyed embedding experiment.
#
# Same architecture as the MCF runner: GA on the sensitive answer, trained
# ONLY on subject input-embedding rows, with the LM head untied and frozen.
#
# One structural difference. MCF supplies target_new, so forgetting is a
# pairwise margin. MQuAKE's locked view carries target_true only and its Eff
# metric scores next-token argmax accuracy on the sensitive answer, so the
# reference is the highest-logit non-sensitive token cached on the BASE model
# -- whatever the model would otherwise say.
#
# Builds the locked split if it is not already present, trains Stage 1, then
# runs the official evaluation.
#
# Usage:
#   bash scripts/run_mquake_sure_subject_emb.sh /path/to/model

MODEL_PATH=${1:?model path required}

SEED=${SEED:-1}
FORGET_NUM=${FORGET_NUM:-50}
RETAIN_NUM=${RETAIN_NUM:-1000}
MQUAKE_PATH=${MQUAKE_PATH:-data/MQuAKE-CF-3k-v2.json}
OUT_ROOT=${OUT_ROOT:-outputs/mquake_subject_emb_seed${SEED}}

STAGE1_STEPS=${STAGE1_STEPS:-1200}
STAGE1_BATCH_SIZE=${STAGE1_BATCH_SIZE:-4}
STAGE1_CACHE_BATCH_SIZE=${STAGE1_CACHE_BATCH_SIZE:-8}
STAGE1_LR=${STAGE1_LR:-5e-4}
MARGIN_WEIGHT=${MARGIN_WEIGHT:-100.0}
TRAIN_MARGIN=${TRAIN_MARGIN:-6.0}
DELTA_L2=${DELTA_L2:-1e-4}
CONSTRAINT_MARGIN=${CONSTRAINT_MARGIN:-0.05}

MAX_SUBJECT_TOKEN_FREQUENCY=${MAX_SUBJECT_TOKEN_FREQUENCY:-1000000000}
ROW_NORM_CAP=${ROW_NORM_CAP:-0.0}
ROW_NORM_CAP_FREQUENCY_ALPHA=${ROW_NORM_CAP_FREQUENCY_ALPHA:-0.0}

# Frequency corpus must stay clear of official PPL's hardcoded [:20].
FREQ_WIKI=${FREQ_WIKI:-data/wikidata}
PPL_WIKI=${PPL_WIKI:-data/wikidata}
FREQUENCY_DOCS=${FREQUENCY_DOCS:-5000}
FREQUENCY_DOC_START=${FREQUENCY_DOC_START:-20}

DTYPE=${DTYPE:-bf16}
DEVICE_MAP=${DEVICE_MAP:-single}

SPLIT_ROOT="$OUT_ROOT/protocol"
STAGE1_OUT="$OUT_ROOT/stage1_subject_emb"
VISIBLE="$SPLIT_ROOT/training_visible_forget.json"
MANIFEST="$SPLIT_ROOT/split_manifest.json"

mkdir -p "$OUT_ROOT"

if [ ! -f "$VISIBLE" ] || [ ! -f "$MANIFEST" ]; then
  echo "Building locked MQuAKE split (seed $SEED)..."
  python -u scripts/build_mquake_sure_canonical_split.py \
    --mquake-path "$MQUAKE_PATH" \
    --output-dir "$SPLIT_ROOT" \
    --seed "$SEED" \
    --forget-num "$FORGET_NUM" \
    --retain-num "$RETAIN_NUM" \
    2>&1 | tee "$OUT_ROOT/split_build.log"
fi

test -f "$VISIBLE" || { echo "split builder produced no $VISIBLE"; exit 1; }
test -f "$MANIFEST" || { echo "split builder produced no $MANIFEST"; exit 1; }

python -u scripts/mquake_sure_subject_directional_emb_stage1.py \
  --model-path "$MODEL_PATH" \
  --training-visible-path "$VISIBLE" \
  --split-manifest "$MANIFEST" \
  --output-dir "$STAGE1_OUT" \
  --seed "$SEED" \
  --forget-num "$FORGET_NUM" \
  --steps "$STAGE1_STEPS" \
  --batch-size "$STAGE1_BATCH_SIZE" \
  --cache-batch-size "$STAGE1_CACHE_BATCH_SIZE" \
  --lr "$STAGE1_LR" \
  --margin-weight "$MARGIN_WEIGHT" \
  --train-margin "$TRAIN_MARGIN" \
  --delta-l2 "$DELTA_L2" \
  --stage1-constraint-margin "$CONSTRAINT_MARGIN" \
  --max-subject-token-frequency "$MAX_SUBJECT_TOKEN_FREQUENCY" \
  --row-norm-cap "$ROW_NORM_CAP" \
  --row-norm-cap-frequency-alpha "$ROW_NORM_CAP_FREQUENCY_ALPHA" \
  --wikidata-dir "$FREQ_WIKI" \
  --frequency-docs "$FREQUENCY_DOCS" \
  --frequency-doc-start "$FREQUENCY_DOC_START" \
  --dtype "$DTYPE" \
  --device-map "$DEVICE_MAP" \
  2>&1 | tee "$OUT_ROOT/stage1_subject_emb.log"

python -u scripts/mquake_zero_unlearn_official_eval.py \
  --model-dir "$STAGE1_OUT/checkpoint" \
  --mquake-path "$MQUAKE_PATH" \
  --wikidata-dir "$PPL_WIKI" \
  --split-manifest "$MANIFEST" \
  --out "$OUT_ROOT/stage1_official_eval.json" \
  --unlearn-num "$FORGET_NUM" \
  --retain-num "$RETAIN_NUM" \
  --seed "$SEED" \
  --dtype "$DTYPE" \
  --device-map "$DEVICE_MAP" \
  2>&1 | tee "$OUT_ROOT/stage1_official_eval.log"

printf '\nFinished MQuAKE subject-keyed embedding run.\n'
printf 'Split   : %s\n' "$SPLIT_ROOT"
printf 'Stage 1 : %s\n' "$STAGE1_OUT/checkpoint"
printf 'Config  : %s\n' "$STAGE1_OUT/stage1_config.json"
printf 'Eval    : %s\n' "$OUT_ROOT/stage1_official_eval.json"
