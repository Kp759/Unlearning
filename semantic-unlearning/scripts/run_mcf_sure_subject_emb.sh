#!/usr/bin/env bash
set -euo pipefail

# MCF SURE subject-keyed embedding experiment.
#
#   Stage 1: GA on target_true / GD on target_new, trained ONLY on
#            frequency-filtered subject input-embedding rows, with the
#            hidden-state change confined to the closed-form sensitive
#            readout direction u_s = normalize(LM_head[target_true]).
#            The LM head is untied and frozen -- never edited.
#
#   Stage 2: OFF by default (RUN_STAGE2=1 to enable).
#
# Stage 2 is opt-in deliberately. The whole point of Stage 1 is that a
# neighborhood prompt with a different subject contains none of the edited
# rows, so its forward pass is bitwise identical to Base and Spe cannot
# move. Stage 2 edits target_true LM-head rows, which every CounterFact
# neighborhood prompt must also produce -- that is precisely the coupling
# that held Eff/Spe on a trade-off curve across ~20 runs. Measure Stage 1
# alone first so its Spe/PPL claim is actually tested, then add Stage 2
# only for whatever direct failures remain.
#
# Usage:
#   bash scripts/run_mcf_sure_subject_emb.sh \
#     /path/to/model \
#     outputs/mcf_targettrue_clean_seed1/seed1/protocol/training_visible_mcf_target_true.json \
#     outputs/mcf_targettrue_clean_seed1/seed1/protocol/split_manifest.json

MODEL_PATH=${1:?model path required}
VISIBLE=${2:?training-visible JSON required}
MANIFEST=${3:?split manifest required}

SEED=${SEED:-1}
FORGET_NUM=${FORGET_NUM:-50}
OUT_ROOT=${OUT_ROOT:-outputs/mcf_sure_subject_emb_seed${SEED}}

# NOTE: these env vars, if already exported in the calling shell, override
# the fallbacks below and will silently keep stale values even after the
# Python argparse defaults change. Unset them to pick up script-side fixes.
STAGE1_STEPS=${STAGE1_STEPS:-1200}
STAGE1_BATCH_SIZE=${STAGE1_BATCH_SIZE:-4}
STAGE1_CACHE_BATCH_SIZE=${STAGE1_CACHE_BATCH_SIZE:-8}
STAGE1_LR=${STAGE1_LR:-5e-4}
MARGIN_WEIGHT=${MARGIN_WEIGHT:-100.0}
TRAIN_MARGIN=${TRAIN_MARGIN:-1.0}
SURGICAL_WEIGHT=${SURGICAL_WEIGHT:-1.0}
DELTA_L2=${DELTA_L2:-1e-4}
CONSTRAINT_MARGIN=${CONSTRAINT_MARGIN:-0.05}
SYNTHETIC_PARAPHRASES_PER_RECORD=${SYNTHETIC_PARAPHRASES_PER_RECORD:-3}

# Real CounterFact paraphrases prepend an arbitrary unrelated sentence, not a
# formulaic lead-in. The first run (98e34f4) trained on four hand-authored
# prefixes and got 30% synthetic failure but 59% real paraphrase failure.
CORPUS_CONTEXT_PREFIXES=${CORPUS_CONTEXT_PREFIXES:-256}

MAX_SUBJECT_TOKEN_FREQUENCY=${MAX_SUBJECT_TOKEN_FREQUENCY:-100}
WIKIDATA_DIR=${WIKIDATA_DIR:-data/wikidata}
FREQUENCY_DOCS=${FREQUENCY_DOCS:-5000}
# Must stay >= 20; official PPL is hardcoded to documents [:20].
FREQUENCY_DOC_START=${FREQUENCY_DOC_START:-20}

DTYPE=${DTYPE:-bf16}
DEVICE_MAP=${DEVICE_MAP:-single}
CANDIDATE_SCALES=${CANDIDATE_SCALES:-1,.875,.75,.625,.5,.375,.25,.1875,.125,.09375,.0625,.046875,.03125,.015625,.0078125,0}

RUN_STAGE2=${RUN_STAGE2:-0}

STAGE1_OUT="$OUT_ROOT/stage1_subject_emb"
STAGE2_OUT="$OUT_ROOT/stage2_fullrow_repair"

mkdir -p "$OUT_ROOT"

python -u scripts/mcf_sure_subject_directional_emb_stage1.py \
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
  --surgical-weight "$SURGICAL_WEIGHT" \
  --delta-l2 "$DELTA_L2" \
  --stage1-constraint-margin "$CONSTRAINT_MARGIN" \
  --synthetic-paraphrases-per-record "$SYNTHETIC_PARAPHRASES_PER_RECORD" \
  --corpus-context-prefixes "$CORPUS_CONTEXT_PREFIXES" \
  --max-subject-token-frequency "$MAX_SUBJECT_TOKEN_FREQUENCY" \
  --wikidata-dir "$WIKIDATA_DIR" \
  --frequency-docs "$FREQUENCY_DOCS" \
  --frequency-doc-start "$FREQUENCY_DOC_START" \
  --candidate-scales "$CANDIDATE_SCALES" \
  --dtype "$DTYPE" \
  --device-map "$DEVICE_MAP" \
  2>&1 | tee "$OUT_ROOT/stage1_subject_emb.log"

FINAL_CKPT="$STAGE1_OUT/checkpoint"

if [ "$RUN_STAGE2" = "1" ]; then
  python -u scripts/mcf_sure_fullrow_failure_repair.py \
    --model-path "$STAGE1_OUT/checkpoint" \
    --training-visible-path "$VISIBLE" \
    --split-manifest "$MANIFEST" \
    --output-dir "$STAGE2_OUT" \
    --seed "$SEED" \
    --forget-num "$FORGET_NUM" \
    --constraint-margin "$CONSTRAINT_MARGIN" \
    --synthetic-paraphrases-per-record "$SYNTHETIC_PARAPHRASES_PER_RECORD" \
    --wikidata-dir "$WIKIDATA_DIR" \
    --dtype "$DTYPE" \
    --device-map "$DEVICE_MAP" \
    2>&1 | tee "$OUT_ROOT/stage2_fullrow_repair.log"
  FINAL_CKPT="$STAGE2_OUT/checkpoint"
fi

printf '\nFinished MCF SURE subject-keyed embedding run.\n'
printf 'Stage 1 : %s\n' "$STAGE1_OUT/checkpoint"
printf 'Config  : %s\n' "$STAGE1_OUT/stage1_config.json"
printf 'Final   : %s\n' "$FINAL_CKPT"
if [ "$RUN_STAGE2" != "1" ]; then
  printf 'Stage 2 : skipped (RUN_STAGE2=1 to enable)\n'
fi
