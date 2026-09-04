#!/usr/bin/env bash
set -euo pipefail

# V1.3 multi-view relation-robust private-vocabulary rewiring.
#
# LEAKAGE FIREWALL:
#   1) Only the split builder may read the full MCF JSON.
#   2) After the split is serialized, MCF_PATH is unset.
#   3) The synthetic-view generator accepts only the sanitized direct forget file.
#   4) The learner accepts only the sanitized protocol + generated training views.
#   5) No official paraphrase/neighborhood/eval-retain prompt text is serialized.
#
# Seed 1 is DEVELOPMENT ONLY because its aggregate official metrics were already
# inspected in V1.1/V1.2. Final certification must use a new untouched seed.
#
# Required environment: MODEL_PATH, MCF_PATH
# Usage: bash scripts/run_mcf_private_vocab_rewiring_v1_3_multiview_manual.sh OUTPUT_DIR

OUTPUT_DIR=${1:?fresh V1.3 output directory required}
MODEL_PATH=${MODEL_PATH:?MODEL_PATH is required}
MCF_SOURCE=${MCF_PATH:?MCF_PATH is required only for split-builder process}

PROTO_TMP="${OUTPUT_DIR}.protocol_tmp"
LOG_TMP="${OUTPUT_DIR}.training.log.tmp"

for path in "$OUTPUT_DIR" "$PROTO_TMP" "$LOG_TMP"; do
  if [ -e "$path" ]; then
    echo "ERROR: V1.3 path already exists: $path" >&2
    exit 1
  fi
done

REGISTRY=protocols/mcf_private_vocab_rewiring_v1_3_registry.json

# ONLY PROCESS ALLOWED TO READ FULL MCF.
python -u scripts/build_mcf_biendpoint_nullspace_rewiring_v2_split.py \
  --mcf-path "$MCF_SOURCE" \
  --output-dir "$PROTO_TMP" \
  --seed 1 \
  --forget-num 50 \
  --official-retain-num 1000 \
  --protection-fit-num 2000 \
  --protection-development-num 500 \
  --protection-certification-num 1000

# Close access to the full source before any paraphrase generation or learning.
unset MCF_PATH OFFICIAL OFFICIAL_DIR OFFICIAL_MCF_PATH MCF_OFFICIAL_OUTPUT
unset RECOVERY RECOVERY_DIR RETAIN_PATH PPL_PATH ALIAS_EVAL_PATH
unset ADVERSARIAL_EVAL_PATH
unset MCF_SOURCE

VIEWS="$PROTO_TMP/training_visible_multiview_forget.json"

# Generate canonical + 4 synthetic training-only relation views from the
# sanitized direct forget file.  This script has no --mcf-path argument.
python -u scripts/build_mcf_private_vocab_rewiring_v1_3_training_views.py \
  --model-path "$MODEL_PATH" \
  --forget-direct "$PROTO_TMP/training_visible_forget_direct.json" \
  --out "$VIEWS" \
  --views-per-case 5 \
  --candidates-per-attempt 10 \
  --max-attempts 4 \
  --max-new-tokens 320 \
  --temperature 0.8 \
  --top-p 0.9 \
  --seed 13131 \
  --dtype bf16 \
  --max-true-logprob-drop 3.0 \
  --max-margin-degradation 1.0

export MCF_V13_VIEW_CORPUS="$VIEWS"
export MCF_V13_VIEW_CHUNK=16

python -u scripts/run_mcf_private_vocab_rewiring_v1_3_multiview.py \
  --model-path "$MODEL_PATH" \
  --protocol-dir "$PROTO_TMP" \
  --experiment-registry "$REGISTRY" \
  --output-dir "$OUTPUT_DIR" \
  --seed 1 \
  --forget-num 50 \
  --dtype bf16 \
  --steps 600 \
  --forget-batch-size 8 \
  --retain-batch-size 16 \
  --check-every 25 \
  --learning-rate 0.001 \
  --minimum-forget-margin 0.1 \
  --train-margin-target 0.1 \
  --retain-kl-weight 20.0 \
  --anchor-weight 0.001 \
  --relative-row-cap 0.5 \
  --topk 64 \
  --initial-equivalence-kl-max 0.0000001 \
  --initial-margin-drift-max 0.00001 \
  --retain-kl-mean-max 0.0001 \
  --nonclone-certification-prompts 64 \
  --save-model \
  2>&1 | tee "$LOG_TMP"

unset MCF_V13_VIEW_CORPUS MCF_V13_VIEW_CHUNK

mkdir -p "$OUTPUT_DIR/logs"
mv "$LOG_TMP" "$OUTPUT_DIR/logs/training.log"
mv "$PROTO_TMP" "$OUTPUT_DIR/protocol"

printf '\nV1.3 multi-view private-vocabulary rewiring finished.\n'
printf 'Method report: %s\n' "$OUTPUT_DIR/method/private_vocab_rewiring_v1_3_multiview.json"
printf 'Completion: %s\n' "$OUTPUT_DIR/method/completion.json"
printf 'Checkpoint/model artifact: %s\n' "$OUTPUT_DIR/model"
printf 'Training-view corpus: %s\n' "$OUTPUT_DIR/protocol/training_visible_multiview_forget.json"
printf 'Protocol: %s\n' "$OUTPUT_DIR/protocol"
printf 'Final-certification status: DEVELOPMENT ONLY; use a new untouched seed for final held-out evaluation.\n'
