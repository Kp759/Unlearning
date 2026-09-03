#!/usr/bin/env bash
set -euo pipefail

# Training-only MCF V1 preflight. This process may fit the overlap-aware
# embedding code and contextual classifier, but it cannot construct an
# actuator or open any official evaluation prompt.
#
# Required environment:
#   MODEL_PATH    Base causal LM
#   WIKIDATA_DIR  training-safe Wikipedia dataset saved with datasets
#   MCF_PATH      full MCF JSON, visible only to the direct-only split builder
#
# Usage:
#   bash scripts/run_mcf_internal_contextual_rewiring_v1_preflight_manual.sh \
#     outputs/mcf_internal_contextual_rewiring_v1_preflight_seed1_3b_aws_v6_2

OUTPUT_DIR=${1:?fresh output directory required}
MODEL_PATH=${MODEL_PATH:?MODEL_PATH is required}
WIKIDATA_DIR=${WIKIDATA_DIR:?WIKIDATA_DIR is required}
MCF_SOURCE=${MCF_PATH:?MCF_PATH is required for the split-builder process}

if [ -e "$OUTPUT_DIR" ]; then
  echo "ERROR: V1 preflight output already exists: $OUTPUT_DIR" >&2
  exit 1
fi

REGISTRY=protocols/mcf_internal_contextual_rewiring_v1_registry.json

python -u scripts/build_mcf_internal_contextual_rewiring_v1_split.py \
  --mcf-path "$MCF_SOURCE" \
  --output-dir "$OUTPUT_DIR/protocol" \
  --seed 1 \
  --forget-num 50

# The learner receives only the direct-only file. Remove every source/evaluator
# path from its environment before model loading.
unset MCF_PATH OFFICIAL OFFICIAL_DIR OFFICIAL_MCF_PATH MCF_OFFICIAL_OUTPUT
unset RECOVERY RECOVERY_DIR RETAIN_PATH PPL_PATH ALIAS_EVAL_PATH
unset ADVERSARIAL_EVAL_PATH

mkdir -p "$OUTPUT_DIR/logs"

python -u scripts/run_mcf_internal_contextual_rewiring_v1_preflight.py \
  --model-path "$MODEL_PATH" \
  --training-visible-path \
    "$OUTPUT_DIR/protocol/training_visible_internal_rewiring_direct.json" \
  --split-manifest "$OUTPUT_DIR/protocol/split_manifest.json" \
  --experiment-registry "$REGISTRY" \
  --wikidata-dir "$WIKIDATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --seed 1 \
  --forget-num 50 \
  --candidate-layers 8,12,16,20 \
  --subject-code-rank 8 \
  --detector-rank 8 \
  --embedding-ridge-lambda 0.0001 \
  --embedding-relative-row-cap 0.5 \
  --embedding-frequency-alpha 0.25 \
  --code-nearest-key-margin 0.05 \
  --frequency-doc-start 20 \
  --frequency-docs 12000 \
  --corpus-fit-prompts 1000 \
  --corpus-development-prompts 1000 \
  --corpus-calibration-prompts 1000 \
  --corpus-certification-prompts 6000 \
  --shared-fit-prompts 25 \
  --shared-development-prompts 25 \
  --shared-calibration-prompts 25 \
  --shared-certification-prompts 100 \
  --wrong-relations-fit 8 \
  --wrong-relations-other 4 \
  --same-relation-other-subjects 4 \
  --classifier-steps 400 \
  --classifier-check-every 20 \
  --classifier-lr 0.01 \
  --classifier-weight-decay 0.01 \
  --classifier-positive-floor 1.0 \
  --classifier-negative-ceiling -1.0 \
  --classifier-auxiliary-weight 0.5 \
  --classifier-softmin-temperature 0.1 \
  --minimum-certification-negative-cells 300000 \
  --minimum-certification-prompts 6000 \
  --capture-batch-size 8 \
  --dtype bf16 \
  --device-map single \
  2>&1 | tee "$OUTPUT_DIR/logs/preflight.log"

printf '\nMCF V1 training-only preflight finished.\n'
printf 'Completion: %s\n' "$OUTPUT_DIR/method/completion.json"
printf 'Certificate: %s\n' "$OUTPUT_DIR/method/classifier_certificate.json"
