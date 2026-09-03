#!/usr/bin/env bash
set -euo pipefail

# Training-only internal MCF V2 run.  The split builder is the only process
# that sees MCF_PATH.  The learner receives direct-only forget/protection
# partitions; all official prompts and PPL documents remain unavailable.
#
# Required environment:
#   MODEL_PATH    Base causal LM
#   WIKIDATA_DIR  Wikipedia dataset saved with datasets
#   MCF_PATH      full MultiCounterFact JSON (split-builder process only)
#
# Usage:
#   bash scripts/run_mcf_biendpoint_nullspace_rewiring_v2_manual.sh \
#     outputs/mcf_biendpoint_nullspace_rewiring_v2_seed1_3b_aws_v6_2

OUTPUT_DIR=${1:?fresh output directory required}
MODEL_PATH=${MODEL_PATH:?MODEL_PATH is required}
WIKIDATA_DIR=${WIKIDATA_DIR:?WIKIDATA_DIR is required}
MCF_SOURCE=${MCF_PATH:?MCF_PATH is required for the split-builder process}

if [ -e "$OUTPUT_DIR" ]; then
  echo "ERROR: V2 output already exists: $OUTPUT_DIR" >&2
  exit 1
fi

REGISTRY=protocols/mcf_biendpoint_nullspace_rewiring_v2_registry.json

python -u scripts/build_mcf_biendpoint_nullspace_rewiring_v2_split.py \
  --mcf-path "$MCF_SOURCE" \
  --output-dir "$OUTPUT_DIR/protocol" \
  --seed 1 \
  --forget-num 50 \
  --official-retain-num 1000 \
  --protection-fit-num 2000 \
  --protection-development-num 500 \
  --protection-certification-num 1000

unset MCF_PATH OFFICIAL OFFICIAL_DIR OFFICIAL_MCF_PATH MCF_OFFICIAL_OUTPUT
unset RECOVERY RECOVERY_DIR RETAIN_PATH PPL_PATH ALIAS_EVAL_PATH
unset ADVERSARIAL_EVAL_PATH

mkdir -p "$OUTPUT_DIR/logs"

python -u scripts/run_mcf_biendpoint_nullspace_rewiring_v2.py \
  --model-path "$MODEL_PATH" \
  --protocol-dir "$OUTPUT_DIR/protocol" \
  --experiment-registry "$REGISTRY" \
  --wikidata-dir "$WIKIDATA_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --seed 1 \
  --forget-num 50 \
  --frequency-doc-start 20 \
  --frequency-docs 12000 \
  --corpus-fit-prompts 1000 \
  --corpus-development-prompts 250 \
  --corpus-certification-prompts 1000 \
  --synthetic-paraphrases 3 \
  --same-subject-prompts 4 \
  --input-jacobian-sketches 64 \
  --input-rank-cap 32 \
  --output-rank-cap 256 \
  --minimum-projected-gradient-norm 0.00000001 \
  --input-relative-cap 0.5 \
  --input-frequency-alpha 0.25 \
  --output-relative-cap 0.1 \
  --steps 1200 \
  --check-every 100 \
  --forget-batch-size 4 \
  --protection-batch-size 8 \
  --capture-batch-size 8 \
  --embedding-lr 0.0005 \
  --lm-head-lr 0.001 \
  --forget-margin-target 6.0 \
  --forget-margin-weight 100.0 \
  --protection-topk 64 \
  --protection-kl-weight 10.0 \
  --protection-top1-weight 10.0 \
  --delta-l2-weight 0.0001 \
  --gradient-clip 1.0 \
  --minimum-forget-margin 0.1 \
  --protected-kl-mean-max 0.0001 \
  --protected-kl-max 0.01 \
  --protected-top1-drift-max 0.05 \
  --dtype bf16 \
  --device-map single \
  2>&1 | tee "$OUTPUT_DIR/logs/training.log"

printf '\nMCF V2 training-only run finished.\n'
printf 'Completion: %s\n' "$OUTPUT_DIR/method/completion.json"
