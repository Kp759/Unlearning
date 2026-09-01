#!/usr/bin/env bash
set -euo pipefail

# Training-only V2.1 folded sensitivity run. The split-builder process alone
# sees MCF_PATH; learner/evaluator paths are removed before model loading.
#
# Required environment: MODEL_PATH, WIKIDATA_DIR, MCF_PATH
# Usage:
#   bash scripts/run_mcf_folded_sensitivity_rewiring_v2_1_manual.sh OUTPUT_DIR

OUTPUT_DIR=${1:?fresh V2.1 output directory required}
MODEL_PATH=${MODEL_PATH:?MODEL_PATH is required}
WIKIDATA_DIR=${WIKIDATA_DIR:?WIKIDATA_DIR is required}
MCF_SOURCE=${MCF_PATH:?MCF_PATH is required for the split-builder process}

if [ -e "$OUTPUT_DIR" ]; then
  echo "ERROR: V2.1 output already exists: $OUTPUT_DIR" >&2
  exit 1
fi

REGISTRY=protocols/mcf_folded_sensitivity_rewiring_v2_1_registry.json

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

python -u scripts/run_mcf_folded_sensitivity_rewiring_v2_1.py \
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
  --targeted-fit-per-row 2 \
  --targeted-development-per-row 1 \
  --targeted-certification-per-row 1 \
  --head-ridge 0.0001 \
  --output-rank-cap 512 \
  --output-relative-cap 0.15 \
  --minimum-signed-correction 0.1 \
  --input-jacobian-sketches 128 \
  --input-rank-cap 64 \
  --input-relative-cap 0.5 \
  --input-frequency-alpha 0.25 \
  --rescue-steps 400 \
  --check-every 20 \
  --head-refit-every 100 \
  --forget-batch-size 8 \
  --protection-batch-size 32 \
  --capture-batch-size 8 \
  --per-step-cap-fraction 0.01 \
  --forget-margin-target 6.0 \
  --minimum-forget-margin 0.1 \
  --protection-topk 64 \
  --protected-kl-mean-max 0.0001 \
  --protected-kl-max 0.01 \
  --protected-top1-drift-max 0.05 \
  --dtype bf16 \
  --device-map single \
  2>&1 | tee "$OUTPUT_DIR/logs/training.log"

printf '\nMCF V2.1 training-only run finished.\n'
printf 'Completion: %s\n' "$OUTPUT_DIR/method/completion.json"
