#!/usr/bin/env bash
set -euo pipefail

# Training-only MCF V2.2. The split builder alone sees MCF_PATH.
# Required environment: MODEL_PATH, WIKIDATA_DIR, MCF_PATH
# Usage: bash scripts/run_mcf_joint_forget_retain_endpoint_v2_2_manual.sh OUTPUT_DIR

OUTPUT_DIR=${1:?fresh V2.2 output directory required}
MODEL_PATH=${MODEL_PATH:?MODEL_PATH is required}
WIKIDATA_DIR=${WIKIDATA_DIR:?WIKIDATA_DIR is required}
MCF_SOURCE=${MCF_PATH:?MCF_PATH is required for the split-builder process}

if [ -e "$OUTPUT_DIR" ]; then
  echo "ERROR: V2.2 output already exists: $OUTPUT_DIR" >&2
  exit 1
fi

REGISTRY=protocols/mcf_joint_forget_retain_endpoint_v2_2_registry.json

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

python -u scripts/run_mcf_joint_forget_retain_endpoint_v2_2.py \
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
  --targeted-fit-per-row 1 \
  --targeted-development-per-row 1 \
  --targeted-certification-per-row 1 \
  --steps 1000 \
  --check-every 50 \
  --hard-tail-refresh-every 50 \
  --hard-tail-add 32 \
  --hard-tail-capacity 256 \
  --hard-tail-active 16 \
  --forget-batch-size 16 \
  --random-retain-batch-size 16 \
  --overlap-retain-batch-size 16 \
  --active-retain-maximum 48 \
  --capture-batch-size 8 \
  --forget-margin-target 6.0 \
  --minimum-forget-margin 0.1 \
  --forget-weight 1.0 \
  --retain-kl-weight 100.0 \
  --retain-top1-weight 100.0 \
  --retain-target-weight 100.0 \
  --delta-l2-weight 0.0001 \
  --input-relative-cap 0.5 \
  --input-frequency-alpha 0.25 \
  --output-relative-cap 0.3 \
  --input-step-cap-fraction 0.002 \
  --output-step-cap-fraction 0.002 \
  --protection-topk 64 \
  --protected-kl-mean-max 0.0001 \
  --protected-kl-max 0.01 \
  --protected-top1-drift-max 0.05 \
  --protected-target-drift-max 0.05 \
  --dtype bf16 \
  --device-map single \
  2>&1 | tee "$OUTPUT_DIR/logs/training.log"

printf '\nMCF V2.2 training-only run finished.\n'
printf 'Completion: %s\n' "$OUTPUT_DIR/method/completion.json"
