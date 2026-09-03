#!/usr/bin/env bash
set -euo pipefail

# MCF private-vocabulary rewiring V1.
# Required environment: MODEL_PATH, MCF_PATH
# Usage: bash scripts/run_mcf_private_vocab_rewiring_v1_manual.sh OUTPUT_DIR

OUTPUT_DIR=${1:?fresh V1 output directory required}
MODEL_PATH=${MODEL_PATH:?MODEL_PATH is required}
MCF_SOURCE=${MCF_PATH:?MCF_PATH is required for the split-builder process}
PROTOCOL_DIR="${OUTPUT_DIR}.protocol_tmp"
PREFLIGHT_DIR="${OUTPUT_DIR}.preflight_tmp"

for path in "$OUTPUT_DIR" "$PROTOCOL_DIR" "$PREFLIGHT_DIR"; do
  if [ -e "$path" ]; then
    echo "ERROR: V1 path already exists: $path" >&2
    exit 1
  fi
done

REGISTRY=protocols/mcf_private_vocab_rewiring_v1_registry.json

# Keep the split outside OUTPUT_DIR because the trainer deliberately requires a
# fresh output directory.  It is copied into the completed artifact afterward.
python -u scripts/build_mcf_biendpoint_nullspace_rewiring_v2_split.py \
  --mcf-path "$MCF_SOURCE" \
  --output-dir "$PROTOCOL_DIR" \
  --seed 1 \
  --forget-num 50 \
  --official-retain-num 1000 \
  --protection-fit-num 2000 \
  --protection-development-num 500 \
  --protection-certification-num 1000

# Tokenizer-only proof that the 50 forget subjects can be assigned to existing
# reserved slots without changing vocabulary size.  This happens before loading
# the 3B model.
python -u scripts/preflight_mcf_private_vocab_rewiring_v1.py \
  --model-path "$MODEL_PATH" \
  --protocol-dir "$PROTOCOL_DIR" \
  --output-dir "$PREFLIGHT_DIR"

unset MCF_PATH OFFICIAL OFFICIAL_DIR OFFICIAL_MCF_PATH MCF_OFFICIAL_OUTPUT
unset RECOVERY RECOVERY_DIR RETAIN_PATH PPL_PATH ALIAS_EVAL_PATH
unset ADVERSARIAL_EVAL_PATH

python -u scripts/run_mcf_private_vocab_rewiring_v1.py \
  --model-path "$MODEL_PATH" \
  --protocol-dir "$PROTOCOL_DIR" \
  --experiment-registry "$REGISTRY" \
  --output-dir "$OUTPUT_DIR" \
  --seed 1 \
  --forget-num 50 \
  --dtype bf16 \
  --clone-init-steps 150 \
  --clone-init-batch-size 16 \
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
  --clone-init-kl-max 0.005 \
  --retain-kl-mean-max 0.0001 \
  --nonclone-certification-prompts 64 \
  --save-model \
  2>&1 | tee "${OUTPUT_DIR}.training.log"

mkdir -p "$OUTPUT_DIR/protocol"
cp -a "$PROTOCOL_DIR"/. "$OUTPUT_DIR/protocol"/
cp "$PREFLIGHT_DIR/method/tokenizer_preflight.json" \
  "$OUTPUT_DIR/method/tokenizer_preflight.json"
mkdir -p "$OUTPUT_DIR/logs"
mv "${OUTPUT_DIR}.training.log" "$OUTPUT_DIR/logs/training.log"
rm -rf "$PROTOCOL_DIR" "$PREFLIGHT_DIR"

printf '\nMCF private-vocabulary rewiring V1 finished.\n'
printf 'Tokenizer preflight: %s\n' "$OUTPUT_DIR/method/tokenizer_preflight.json"
printf 'Method report: %s\n' "$OUTPUT_DIR/method/private_vocab_rewiring.json"
printf 'Completion: %s\n' "$OUTPUT_DIR/method/completion.json"
printf 'Checkpoint: %s\n' "$OUTPUT_DIR/model"
