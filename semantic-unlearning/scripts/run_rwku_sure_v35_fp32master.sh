#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${script_dir}/.." && pwd)"
cd "$project_dir"

if [[ "$#" -ne 4 ]]; then
  echo "Usage: bash scripts/run_rwku_sure_v35_fp32master.sh MODEL CORPUS_DIR WIKIPEDIA_DIR SOURCE_HEAD_ONLY_RUN" >&2
  exit 2
fi

MODEL="$1"
CORPUS="$2"
WIKI="$3"
SOURCE="$4"
EXPERIMENT_ID="rwku-h-w1k-stephen-king-emb-head-hidden-direction-seed0-v35-kl-fp32master"
CONFIG="config/rwku/sure_v35_emb_head_hidden_direction_kl_w1k_fp32master_seed0.json"
OUTPUT_ROOT="${RWKU_V35_FP32_OUTPUT_ROOT:-outputs/rwku_v35_fp32master}"
UTILITY_CACHE="${RWKU_H_W1K_UTILITY_CACHE:?RWKU_H_W1K_UTILITY_CACHE must be set}"
TRAINING_BUNDLE="$CORPUS/generated_training_bundle.json"
GENERATOR_RECEIPT="$CORPUS/generator_receipt.json"

[[ -d "$MODEL" ]]
[[ -f "$TRAINING_BUNDLE" ]]
[[ -f "$GENERATOR_RECEIPT" ]]
[[ -d "$WIKI" ]]
[[ -d "$SOURCE" ]]
[[ -f "$UTILITY_CACHE" ]]
[[ -f "$CONFIG" ]]

RUN="$OUTPUT_ROOT/$EXPERIMENT_ID"
if [[ -e "$RUN" ]]; then
  echo "Refusing to overwrite v3.5 FP32-master run: $RUN" >&2
  exit 2
fi

python scripts/rwku_experiment.py \
  --seed 0 \
  --stage prepare \
  --training-source target_only_generated_entity_corpus \
  --experiment-id "$EXPERIMENT_ID" \
  --model-path "$MODEL" \
  --output-root "$OUTPUT_ROOT" \
  --data-root "${RWKU_DATA_ROOT:-data/rwku}" \
  --generated-entity-fact-bundle "$TRAINING_BUNDLE" \
  --generator-receipt "$GENERATOR_RECEIPT" \
  ${RWKU_NO_DOWNLOAD:+--no-download}

python scripts/run_rwku_sure_v35_fp32master.py \
  --model-path "$MODEL" \
  --training-bundle "$TRAINING_BUNDLE" \
  --generator-receipt "$GENERATOR_RECEIPT" \
  --utility-cache "$UTILITY_CACHE" \
  --wikipedia-dir "$WIKI" \
  --source-head-only-run "$SOURCE" \
  --output-root "$OUTPUT_ROOT" \
  --experiment-id "$EXPERIMENT_ID" \
  --configuration "$CONFIG"

echo "RWKU v3.5 FP32-master development run complete: $RUN"
