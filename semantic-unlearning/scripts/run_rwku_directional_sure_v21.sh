#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${script_dir}/.." && pwd)"
cd "$project_dir"

if [[ "$#" -ne 3 ]]; then
  echo "Usage: bash scripts/run_rwku_directional_sure_v21.sh MODEL CORPUS_DIR WIKIPEDIA_DIR" >&2
  exit 2
fi

MODEL="$1"
CORPUS="$2"
WIKI="$3"
EXPERIMENT_ID="rwku-directional-sure-v21-stephen-king-seed0"
CONFIG="config/rwku/directional_sure_v21_seed0.json"
OUTPUT_ROOT="${RWKU_DIRECTIONAL_SURE_V21_OUTPUT_ROOT:-outputs/rwku_directional_sure_v21}"
TRAINING_BUNDLE="$CORPUS/generated_training_bundle.json"
GENERATOR_RECEIPT="$CORPUS/generator_receipt.json"

[[ -d "$MODEL" ]] || { echo "Missing model: $MODEL" >&2; exit 2; }
[[ -f "$TRAINING_BUNDLE" ]] || { echo "Missing training bundle: $TRAINING_BUNDLE" >&2; exit 2; }
[[ -f "$GENERATOR_RECEIPT" ]] || { echo "Missing generator receipt: $GENERATOR_RECEIPT" >&2; exit 2; }
[[ -d "$WIKI" ]] || { echo "Missing Wikipedia dataset: $WIKI" >&2; exit 2; }
[[ -f "$CONFIG" ]] || { echo "Missing configuration: $CONFIG" >&2; exit 2; }

RUN="$OUTPUT_ROOT/$EXPERIMENT_ID"
if [[ -e "$RUN" ]]; then
  echo "Refusing to overwrite Directional SURE v2.1 run: $RUN" >&2
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

python scripts/rwku_directional_sure_v21.py \
  --model-path "$MODEL" \
  --training-bundle "$TRAINING_BUNDLE" \
  --generator-receipt "$GENERATOR_RECEIPT" \
  --wikipedia-dir "$WIKI" \
  --output-root "$OUTPUT_ROOT" \
  --experiment-id "$EXPERIMENT_ID" \
  --configuration "$CONFIG" \
  ${RWKU_DIRECTIONAL_SAVE_CHECKPOINT:+--save-checkpoint}

echo "RWKU Directional SURE v2.1 development run complete: $RUN"
