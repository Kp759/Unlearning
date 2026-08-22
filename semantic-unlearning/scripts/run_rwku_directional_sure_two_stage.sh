#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${script_dir}/.." && pwd)"
cd "$project_dir"

if [[ "$#" -ne 3 ]]; then
  echo "Usage: bash scripts/run_rwku_directional_sure_two_stage.sh MODEL CORPUS_DIR WIKIPEDIA_DIR" >&2
  exit 2
fi

MODEL="$1"
CORPUS="$2"
WIKI="$3"
EXPERIMENT_ID="rwku-directional-sure-two-stage-stephen-king-seed0"
CONFIG="config/rwku/directional_sure_two_stage_seed0.json"
OUTPUT_ROOT="${RWKU_DIRECTIONAL_SURE_TWO_STAGE_OUTPUT_ROOT:-outputs/rwku_directional_sure_two_stage}"
TRAINING_BUNDLE="$CORPUS/generated_training_bundle.json"
GENERATOR_RECEIPT="$CORPUS/generator_receipt.json"

[[ -d "$MODEL" ]] || { echo "Missing model: $MODEL" >&2; exit 2; }
[[ -f "$TRAINING_BUNDLE" ]] || { echo "Missing training bundle: $TRAINING_BUNDLE" >&2; exit 2; }
[[ -f "$GENERATOR_RECEIPT" ]] || { echo "Missing generator receipt: $GENERATOR_RECEIPT" >&2; exit 2; }
[[ -d "$WIKI" ]] || { echo "Missing Wikipedia dataset: $WIKI" >&2; exit 2; }
[[ -f "$CONFIG" ]] || { echo "Missing configuration: $CONFIG" >&2; exit 2; }

RUN="$OUTPUT_ROOT/$EXPERIMENT_ID"
if [[ -e "$RUN" ]]; then
  echo "Refusing to overwrite pure two-stage Directional SURE run: $RUN" >&2
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

python scripts/rwku_directional_sure_two_stage.py \
  --model-path "$MODEL" \
  --training-bundle "$TRAINING_BUNDLE" \
  --generator-receipt "$GENERATOR_RECEIPT" \
  --wikipedia-dir "$WIKI" \
  --output-root "$OUTPUT_ROOT" \
  --experiment-id "$EXPERIMENT_ID" \
  --configuration "$CONFIG" \
  --save-checkpoint

RESULT="$RUN/directional_sure_two_stage/result.json"
[[ -f "$RESULT" ]] || { echo "Missing two-stage result: $RESULT" >&2; exit 1; }
python - "$RESULT" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
result = json.loads(path.read_text())
if result.get("level3_used") is not False:
    raise SystemExit("ERROR: Level 3 / representation repair was unexpectedly used")
if result.get("representation_repair_used") is not False:
    raise SystemExit("ERROR: representation repair was unexpectedly used")
if result.get("transformer_exactly_frozen") is not True:
    raise SystemExit("ERROR: transformer exact-freeze audit failed")
if result.get("feasible") is not True:
    raise SystemExit("ERROR: pure two-stage Directional SURE did not pass all gates")
print("Pure two-stage Directional SURE result PASS")
print("  L1 anchor step:", result.get("level1_anchor_step"))
print("  L2 used:", result.get("level2_used"))
print("  final direct/other:",
      result.get("final_atomic", {}).get("FS"),
      result.get("final_atomic", {}).get("generated_subject_FS"))
PY

echo "RWKU pure two-stage Directional SURE development run complete: $RUN"
