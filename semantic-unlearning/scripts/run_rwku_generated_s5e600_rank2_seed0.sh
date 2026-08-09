#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 2 ]]; then
  echo "Usage: $0 MODEL_PATH {preflight|prepare|train|evaluate}" >&2
  exit 2
fi

MODEL_PATH=$1
STAGE=$2
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)

EXPERIMENT_ID=rwku-s5e600-rank2-active-sk-v3atomic-seed0-v1
CORPUS="$ROOT/outputs/rwku_target_only/corpus/stephen_king_v3_atomic_seed0_run1"
OUTPUT_ROOT="$ROOT/outputs/rwku_s5e600_rank2_active"
MODEL_REVISION=0cb88a4f764b7a12671c53f0838cd831a0843b95

exec python "$SCRIPT_DIR/rwku_generated_s5e_rank2_active_repair.py" \
  --stage "$STAGE" \
  --experiment-id "$EXPERIMENT_ID" \
  --seed 0 \
  --model-path "$MODEL_PATH" \
  --model-revision "$MODEL_REVISION" \
  --generated-training-bundle "$CORPUS/generated_training_bundle.json" \
  --generator-receipt "$CORPUS/generator_receipt.json" \
  --output-root "$OUTPUT_ROOT" \
  --mcf-path "$ROOT/data/multi_counterfact.json" \
  --protection-source "$ROOT/data/multi_counterfact.json" \
  --data-root "$ROOT/data/rwku" \
  --wikidata-dir "$ROOT/data/wikidata" \
  --dtype bf16 \
  --eval-batch-size "${EVAL_BATCH_SIZE:-4}" \
  --gradient-checkpointing \
  --no-download
