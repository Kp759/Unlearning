#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: $0 MODEL_PATH CORPUS_DIR WIKIPEDIA_DIR SOURCE_HEAD_ONLY_RUN" >&2
  exit 2
fi

MODEL_PATH=$1
CORPUS_DIR=$2
WIKIPEDIA_DIR=$3
SOURCE_HEAD_ONLY_RUN=$4

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUTPUT_ROOT=${RWKU_EMB_HEAD_ABLATION_OUTPUT_ROOT:-/home/ec2-user/workspace/Unlearning/semantic-unlearning/outputs/rwku_emb_head_ablation_v1}
UTILITY_CACHE=${RWKU_H_W1K_UTILITY_CACHE:?RWKU_H_W1K_UTILITY_CACHE must point to the frozen RWKU Wikipedia utility cache}
CONFIG=${RWKU_EMB_HEAD_ABLATION_CONFIG:-$PROJECT_ROOT/config/rwku/sure_emb_head_ablation_v1_seed0.json}

TRAINING_BUNDLE="$CORPUS_DIR/generated_training_bundle.json"
GENERATOR_RECEIPT="$CORPUS_DIR/generator_receipt.json"
if [[ ! -f "$TRAINING_BUNDLE" ]]; then
  TRAINING_BUNDLE=$(find "$CORPUS_DIR" -maxdepth 2 -type f \( -name 'generated_training_bundle.json' -o -name 'training_bundle.json' \) | head -1 || true)
fi
if [[ ! -f "$GENERATOR_RECEIPT" ]]; then
  GENERATOR_RECEIPT=$(find "$CORPUS_DIR" -maxdepth 2 -type f -name 'generator_receipt.json' | head -1 || true)
fi
[[ -f "$TRAINING_BUNDLE" ]] || { echo "Missing generated_training_bundle.json under $CORPUS_DIR" >&2; exit 2; }
[[ -f "$GENERATOR_RECEIPT" ]] || { echo "Missing generator_receipt.json under $CORPUS_DIR" >&2; exit 2; }
[[ -d "$MODEL_PATH" ]] || { echo "Missing model directory: $MODEL_PATH" >&2; exit 2; }
[[ -d "$WIKIPEDIA_DIR" ]] || { echo "Missing Wikipedia directory: $WIKIPEDIA_DIR" >&2; exit 2; }
[[ -d "$SOURCE_HEAD_ONLY_RUN" ]] || { echo "Missing source run: $SOURCE_HEAD_ONLY_RUN" >&2; exit 2; }
[[ -f "$SOURCE_HEAD_ONLY_RUN/sure_head_only_w1k/stage1_delta.pt" ]] || { echo "Missing source Stage-1 delta" >&2; exit 2; }
[[ -f "$UTILITY_CACHE" ]] || { echo "Missing utility cache: $UTILITY_CACHE" >&2; exit 2; }
[[ -f "$CONFIG" ]] || { echo "Missing configuration: $CONFIG" >&2; exit 2; }

mkdir -p "$OUTPUT_ROOT"

run_variant() {
  local variant=$1
  local log="$OUTPUT_ROOT/${variant}.log"
  echo "============================================================"
  echo "RWKU ablation: $variant"
  echo "output root: $OUTPUT_ROOT"
  echo "============================================================"
  set +e
  python "$PROJECT_ROOT/scripts/rwku_sure_emb_head_ablation_v1.py" \
    --variant "$variant" \
    --model-path "$MODEL_PATH" \
    --training-bundle "$TRAINING_BUNDLE" \
    --generator-receipt "$GENERATOR_RECEIPT" \
    --utility-cache "$UTILITY_CACHE" \
    --wikipedia-dir "$WIKIPEDIA_DIR" \
    --source-head-only-run "$SOURCE_HEAD_ONLY_RUN" \
    --output-root "$OUTPUT_ROOT" \
    --configuration "$CONFIG" \
    2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}
  set -e
  echo "$variant rc=$rc"
  return "$rc"
}

rc_a=0
rc_b=0
run_variant emb_head || rc_a=$?
run_variant emb_head_downproj || rc_b=$?

python "$PROJECT_ROOT/scripts/summarize_rwku_emb_head_ablation_v1.py" \
  --output-root "$OUTPUT_ROOT" || true

echo "emb_head rc=$rc_a"
echo "emb_head_downproj rc=$rc_b"
if [[ $rc_a -ne 0 || $rc_b -ne 0 ]]; then
  exit 1
fi
