#!/usr/bin/env bash
set -euo pipefail

SEED="${1:?Usage: bash scripts/run_mcf_router_v2_confirmatory_seed.sh SEED}"
if (( SEED < 2 || SEED > 10 )); then
  echo "SEED must be an integer from 2 through 10 (seed 1 is development)." >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

REF="$ROOT/outputs/mcf_fact_assoc_router_v2_seed1"
test -f "$REF/association_manifest.json" || {
  echo "Missing development manifest: $REF/association_manifest.json" >&2
  exit 2
}

MODEL_PATH="$(jq -r '.model_path' "$REF/association_manifest.json")"
MCF_PATH="$(jq -r '.mcf_path' "$REF/association_manifest.json")"
OUT="$ROOT/outputs/mcf_fact_assoc_router_v2_seed${SEED}"

test ! -e "$OUT" || {
  echo "Refusing to overwrite existing run: $OUT" >&2
  exit 2
}

echo "===== TRAIN MCF Router V2 confirmatory seed ${SEED} ====="
python -u scripts/run_mcf_fact_association_router_v2_confirmatory.py \
  --model-path "$MODEL_PATH" \
  --mcf-path "$MCF_PATH" \
  --output-dir "$OUT" \
  --device cuda \
  --local-files-only \
  --forget-num 50 \
  --seed "$SEED"

echo "===== EVAL MCF Router V2 confirmatory seed ${SEED} ====="
python -u scripts/evaluate_mcf_fact_association_router_v2_confirmatory.py \
  --run-dir "$OUT" \
  --mcf-path "$MCF_PATH" \
  --wikidata-dir "$ROOT/data/wikidata" \
  --device cuda \
  --dtype bfloat16 \
  --local-files-only

echo "===== COMPLETE MCF Router V2 confirmatory seed ${SEED} ====="
