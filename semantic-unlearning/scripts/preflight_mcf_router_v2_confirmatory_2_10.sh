#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

REF="$ROOT/outputs/mcf_fact_assoc_router_v2_seed1"
test -f "$REF/association_manifest.json" || {
  echo "Missing development manifest: $REF/association_manifest.json" >&2
  exit 2
}
MODEL_PATH="$(jq -r '.model_path' "$REF/association_manifest.json")"
MCF_PATH="$(jq -r '.mcf_path' "$REF/association_manifest.json")"

for SEED in 2 3 4 5 6 7 8 9 10; do
  OUT="$ROOT/outputs/mcf_fact_assoc_router_v2_seed${SEED}_preflight"
  test ! -e "$OUT" || {
    echo "Refusing to overwrite preflight: $OUT" >&2
    exit 2
  }
  echo "===== PREFLIGHT MCF Router V2 confirmatory seed ${SEED} ====="
  python -u scripts/run_mcf_fact_association_router_v2_confirmatory.py \
    --model-path "$MODEL_PATH" \
    --mcf-path "$MCF_PATH" \
    --output-dir "$OUT" \
    --device cuda \
    --local-files-only \
    --forget-num 50 \
    --seed "$SEED" \
    --preflight-only
done

echo "===== MCF Router V2 confirmatory preflights 2--10 complete ====="
