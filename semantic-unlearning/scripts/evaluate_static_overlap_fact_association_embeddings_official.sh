#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_DIR="${1:?usage: $0 RUN_DIR [MCF_PATH] [WIKIDATA_DIR]}"
MCF_PATH="${2:-$ROOT/data/multi_counterfact.json}"
WIKIDATA_DIR="${3:-$ROOT/data/wikidata}"

python -u scripts/evaluate_static_overlap_fact_association_embeddings_official.py \
  --run-dir "$RUN_DIR" \
  --mcf-path "$MCF_PATH" \
  --wikidata-dir "$WIKIDATA_DIR" \
  --device cuda \
  --dtype bfloat16 \
  --local-files-only
