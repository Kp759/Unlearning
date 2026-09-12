#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_DIR="${1:?usage: $0 RUN_DIR ZSRE_PATH WIKIDATA_DIR [DTYPE]}"
ZSRE_PATH="${2:?usage: $0 RUN_DIR ZSRE_PATH WIKIDATA_DIR [DTYPE]}"
WIKIDATA_DIR="${3:?usage: $0 RUN_DIR ZSRE_PATH WIKIDATA_DIR [DTYPE]}"
DTYPE="${4:-bfloat16}"

python -u scripts/evaluate_zsre_fact_association_embeddings_official.py \
  --run-dir "$RUN_DIR" \
  --zsre-path "$ZSRE_PATH" \
  --wikidata-dir "$WIKIDATA_DIR" \
  --dtype "$DTYPE" \
  --device cuda \
  --batch-size 8 \
  --local-files-only
