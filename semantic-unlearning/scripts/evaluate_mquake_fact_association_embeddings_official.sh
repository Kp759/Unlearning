#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_DIR="${1:?usage: $0 RUN_DIR MQUAKE_PATH WIKIDATA_DIR [DTYPE]}"
MQUAKE_PATH="${2:?usage: $0 RUN_DIR MQUAKE_PATH WIKIDATA_DIR [DTYPE]}"
WIKIDATA_DIR="${3:?usage: $0 RUN_DIR MQUAKE_PATH WIKIDATA_DIR [DTYPE]}"
DTYPE="${4:-bfloat16}"

python -u scripts/evaluate_mquake_fact_association_embeddings_official.py \
  --run-dir "$RUN_DIR" \
  --mquake-path "$MQUAKE_PATH" \
  --wikidata-dir "$WIKIDATA_DIR" \
  --dtype "$DTYPE" \
  --device cuda \
  --batch-size 8 \
  --local-files-only
