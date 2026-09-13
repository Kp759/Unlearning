#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_DIR="${1:?usage: $0 RUN_DIR RWKU_DATA_ROOT WIKIDATA_DIR [DTYPE]}"
RWKU_DATA_ROOT="${2:?usage: $0 RUN_DIR RWKU_DATA_ROOT WIKIDATA_DIR [DTYPE]}"
WIKIDATA_DIR="${3:?usage: $0 RUN_DIR RWKU_DATA_ROOT WIKIDATA_DIR [DTYPE]}"
DTYPE="${4:-bfloat16}"

python -u scripts/evaluate_rwku_fact_association_embeddings_seed1.py \
  --run-dir "$RUN_DIR" \
  --data-root "$RWKU_DATA_ROOT" \
  --wikidata-dir "$WIKIDATA_DIR" \
  --dtype "$DTYPE" \
  --device cuda \
  --local-files-only \
  --no-download
