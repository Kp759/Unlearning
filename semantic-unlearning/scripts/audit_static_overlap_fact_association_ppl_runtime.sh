#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
RUN_DIR="${1:?usage: $0 RUN_DIR WIKIDATA_DIR [DTYPE]}"
WIKIDATA_DIR="${2:?usage: $0 RUN_DIR WIKIDATA_DIR [DTYPE]}"
DTYPE="${3:-bfloat16}"
python -u scripts/audit_static_overlap_fact_association_ppl_runtime.py \
  --run-dir "$RUN_DIR" \
  --wikidata-dir "$WIKIDATA_DIR" \
  --dtype "$DTYPE" \
  --device cuda \
  --local-files-only
