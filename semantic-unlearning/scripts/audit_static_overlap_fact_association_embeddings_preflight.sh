#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_DIR="${1:?usage: $0 RUN_DIR [DTYPE]}"
DTYPE="${2:-bfloat16}"

python -u scripts/audit_static_overlap_fact_association_embeddings_preflight.py \
  --run-dir "$RUN_DIR" \
  --dtype "$DTYPE" \
  --device cuda \
  --local-files-only
