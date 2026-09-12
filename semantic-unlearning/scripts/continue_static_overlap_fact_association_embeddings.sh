#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PARENT_RUN="${1:?usage: $0 PARENT_RUN PREFLIGHT_JSON OUTPUT_DIR [STEPS]}"
PREFLIGHT_JSON="${2:?usage: $0 PARENT_RUN PREFLIGHT_JSON OUTPUT_DIR [STEPS]}"
OUTPUT_DIR="${3:?usage: $0 PARENT_RUN PREFLIGHT_JSON OUTPUT_DIR [STEPS]}"
STEPS="${4:-750}"

python -u scripts/continue_static_overlap_fact_association_embeddings.py \
  --parent-run "$PARENT_RUN" \
  --preflight-path "$PREFLIGHT_JSON" \
  --output-dir "$OUTPUT_DIR" \
  --steps "$STEPS" \
  --max-training-seconds 3600 \
  --device cuda \
  --local-files-only
