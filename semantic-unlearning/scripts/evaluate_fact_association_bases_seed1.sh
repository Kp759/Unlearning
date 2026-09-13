#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MCF_RUN="${1:?usage: $0 MCF_RUN ZSRE_RUN MQUAKE_RUN ZSRE_PATH MQUAKE_PATH [MCF_PATH] [WIKIDATA_DIR]}"
ZSRE_RUN="${2:?usage: $0 MCF_RUN ZSRE_RUN MQUAKE_RUN ZSRE_PATH MQUAKE_PATH [MCF_PATH] [WIKIDATA_DIR]}"
MQUAKE_RUN="${3:?usage: $0 MCF_RUN ZSRE_RUN MQUAKE_RUN ZSRE_PATH MQUAKE_PATH [MCF_PATH] [WIKIDATA_DIR]}"
ZSRE_PATH="${4:?usage: $0 MCF_RUN ZSRE_RUN MQUAKE_RUN ZSRE_PATH MQUAKE_PATH [MCF_PATH] [WIKIDATA_DIR]}"
MQUAKE_PATH="${5:?usage: $0 MCF_RUN ZSRE_RUN MQUAKE_RUN ZSRE_PATH MQUAKE_PATH [MCF_PATH] [WIKIDATA_DIR]}"
MCF_PATH="${6:-$ROOT/data/multi_counterfact.json}"
WIKIDATA_DIR="${7:-$ROOT/data/wikidata}"

OUT_DIR="$ROOT/outputs/fact_association_seed1_frozen_base"
mkdir -p "$OUT_DIR"

python -u scripts/evaluate_mcf_fact_association_base_seed1.py   --reference-run-dir "$MCF_RUN"   --mcf-path "$MCF_PATH"   --wikidata-dir "$WIKIDATA_DIR"   --device cuda   --dtype bfloat16   --local-files-only   --out "$OUT_DIR/mcf_base_seed1.json"

python -u scripts/evaluate_zsre_fact_association_base_seed1.py   --reference-run-dir "$ZSRE_RUN"   --zsre-path "$ZSRE_PATH"   --wikidata-dir "$WIKIDATA_DIR"   --device cuda   --dtype bfloat16   --batch-size 8   --local-files-only   --out "$OUT_DIR/zsre_base_seed1.json"

python -u scripts/evaluate_mquake_fact_association_base_seed1.py   --reference-run-dir "$MQUAKE_RUN"   --mquake-path "$MQUAKE_PATH"   --wikidata-dir "$WIKIDATA_DIR"   --device cuda   --dtype bfloat16   --batch-size 8   --local-files-only   --out "$OUT_DIR/mquake_base_seed1.json"

echo "Base results written to $OUT_DIR"
