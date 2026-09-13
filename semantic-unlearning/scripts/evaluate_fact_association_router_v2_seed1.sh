#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MCF_RUN="${1:-$ROOT/outputs/mcf_fact_assoc_router_v2_seed1}"
ZSRE_RUN="${2:-$ROOT/outputs/zsre_fact_assoc_router_v2_seed1}"
MQUAKE_RUN="${3:-$ROOT/outputs/mquake_fact_assoc_router_v2_seed1}"
RWKU_RUN="${4:-$ROOT/outputs/rwku_fact_assoc_router_v2_seed1_direct}"

MCF_PATH="$ROOT/data/multi_counterfact.json"
ZSRE_PATH="$ROOT/data/zsre_mend_eval.json"
MQUAKE_PATH="$ROOT/data/MQuAKE-CF-3k-v2.json"
RWKU_DATA="$ROOT/data/rwku"
WIKIDATA="$ROOT/data/wikidata"

for run in "$MCF_RUN" "$ZSRE_RUN" "$MQUAKE_RUN" "$RWKU_RUN"; do
  test -f "$run/fact_association_embeddings.pt" || {
    echo "Missing trained artifact: $run/fact_association_embeddings.pt" >&2
    exit 2
  }
done

echo "===== EVAL MCF Router V2 seed 1 ====="
python -u scripts/evaluate_static_overlap_fact_association_embeddings_official.py   --run-dir "$MCF_RUN"   --mcf-path "$MCF_PATH"   --wikidata-dir "$WIKIDATA"   --device cuda   --dtype bfloat16   --local-files-only

echo "===== EVAL ZsRE Router V2 seed 1 ====="
python -u scripts/evaluate_zsre_fact_association_embeddings_official.py   --run-dir "$ZSRE_RUN"   --zsre-path "$ZSRE_PATH"   --wikidata-dir "$WIKIDATA"   --device cuda   --dtype bfloat16   --batch-size 8   --local-files-only

echo "===== EVAL MQuAKE Router V2 seed 1 ====="
python -u scripts/evaluate_mquake_fact_association_embeddings_official.py   --run-dir "$MQUAKE_RUN"   --mquake-path "$MQUAKE_PATH"   --wikidata-dir "$WIKIDATA"   --device cuda   --dtype bfloat16   --batch-size 8   --local-files-only

echo "===== EVAL RWKU Router V2 seed 1 (direct stage) ====="
python -u scripts/evaluate_rwku_fact_association_embeddings_seed1.py   --run-dir "$RWKU_RUN"   --data-root "$RWKU_DATA"   --wikidata-dir "$WIKIDATA"   --device cuda   --dtype bfloat16   --local-files-only   --no-download

echo "===== Router V2 seed-1 direct evaluation complete ====="
