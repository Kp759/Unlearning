#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MCF_REF="$ROOT/outputs/static_overlap_fact_association_embeddings_v1_hierarchical_seed1"
ZSRE_REF="$ROOT/outputs/zsre_fact_assoc_seed1_exacttoken"
MQUAKE_REF="$ROOT/outputs/mquake_fact_assoc_seed1_uniqueassoc_train"
RWKU_REF="$ROOT/outputs/rwku_fact_assoc_seed1_train_v1"

for ref in "$MCF_REF" "$ZSRE_REF" "$MQUAKE_REF" "$RWKU_REF"; do
  test -f "$ref/association_manifest.json" || {
    echo "Missing reference manifest: $ref/association_manifest.json" >&2
    exit 2
  }
done

MODEL_PATH="$(jq -r '.model_path' "$ZSRE_REF/association_manifest.json")"
for ref in "$MCF_REF" "$MQUAKE_REF" "$RWKU_REF"; do
  current="$(jq -r '.model_path' "$ref/association_manifest.json")"
  test "$current" = "$MODEL_PATH" || {
    echo "Reference model mismatch: $ref -> $current; expected $MODEL_PATH" >&2
    exit 2
  }
done

MCF_PATH="$(jq -r '.mcf_path' "$MCF_REF/association_manifest.json")"
ZSRE_VISIBLE="$(jq -r '.training_visible_path' "$ZSRE_REF/association_manifest.json")"
ZSRE_SPLIT="$(jq -r '.split_manifest_path' "$ZSRE_REF/association_manifest.json")"
MQUAKE_VISIBLE="$(jq -r '.training_visible_path' "$MQUAKE_REF/association_manifest.json")"
MQUAKE_SPLIT="$(jq -r '.split_manifest_path' "$MQUAKE_REF/association_manifest.json")"
RWKU_DATA="$(jq -r '.data_root' "$RWKU_REF/association_manifest.json")"
RWKU_SPLIT="$(jq -r '.split_dir' "$RWKU_REF/association_manifest.json")"

PREFLIGHT_ROOT="$ROOT/outputs/fact_assoc_router_v2_seed1_preflight"
mkdir -p "$PREFLIGHT_ROOT"

for child in mcf zsre mquake rwku; do
  test ! -e "$PREFLIGHT_ROOT/$child" || {
    echo "Refusing to overwrite preflight directory: $PREFLIGHT_ROOT/$child" >&2
    exit 2
  }
done

echo "===== MCF Router V2 preflight ====="
python -u scripts/run_mcf_fact_association_router_v2_seed1.py   --model-path "$MODEL_PATH"   --mcf-path "$MCF_PATH"   --output-dir "$PREFLIGHT_ROOT/mcf"   --device cuda   --local-files-only   --preflight-only

echo "===== ZsRE Router V2 preflight ====="
python -u scripts/run_zsre_fact_association_router_v2_seed1.py   --model-path "$MODEL_PATH"   --training-visible "$ZSRE_VISIBLE"   --split-manifest "$ZSRE_SPLIT"   --output-dir "$PREFLIGHT_ROOT/zsre"   --device cuda   --local-files-only   --seed 1   --forget-num 50   --preflight-only

echo "===== MQuAKE Router V2 preflight ====="
python -u scripts/run_mquake_fact_association_router_v2_seed1.py   --model-path "$MODEL_PATH"   --training-visible "$MQUAKE_VISIBLE"   --split-manifest "$MQUAKE_SPLIT"   --output-dir "$PREFLIGHT_ROOT/mquake"   --device cuda   --local-files-only   --seed 1   --forget-num 50   --retain-num 1000   --preflight-only

echo "===== RWKU Router V2 preflight ====="
python -u scripts/run_rwku_fact_association_router_v2_seed1.py   --model-path "$MODEL_PATH"   --data-root "$RWKU_DATA"   --split-dir "$RWKU_SPLIT"   --output-dir "$PREFLIGHT_ROOT/rwku"   --device cuda   --local-files-only   --no-download   --seed 1   --preflight-only

echo "===== Router V2 seed-1 preflight complete ====="
echo "Outputs: $PREFLIGHT_ROOT"
