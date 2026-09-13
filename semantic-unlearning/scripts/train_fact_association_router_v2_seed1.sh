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

MCF_OUT="$ROOT/outputs/mcf_fact_assoc_router_v2_seed1"
ZSRE_OUT="$ROOT/outputs/zsre_fact_assoc_router_v2_seed1"
MQUAKE_OUT="$ROOT/outputs/mquake_fact_assoc_router_v2_seed1"
RWKU_OUT="$ROOT/outputs/rwku_fact_assoc_router_v2_seed1_direct"

for out in "$MCF_OUT" "$ZSRE_OUT" "$MQUAKE_OUT" "$RWKU_OUT"; do
  test ! -e "$out" || {
    echo "Refusing to overwrite existing Router V2 run: $out" >&2
    exit 2
  }
done

echo "===== TRAIN MCF Router V2 seed 1 ====="
python -u scripts/run_mcf_fact_association_router_v2_seed1.py   --model-path "$MODEL_PATH"   --mcf-path "$MCF_PATH"   --output-dir "$MCF_OUT"   --device cuda   --local-files-only   --forget-num 50   --seed 1

echo "===== TRAIN ZsRE Router V2 seed 1 ====="
python -u scripts/run_zsre_fact_association_router_v2_seed1.py   --model-path "$MODEL_PATH"   --training-visible "$ZSRE_VISIBLE"   --split-manifest "$ZSRE_SPLIT"   --output-dir "$ZSRE_OUT"   --device cuda   --local-files-only   --seed 1   --forget-num 50   --steps 1500   --max-training-seconds 3600

echo "===== TRAIN MQuAKE Router V2 seed 1 ====="
python -u scripts/run_mquake_fact_association_router_v2_seed1.py   --model-path "$MODEL_PATH"   --training-visible "$MQUAKE_VISIBLE"   --split-manifest "$MQUAKE_SPLIT"   --output-dir "$MQUAKE_OUT"   --device cuda   --local-files-only   --seed 1   --forget-num 50   --retain-num 1000   --row-updates-per-fact 30   --max-training-seconds 7200

echo "===== TRAIN RWKU Router V2 seed 1 (direct stage) ====="
python -u scripts/run_rwku_fact_association_router_v2_seed1.py   --model-path "$MODEL_PATH"   --data-root "$RWKU_DATA"   --split-dir "$RWKU_SPLIT"   --output-dir "$RWKU_OUT"   --device cuda   --local-files-only   --no-download   --seed 1   --row-updates-per-fact 30   --max-training-seconds 7200

echo "===== Router V2 seed-1 direct training complete ====="
printf 'MCF_OUT=%s\n' "$MCF_OUT"
printf 'ZSRE_OUT=%s\n' "$ZSRE_OUT"
printf 'MQUAKE_OUT=%s\n' "$MQUAKE_OUT"
printf 'RWKU_OUT=%s\n' "$RWKU_OUT"
