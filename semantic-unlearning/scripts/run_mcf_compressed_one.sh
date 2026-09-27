#!/usr/bin/env bash
# One compressed-bank configuration on MCF (linear router, no V2).
#
#   bash scripts/run_mcf_compressed_one.sh HEAD_SHARING VALUE_MODE
#     HEAD_SHARING : fact (one router head per fact) | relation (one per relation)
#     VALUE_MODE   : full | lowrank:K | tied_answer | tied_relation |
#                    answer_fixed | answer_map:r | relation_plus_answer
#
# Stages (each skipped once done; half-written stages are moved aside):
#   1. prep    MCF seed-1 data + zero rows at LAYER            (one per HEAD_SHARING)
#   2. router  linear router at LAYER, frozen MCF policy       (one per HEAD_SHARING)
#   3. values  compressed bank trained in the loop            (one per config)
#   4. official MCF eval
#
# Env: LAYER (default 19), TRAINING_ROUTE (router|oracle, default router),
# TAG (default mcf_compressed_v1), EXTRA_TRAIN_ARGS, MOVE_INCOMPLETE (default 1).
set -euo pipefail

HEAD_SHARING="${1:?Usage: run_mcf_compressed_one.sh HEAD_SHARING VALUE_MODE}"
VALUE_MODE="${2:?Usage: run_mcf_compressed_one.sh HEAD_SHARING VALUE_MODE}"
LAYER="${LAYER:-19}"
TRAINING_ROUTE="${TRAINING_ROUTE:-router}"
TAG="${TAG:-mcf_compressed_v1}"
EXTRA_TRAIN_ARGS="${EXTRA_TRAIN_ARGS:-}"
MOVE_INCOMPLETE="${MOVE_INCOMPLETE:-1}"
case "$HEAD_SHARING" in fact|relation) ;; *) echo "HEAD_SHARING must be fact or relation" >&2; exit 2;; esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
REF="$ROOT/outputs/mcf_fact_assoc_router_v2_seed1"
test -f "$REF/association_manifest.json" || { echo "Missing $REF/association_manifest.json" >&2; exit 2; }
MODEL_PATH="$(jq -r '.model_path' "$REF/association_manifest.json")"
MCF_PATH="$(jq -r '.mcf_path' "$REF/association_manifest.json")"

BASE="$ROOT/outputs/$TAG/L$(printf '%02d' "$LAYER")"
# One prep per router type so the two SLURM jobs never share a stage dir.
PREP="$BASE/prep_$HEAD_SHARING"
ROUTER="$BASE/router_$HEAD_SHARING"
FINAL="$ROUTER/${TRAINING_ROUTE}_${VALUE_MODE//:/_}"
mkdir -p "$BASE"

stage_ready() {  # dir, marker
  if [[ -f "$1/$2" ]]; then return 0; fi
  if [[ -e "$1" ]]; then
    if [[ "$MOVE_INCOMPLETE" == "1" ]]; then
      local aside="$1.incomplete_$(date +%Y%m%d_%H%M%S)"
      echo "Moving incomplete stage dir aside: $1 -> $aside" >&2
      mv "$1" "$aside"
    else
      echo "Incomplete stage dir (move it aside first): $1" >&2; exit 2
    fi
  fi
  return 1
}

if ! stage_ready "$PREP" fact_association_embeddings.pt; then
  echo "===== [compressed L$LAYER] 1/4 PREP ====="
  python -u scripts/prepare_mcf_association_source.py \
    --model-path "$MODEL_PATH" --mcf-path "$MCF_PATH" \
    --output-dir "$PREP" --layer "$LAYER" --device cuda --local-files-only
fi

if ! stage_ready "$ROUTER" fact_association_embeddings.pt; then
  echo "===== [compressed L$LAYER] 2/4 ROUTER (head sharing: $HEAD_SHARING) ====="
  python -u scripts/fit_linear_router.py \
    --run-dir "$PREP" --output-dir "$ROUTER" --device cuda --local-files-only \
    --threshold-policy global --min-recall 0.98 --threshold-placement-fraction 0.1 \
    --head-sharing "$HEAD_SHARING"
fi

if ! stage_ready "$FINAL" fact_association_embeddings.pt; then
  echo "===== [compressed L$LAYER] 3/4 VALUES $VALUE_MODE (route=$TRAINING_ROUTE) ====="
  # shellcheck disable=SC2086
  python -u scripts/train_mcf_compressed_bank.py \
    --router-dir "$ROUTER" --output-dir "$FINAL" \
    --value-mode "$VALUE_MODE" --training-route "$TRAINING_ROUTE" \
    --device cuda --local-files-only $EXTRA_TRAIN_ARGS
fi

if [[ ! -f "$FINAL/official_mcf_eval.json" ]]; then
  echo "===== [compressed L$LAYER] 4/4 OFFICIAL MCF eval ====="
  python -u scripts/evaluate_static_overlap_fact_association_embeddings_official.py \
    --run-dir "$FINAL" --mcf-path "$MCF_PATH" --wikidata-dir "$ROOT/data/wikidata" \
    --device cuda --dtype bfloat16 --local-files-only
fi
echo "===== [compressed L$LAYER] COMPLETE -> $FINAL ====="
