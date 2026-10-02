#!/usr/bin/env bash
# Shared-vector (compressed) banks on the multiseed sweep's own routers, one task:
#
#   bash scripts/run_compressed_multiseed_one.sh DATASET SEED VALUE_MODE
#     DATASET     mcf | zsre | mquake
#     VALUE_MODE  full | tied_answer | answer_fixed | lowrank:K | answer_map:r ...
#
# The router is reused unchanged from outputs/<dataset>_<REF_TAG>/seed<S>/L<LL>/router
# (the router behind the shipped multiseed rows), so only the values differ:
#   full          one vector per fact, trained by the same joint trainer (reference)
#   tied_answer   facts with the same answer share ONE vector (+ a scalar each)
#   answer_fixed  no stored vector: one scalar on the answer token's direction
# Shipped row-wise rows for comparison: .../seed<S>/L<LL>/linear_global.
#
# Output: outputs/<OUT_TAG>/<dataset>/seed<S>/L<LL>/<mode>/{fact_association_embeddings.pt,
#   training_report.json, official_<dataset>_eval.json}. Resumable.
# Env: LAYER (19), BATCH_FACTS (8: facts whose gradients form one optimizer step),
# OUT_TAG (compressed_multiseed_v1 for BATCH_FACTS=8, else compressed_multiseed_bf<N>_v1),
# REF_TAG (per dataset: mcf/mquake multiseed_regular_v1, zsre multiseed_reworded_v2),
# MOVE_INCOMPLETE (1).
set -euo pipefail
DATASET="${1:?Usage: run_compressed_multiseed_one.sh DATASET SEED VALUE_MODE}"
SEED="${2:?seed}"
MODE="${3:?value mode}"
case "$DATASET" in mcf|zsre|mquake) ;; *) echo "DATASET must be mcf, zsre or mquake" >&2; exit 2;; esac
LAYER="${LAYER:-19}"
BATCH_FACTS="${BATCH_FACTS:-8}"
if [[ "$BATCH_FACTS" == 8 ]]; then OUT_TAG="${OUT_TAG:-compressed_multiseed_v1}"
else OUT_TAG="${OUT_TAG:-compressed_multiseed_bf${BATCH_FACTS}_v1}"; fi
MOVE_INCOMPLETE="${MOVE_INCOMPLETE:-1}"
case "$DATASET" in
  zsre) REF_TAG="${REF_TAG:-multiseed_reworded_v2}" ;;
  *)    REF_TAG="${REF_TAG:-multiseed_regular_v1}" ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
LL="$(printf '%02d' "$LAYER")"
REF="$ROOT/outputs/${DATASET}_${REF_TAG}/seed${SEED}/L${LL}"
ROUTER="$REF/router"
OUT="$ROOT/outputs/${OUT_TAG}/${DATASET}/seed${SEED}/L${LL}/${MODE//:/_}"
EVAL_JSON="official_${DATASET}_eval.json"
for f in "$ROUTER/fact_association_embeddings.pt" "$ROUTER/association_manifest.json"; do
  test -f "$f" || { echo "Missing router from the multiseed sweep: $f" >&2; exit 2; }
done
echo "[$DATASET s$SEED L$LL $MODE batch=$BATCH_FACTS] router $ROUTER -> $OUT"

if [[ ! -f "$OUT/training_report.json" ]]; then
  if [[ -e "$OUT" ]]; then
    if [[ "$MOVE_INCOMPLETE" == "1" ]]; then mv "$OUT" "$OUT.incomplete_$(date +%Y%m%d_%H%M%S)"
    else echo "Incomplete output dir: $OUT" >&2; exit 2; fi
  fi
  mkdir -p "$(dirname "$OUT")"
  echo "===== [$DATASET s$SEED L$LL $MODE] 1/2 TRAIN compressed values in the loop ====="
  if [[ "$DATASET" == mcf ]]; then
    python -u scripts/train_mcf_compressed_bank.py --router-dir "$ROUTER" --output-dir "$OUT" \
      --value-mode "$MODE" --training-route router --batch-facts "$BATCH_FACTS" \
      --device cuda --local-files-only
  else
    python -u scripts/train_direct_compressed_bank.py --dataset "$DATASET" --router-dir "$ROUTER" \
      --output-dir "$OUT" --value-mode "$MODE" --training-route router --batch-facts "$BATCH_FACTS" \
      --device cuda --local-files-only
  fi
fi

if [[ ! -f "$OUT/$EVAL_JSON" ]]; then
  echo "===== [$DATASET s$SEED L$LL $MODE] 2/2 OFFICIAL $DATASET eval ====="
  case "$DATASET" in
    mcf)
      MCF_PATH="$(jq -r '.mcf_path' "$ROUTER/association_manifest.json")"
      python -u scripts/evaluate_static_overlap_fact_association_embeddings_official.py \
        --run-dir "$OUT" --mcf-path "$MCF_PATH" --wikidata-dir "$ROOT/data/wikidata" \
        --seed "$SEED" --device cuda --dtype bfloat16 --local-files-only ;;
    zsre)
      python -u scripts/evaluate_zsre_fact_association_embeddings_official.py \
        --run-dir "$OUT" --zsre-path "$ROOT/data/zsre_mend_eval.json" \
        --wikidata-dir "$ROOT/data/wikidata" --seed "$SEED" --device cuda \
        --dtype bfloat16 --batch-size 8 --local-files-only ;;
    mquake)
      python -u scripts/evaluate_mquake_fact_association_embeddings_official.py \
        --run-dir "$OUT" --mquake-path "$ROOT/data/MQuAKE-CF-3k-v2.json" \
        --wikidata-dir "$ROOT/data/wikidata" --seed "$SEED" --device cuda \
        --dtype bfloat16 --batch-size 8 --local-files-only --allow-imperfect-direct-routing ;;
  esac
fi
echo "===== [$DATASET s$SEED L$LL $MODE] COMPLETE -> $OUT ====="
