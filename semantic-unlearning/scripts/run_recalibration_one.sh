#!/usr/bin/env bash
# Re-pick the router bias on the validation set (calibration + audit), weights
# and rows unchanged, then run the official eval. One dataset/seed/layer.
#
#   SEED=1 LAYER=19 bash scripts/run_recalibration_one.sh mcf
#   REF_TAG=multiseed_reworded_v2 SEED=3 LAYER=23 bash scripts/run_recalibration_one.sh zsre
#   OBJECTIVE=min_recall MACRO=fact SEED=2 LAYER=7 bash scripts/run_recalibration_one.sh rwku
#   (RWKU: threshold-gate reference runs only; the subject gate has no cutoff)
#
# Env: SEED (1), LAYER (19), REF_TAG (multiseed_regular_v1), OUT_TAG
# (calibration_rules_v1), OBJECTIVE (balanced | target_fpr | min_recall),
# MACRO (fact | prompt; default fact for balanced/constrained, prompt otherwise),
# TARGET_FPR (0.1), MIN_RECALL (0.98).
# Output: outputs/<OUT_TAG>/<dataset>/seed<S>/L<LL>/<arm>/{fact_association_embeddings.pt,
#   recalibration.json, official_<dataset>_eval.json}; arm = balanced_fact |
#   balanced_prompt | fpr<X>[_fact] | recall<X>[_fact] | constrained_r<R>_f<F>_<macro>. Resumable.
set -euo pipefail
DATASET="${1:?Usage: bash scripts/run_recalibration_one.sh mcf|zsre|mquake|rwku}"
case "$DATASET" in mcf|zsre|mquake|rwku) ;; *) echo "unknown dataset '$DATASET'" >&2; exit 2;; esac
SEED="${SEED:-1}"; LAYER="${LAYER:-19}"
REF_TAG="${REF_TAG:-multiseed_regular_v1}"; OUT_TAG="${OUT_TAG:-calibration_rules_v1}"
OBJECTIVE="${OBJECTIVE:-balanced}"
case "$OBJECTIVE" in balanced|constrained) MACRO="${MACRO:-fact}" ;; *) MACRO="${MACRO:-prompt}" ;; esac
TARGET_FPR="${TARGET_FPR:-0.1}"; MIN_RECALL="${MIN_RECALL:-0.98}"
case "$OBJECTIVE" in
  balanced)   ARM="balanced_${MACRO}" ;;
  target_fpr) ARM="fpr${TARGET_FPR}"; [[ "$MACRO" == fact ]] && ARM="${ARM}_fact" ;;
  min_recall) ARM="recall${MIN_RECALL}"; [[ "$MACRO" == fact ]] && ARM="${ARM}_fact" ;;
  constrained) ARM="constrained_r${MIN_RECALL}_f${TARGET_FPR}_${MACRO}" ;;
  *) echo "OBJECTIVE must be balanced, target_fpr, min_recall or constrained" >&2; exit 2 ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
LL="$(printf '%02d' "$LAYER")"
REF="$ROOT/outputs/${DATASET}_${REF_TAG}/seed${SEED}/L${LL}"
OUT="$ROOT/outputs/${OUT_TAG}/${DATASET}/seed${SEED}/L${LL}/${ARM}"
EVAL_JSON="official_${DATASET}_eval.json"
for f in "$REF/router/linear_router_dataset.json" "$REF/router/fact_association_embeddings.pt" \
         "$REF/linear_global/fact_association_embeddings.pt" "$REF/linear_global/$EVAL_JSON"; do
  test -f "$f" || { echo "Reference run incomplete, missing: $f" >&2; exit 2; }
done
echo "reference $REF -> $OUT"

if [[ ! -f "$OUT/recalibration.json" ]]; then
  if [[ -e "$OUT" ]]; then mv "$OUT" "$OUT.incomplete_$(date +%Y%m%d_%H%M%S)"; fi
  mkdir -p "$(dirname "$OUT")"
  python -u scripts/recalibrate_router.py --router-dir "$REF/router" --rows-from "$REF/linear_global" \
    --output-dir "$OUT" --objective "$OBJECTIVE" --macro "$MACRO" \
    --target-fpr "$TARGET_FPR" --min-recall "$MIN_RECALL" --device cuda --local-files-only
fi

if [[ ! -f "$OUT/$EVAL_JSON" ]]; then
  case "$DATASET" in
    mcf)
      MCF_PATH="$(jq -r '.mcf_path' "$REF/router/association_manifest.json")"
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
    rwku)
      python -u scripts/evaluate_rwku_fact_association_embeddings_seed1.py \
        --run-dir "$OUT" --data-root "$ROOT/data/rwku" --wikidata-dir "$ROOT/data/wikidata" \
        --seed "$SEED" --device cuda --dtype bfloat16 --local-files-only --no-download \
        --allow-imperfect-direct-routing --out "$OUT/$EVAL_JSON" ;;
  esac
fi
echo "===== [$DATASET s$SEED L$LL $ARM] done -> $OUT ====="
