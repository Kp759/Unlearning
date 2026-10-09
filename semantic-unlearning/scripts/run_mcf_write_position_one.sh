#!/usr/bin/env bash
# One (arm, seed) of the MCF write-position experiment at layer 19.
#
#   bash scripts/run_mcf_write_position_one.sh ARM SEED OUT_DIR
#
# ARM: A last | B last_subject | C subject_span | D all_prompt
# Stages (each skipped once its output exists, so a stopped run resumes):
#   1. router   the seed's L19 linear classifier from the multiseed regular run
#               (outputs/mcf_multiseed_regular_v1/seed<S>/L19/router); for seed 1
#               also the exploratory sweep router; otherwise prep + fit one.
#   2. arm      copy it and set write_mode                         -> OUT_DIR/router
#   3. rows     train with the multiseed settings (router route,
#               norm-scale 1, 10800 s cap so the step budget binds) -> OUT_DIR/linear_global
#   4. eval     official MCF eval, bf16, --seed SEED
#
# Env overrides (mainly for testing): MODEL_PATH, MCF_PATH, ROUTER_SRC,
# LAYER, MAX_TRAIN_SECONDS, DEVICE, EVAL_DTYPE, EXTRA_TRAIN_ARGS, EXTRA_PREP_ARGS.
set -euo pipefail

ARM="${1:?Usage: run_mcf_write_position_one.sh ARM SEED OUT_DIR}"
SEED="${2:?Usage: run_mcf_write_position_one.sh ARM SEED OUT_DIR}"
OUT_DIR="${3:?Usage: run_mcf_write_position_one.sh ARM SEED OUT_DIR}"
case "$ARM" in A|B|C|D) ;; *) echo "ARM must be A, B, C or D, got '$ARM'" >&2; exit 2;; esac
LAYER="${LAYER:-19}"
MAX_TRAIN_SECONDS="${MAX_TRAIN_SECONDS:-10800}"
DEVICE="${DEVICE:-cuda}"
EVAL_DTYPE="${EVAL_DTYPE:-bfloat16}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -z "${MODEL_PATH:-}" || -z "${MCF_PATH:-}" ]]; then
  REF="$ROOT/outputs/mcf_fact_assoc_router_v2_seed1/association_manifest.json"
  test -f "$REF" || { echo "Missing $REF (model/data paths)" >&2; exit 2; }
  MODEL_PATH="${MODEL_PATH:-$(jq -r '.model_path' "$REF")}"
  MCF_PATH="${MCF_PATH:-$(jq -r '.mcf_path' "$REF")}"
fi
mkdir -p "$OUT_DIR"
echo "===== write-position arm $ARM | seed $SEED | $OUT_DIR | $(date) ====="

# 1. Router for this seed (fitted, rows still zero).
if [[ -z "${ROUTER_SRC:-}" ]]; then
  CANDIDATES=("$ROOT/outputs/mcf_multiseed_regular_v1/seed$SEED/L19/router")
  [[ "$SEED" == "1" ]] && CANDIDATES+=("$ROOT/outputs/mcf_layer_sweep_linear_regular_v1/L19/router")
  for CANDIDATE in "${CANDIDATES[@]}"; do
    if [[ -f "$CANDIDATE/fact_association_embeddings.pt" ]]; then ROUTER_SRC="$CANDIDATE"; break; fi
  done
fi
if [[ -z "${ROUTER_SRC:-}" ]]; then
  echo "===== [$ARM s$SEED] no L19 router for seed $SEED: fitting one (prep + router) ====="
  if [[ ! -f "$OUT_DIR/prep/fact_association_embeddings.pt" ]]; then
    rm -rf "$OUT_DIR/prep"
    python -u scripts/prepare_mcf_association_source.py \
      --model-path "$MODEL_PATH" --mcf-path "$MCF_PATH" \
      --output-dir "$OUT_DIR/prep" --layer "$LAYER" --seed "$SEED" \
      --device "$DEVICE" --local-files-only ${EXTRA_PREP_ARGS:-}
  fi
  if [[ ! -f "$OUT_DIR/router_fit/fact_association_embeddings.pt" ]]; then
    rm -rf "$OUT_DIR/router_fit"
    python -u scripts/fit_linear_router.py \
      --run-dir "$OUT_DIR/prep" --output-dir "$OUT_DIR/router_fit" \
      --device "$DEVICE" --local-files-only \
      --threshold-policy global --min-recall 0.98 --threshold-placement-fraction 0.1
  fi
  ROUTER_SRC="$OUT_DIR/router_fit"
fi
echo "router: $ROUTER_SRC"

# 2. Arm router = same classifier, write_mode set.
if [[ ! -f "$OUT_DIR/router/fact_association_embeddings.pt" ]]; then
  rm -rf "$OUT_DIR/router"
  python -u scripts/mcf_write_position.py make-arm \
    --router-dir "$ROUTER_SRC" --output-dir "$OUT_DIR/router" --arm "$ARM"
fi

# 3. Train rows under this write mode.
FINAL="$OUT_DIR/linear_global"
if [[ ! -f "$FINAL/training_report.json" ]]; then
  if [[ -e "$FINAL" ]]; then mv "$FINAL" "$FINAL.incomplete_$(date +%Y%m%d_%H%M%S)"; fi
  echo "===== [$ARM s$SEED] TRAIN rows (route=router, norm-scale=1, cap ${MAX_TRAIN_SECONDS}s) | $(date) ====="
  python -u scripts/train_mcf_linear_router_rows.py \
    --router-dir "$OUT_DIR/router" --output-dir "$FINAL" \
    --training-route router --norm-scale 1 \
    --max-training-seconds "$MAX_TRAIN_SECONDS" \
    --device "$DEVICE" --local-files-only ${EXTRA_TRAIN_ARGS:-}
fi

# 4. Official MCF eval.
if [[ ! -f "$FINAL/official_mcf_eval.json" ]]; then
  echo "===== [$ARM s$SEED] OFFICIAL MCF eval | $(date) ====="
  python -u scripts/evaluate_static_overlap_fact_association_embeddings_official.py \
    --run-dir "$FINAL" --mcf-path "$MCF_PATH" \
    --wikidata-dir "$ROOT/data/wikidata" --seed "$SEED" \
    --device "$DEVICE" --dtype "$EVAL_DTYPE" --local-files-only
fi
echo "===== [$ARM s$SEED] COMPLETE -> $FINAL | $(date) ====="
