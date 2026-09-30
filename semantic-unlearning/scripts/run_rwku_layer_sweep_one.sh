#!/usr/bin/env bash
# One layer of the RWKU layer-wise study, linear classifier only (no Router V2).
#
#   SEED=2 GATE=threshold bash scripts/run_rwku_layer_sweep_one.sh LAYER
#
# Stages (each skipped once done; half-written stages are moved aside):
#   1. prep    RWKU-Batch-50-v1 forget associations of batch SEED (people
#              SEED..SEED+4 of RWKU's first ten) + untrained rows at LAYER
#   2. router  linear classifier at LAYER
#                GATE=threshold  the current rule of the MCF/ZsRE/MQuAKE sweeps:
#                                global cutoff at recall >= MIN_RECALL (0.98) on
#                                the calibration split, folded into the bias
#                GATE=subject    RWKU's native entity-level gate: every prompt
#                                naming a protected person fires, heads pick the row
#   3. rows    TRAINING_ROUTE=router (regular) | oracle (genie); the shipped RWKU
#              optimizer (worst sensitive-token prob < 1e-6, 30 updates/association)
#   4. official RWKU eval (same-50 Eff, held-out L1/L2 + paraphrase Gen, Level-3,
#      neighbours, PPL) -> linear_global/official_rwku_eval.json
#   5. BASE_EVAL=1: the same evaluator on the router's zero rows = the base model
#      -> outputs/rwku_multiseed_base_v1/seed<S>/official_rwku_eval.json
#   6. RECAL=1 (threshold gate, regular): the 98%-recall rule re-applied on the
#      merged validation set (calibration + audit, per-fact average) ->
#      outputs/calibration_rules_v1/rwku/seed<S>/L<LL>/recall<R>_fact
#
# Env: SEED (1), GATE (threshold), TRAINING_ROUTE (router), SWEEP_TAG (by gate/route:
# multiseed_regular_v1 | multiseed_genie_v1 | multiseed_subject_v1 |
# multiseed_subject_genie_v1), MIN_RECALL (0.98), NORM_SCALE (1),
# MAX_TRAIN_SECONDS (14400, the shipped RWKU cap), MODEL_PATH (default: from an
# earlier RWKU/MQuAKE run), BASE_EVAL (0), RECAL (1), MOVE_INCOMPLETE (1).
set -euo pipefail

LAYER="${1:?Usage: bash scripts/run_rwku_layer_sweep_one.sh LAYER}"
SEED="${SEED:-1}"
GATE="${GATE:-threshold}"
TRAINING_ROUTE="${TRAINING_ROUTE:-router}"
MIN_RECALL="${MIN_RECALL:-0.98}"
NORM_SCALE="${NORM_SCALE:-1}"
MAX_TRAIN_SECONDS="${MAX_TRAIN_SECONDS:-14400}"
BASE_EVAL="${BASE_EVAL:-0}"
RECAL="${RECAL:-1}"
MOVE_INCOMPLETE="${MOVE_INCOMPLETE:-1}"
case "$GATE" in threshold|subject) ;; *) echo "GATE must be threshold or subject, got '$GATE'" >&2; exit 2;; esac
case "$TRAINING_ROUTE" in router|oracle) ;; *)
  echo "TRAINING_ROUTE must be router or oracle, got '$TRAINING_ROUTE'" >&2; exit 2;;
esac
if [[ -z "${SWEEP_TAG:-}" ]]; then
  case "$GATE:$TRAINING_ROUTE" in
    threshold:router) SWEEP_TAG=multiseed_regular_v1 ;;
    threshold:oracle) SWEEP_TAG=multiseed_genie_v1 ;;
    subject:router)   SWEEP_TAG=multiseed_subject_v1 ;;
    subject:oracle)   SWEEP_TAG=multiseed_subject_genie_v1 ;;
  esac
fi
# Llama-3.x chat templates embed today's date; pin it so every stage of every
# task sees identical RWKU prompts (see rwku_eval.chat_prompt).
export RWKU_CHAT_DATE_STRING="${RWKU_CHAT_DATE_STRING:-26 Jul 2024}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
DATA_ROOT="$ROOT/data/rwku"

if [[ -z "${MODEL_PATH:-}" ]]; then
  for cand in "$ROOT/outputs/rwku_linear_subject_seed1_v24" "$ROOT/outputs/rwku_linear_global_seed1_v24" \
              "$ROOT/outputs/rwku_fact_assoc_router_v2_seed1_direct" "$ROOT/outputs/rwku_fact_assoc_router_v2_seed1" \
              "$ROOT/outputs/rwku_fact_assoc_seed1_train_v1" \
              "$ROOT/outputs/mquake_fact_assoc_router_v2_seed1" "$ROOT/outputs/mquake_fact_assoc_seed1_uniqueassoc_train"; do
    if [[ -f "$cand/association_manifest.json" ]]; then
      MODEL_PATH="$(jq -r '.model_path // empty' "$cand/association_manifest.json")"
      [[ -n "$MODEL_PATH" ]] && { echo "model from $cand"; break; }
    fi
  done
fi
test -n "${MODEL_PATH:-}" && test -e "$MODEL_PATH" \
  || { echo "No model path (set MODEL_PATH=/path/to/Llama-3.2-3B-Instruct)" >&2; exit 2; }
echo "MODEL_PATH=$MODEL_PATH"

ROUTER_ARGS=(--gate "$GATE")
if [[ "$GATE" == threshold ]]; then
  # Same calibration settings as the shipped linear routers; the recall target
  # is always MIN_RECALL (the "current" rule of the other sweeps).
  ROUTER_ARGS+=(--threshold-policy global --min-recall "$MIN_RECALL")
  PLACEMENT=""; MARGIN=""
  for rep in "$ROOT/outputs/rwku_linear_global_seed1_v24/linear_router_report.json" \
             "$ROOT/outputs/mquake_linear_2x2_seed1_v24/linear_router_report.json"; do
    if [[ -f "$rep" ]]; then
      PLACEMENT="$(jq -r '.router_fit.threshold_placement_fraction // empty' "$rep")"
      MARGIN="$(jq -r '.router_fit.ambiguity_margin // empty' "$rep")"
      echo "Router placement/margin from $rep"; break
    fi
  done
  ROUTER_ARGS+=(--threshold-placement-fraction "${PLACEMENT:-0.1}" --ambiguity-margin "${MARGIN:-0.5}")
fi
echo "Router settings: ${ROUTER_ARGS[*]}"

LL="$(printf '%02d' "$LAYER")"
BASE="$ROOT/outputs/rwku_${SWEEP_TAG}/seed$SEED/L$LL"
PREP="$BASE/prep"
ROUTER="$BASE/router"
FINAL="$BASE/linear_global"
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
  echo "===== [RWKU s$SEED L$LAYER $GATE] 1/4 PREP ====="
  python -u scripts/prepare_rwku_association_source.py \
    --model-path "$MODEL_PATH" --data-root "$DATA_ROOT" --seed "$SEED" \
    --output-dir "$PREP" --layer "$LAYER" --device cuda --local-files-only
fi

if ! stage_ready "$ROUTER" fact_association_embeddings.pt; then
  echo "===== [RWKU s$SEED L$LAYER $GATE] 2/4 FIT linear classifier router ====="
  python -u scripts/fit_linear_router.py \
    --run-dir "$PREP" --output-dir "$ROUTER" \
    --device cuda --local-files-only "${ROUTER_ARGS[@]}"
fi

if ! stage_ready "$FINAL" fact_association_embeddings.pt; then
  echo "===== [RWKU s$SEED L$LAYER $GATE] 3/4 TRAIN rows (route=$TRAINING_ROUTE, norm-scale=$NORM_SCALE) ====="
  python -u scripts/train_direct_linear_router_rows.py --dataset rwku \
    --router-dir "$ROUTER" --output-dir "$FINAL" \
    --training-route "$TRAINING_ROUTE" --norm-scale "$NORM_SCALE" \
    --max-training-seconds "$MAX_TRAIN_SECONDS" \
    --device cuda --local-files-only
fi

EVAL_ARGS=(--data-root "$DATA_ROOT" --wikidata-dir "$ROOT/data/wikidata" --seed "$SEED"
           --device cuda --dtype bfloat16 --local-files-only --no-download
           --allow-imperfect-direct-routing)
if [[ ! -f "$FINAL/official_rwku_eval.json" ]]; then
  echo "===== [RWKU s$SEED L$LAYER $GATE] 4/4 OFFICIAL RWKU eval (linear classifier routing) ====="
  python -u scripts/evaluate_rwku_fact_association_embeddings_seed1.py \
    --run-dir "$FINAL" "${EVAL_ARGS[@]}" --out "$FINAL/official_rwku_eval.json"
fi

if [[ "$BASE_EVAL" == "1" ]]; then
  BASE_OUT="$ROOT/outputs/rwku_multiseed_base_v1/seed$SEED"
  if [[ ! -f "$BASE_OUT/official_rwku_eval.json" ]]; then
    echo "===== [RWKU s$SEED] BASE eval (router with all-zero rows = unedited model) ====="
    mkdir -p "$BASE_OUT"
    python -u scripts/evaluate_rwku_fact_association_embeddings_seed1.py \
      --run-dir "$ROUTER" "${EVAL_ARGS[@]}" --out "$BASE_OUT/official_rwku_eval.json.tmp"
    mv "$BASE_OUT/official_rwku_eval.json.tmp" "$BASE_OUT/official_rwku_eval.json"
  fi
fi

if [[ "$RECAL" == "1" && "$GATE" == threshold && "$TRAINING_ROUTE" == router ]]; then
  echo "===== [RWKU s$SEED L$LAYER] RECAL recall>=$MIN_RECALL per fact on validation (cal + audit) ====="
  SEED="$SEED" LAYER="$LAYER" REF_TAG="$SWEEP_TAG" OBJECTIVE=min_recall MACRO=fact \
    MIN_RECALL="$MIN_RECALL" bash scripts/run_recalibration_one.sh rwku
fi

echo "===== [RWKU s$SEED L$LAYER $GATE] COMPLETE -> $BASE ====="
