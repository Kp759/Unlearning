#!/usr/bin/env bash
# Paper table, one (dataset, seed): base | ours (question-side Δe) | genie (same rows, perfect
# routing). Official evaluator of each benchmark; Eff / Gen / Spe / PPL.
#
# Defaults reproduce the published table (TAG l16_paper_v1): pass 1 routes at layer 19, pass 2
# adds Δe at layer 16 (scripts/early_write.py), logit-lens feedback at 24, regular_v1 routers.
# ONEPASS=1 (scripts/run_onepass_paper_one.sh) = ours now: ONE forward pass, the classifier
# reads layer ROUTER_LAYER (19) and Δe is added at the same layer, at the last prompt token and
# every generated token; no logit lens; MCF on the linear L19 router (v2.2 reads layers 23 / 27,
# above the write layer: it needs its own routing pass). Routers (ROUTER_RECIPE=ours, in
# outputs/<ds>_multiseed_ours_v1): no PCA, fuzzy subject matching, 20 positives per fact (the
# fact's prompts + quality-checked base-model rewrites, build_router_rewrites.py, + at most 8
# context-prefix copies; every third rewrite held out to calibrate the firing threshold) and 20
# negatives (same-subject competitors, then model-written hard negatives, then subject transplants;
# base-model answer-collision and plausibility checks).
#   ours                ONEPASS=1                  -> TAG onepass_l19_v1
#   ours, L16 ablation  ONEPASS=1 WRITE_LAYER=16   -> TAG l16_ours_v1 (same routers, Δe at 16)
#   published table     (defaults)                 -> TAG l16_paper_v1
#
#   bash scripts/run_l16_paper_one.sh DATASET SEED       DATASET in mcf zsre mquake rwku
#
# Stages (each skipped once its output exists; a half-written stage dir is moved aside):
#   1. router   linear classifier at L19 for this seed (the multiseed_regular_v1 router,
#               reused when present, else fitted via run_<ds>_layer_sweep_one.sh, STOP_AFTER=router)
#   2. v22      MCF only: factorized subject x relation router v2.2 (subject layer 6,
#               relation 19+23+27 picked on validation), reference = the L19 router above.
#               If no layer is deployable the seed falls back to the L19 router (logged in
#               router_choice.json).
#   3. base     official eval of the router with all-zero rows = the unedited model
#   4. ours     rows trained by train_mcf_jlens_feedback_rows.py: write layer WRITE_LAYER, query mode
#               (last prompt token + every generated token), output loss + KL keep, plus the plain
#               logit-lens feedback at LENS_LAYER (L24) unless LENS_TYPE=none (ours: the ablation
#               found it unnecessary); ZsRE/MQuAKE/RWKU add teacher-forced gold views
#               (their metrics score every answer token under the gold prefix)
#   5. genie    official eval of the SAME rows with --genie (forget prompts -> own row,
#               everything else -> no row; router not run)
#   5b. RWKU only: genie_entity (--genie-entity): every probe about a target person -> one of
#               that person's rows (the router's pick, else the first); neighbours / PPL none
#   6. recoverability (RECOVERABILITY=1, default off): scripts/representation_recoverability.py on
#               base, every baseline, ours and genie: can a linear readout (logit lens, tuned lens
#               fitted on retain text, answer probe trained on other subjects) still read the
#               forgotten answer from the residual stream after each block? (Gao et al. 2026,
#               arXiv:2604.08271). -> <run>/representation_recoverability.{json,md}; RR in the
#               paper table. Baseline dirs recompute the baseline edit (same seed) as the evaluator
#               does. RECOVER_ARGS = extra flags (e.g. "--layers 12,16,19,24 --no-probe").
#               scripts/run_recoverability_one.sh runs only this stage on finished dirs.
#
# Outputs: outputs/l16_paper_v1/<ds>/seed<S>/{base,ours,genie}/official_<ds>_eval.json
# Env: WRITE_LAYER (16), V22_SUBJECT_LAYERS ("6"), RWKU_GATE (threshold), TRAIN_ARGS (extra),
#      TAG (l16_paper_v1), MOVE_INCOMPLETE (1), ORACLE_UNROUTED (1: facts the router never
#      routes are trained with their own row forced, training only),
#      ZSRE_REWORD_TAG (reworded_v2 | reworded_v1 | none: ZsRE router trained on rewordings;
#                       ROUTER_RECIPE=ours defaults to none: its own checked rewrites replace them),
#      BASELINES ("ga gd npo rmu zerounlearn rome memit" -> <seed>/<baseline>/), SKIP_OURS (1 = baselines only),
#      BASELINE_MOM2_DATASET / BASELINE_MOM2_TEXT (zerounlearn rome memit: covariance cache name / corpus,
#                 scripts/build_edit_covariance.py; results on another covariance are superseded),
#      BUILD_ROUTERS (1; 0 = fail instead of fitting a missing router, for baseline jobs)
#      MCF_ROUTER (v22 | linear: MCF ours/genie on the stage-1 linear router instead of v2.2;
#                  needs its own TAG so the v2.2 rows under l16_paper_v1 are never touched)
#      BASE_FROM (a TAG: copy <TAG>/<ds>/seed<S>/base instead of re-evaluating; base = the
#                 stage-1 router with zero rows, identical for every router choice)
#        e.g. TAG=l16_paper_mcf_linear19 MCF_ROUTER=linear BASE_FROM=l16_paper_v1
#      HYBRID_GATE (learned | learned_or_exact: the linear router's heads gated by a learned
#                   subject head instead of / besides the exact subject match; MCF uses the
#                   stage-1 L19 heads, ZsRE its reworded router; scripts/fit_hybrid_subject_router.py
#                   -> outputs/<ds>_hybrid_subject_v1/seed<S>/{learned,learned_or_exact};
#                   own TAG required), HYBRID_SUBJECT_LAYERS ("6")
#        e.g. TAG=l16_paper_hybrid_learned_v1 HYBRID_GATE=learned BASE_FROM=l16_paper_v1
# Other models (scripts/paper_model_config.sh sets all of these; defaults = Llama-3.2-3B):
#      MODEL_PATH (default: from the reference manifests), MODEL_TAG (prefix of the router
#      dirs, outputs/<ds>_<MODEL_TAG>_multiseed_regular_v1/...; empty = the 3B dirs),
#      ROUTER_LAYER (19), ONEPASS (0 | 1), ROUTER_RECIPE (regular_v1 | ours),
#      LENS_TYPE (logit; ONEPASS=1: none | jlens), LENS_LAYER (24, used when LENS_TYPE != none),
#      V22_RELATION_LAYERS ("19 23 24 27"),
#      V22_RELATION_CONCAT (19,23,27), BASELINE_ARGS (extra paper_baselines.py flags)
set -euo pipefail

DS="${1:?Usage: run_l16_paper_one.sh DATASET SEED}"
SEED="${2:?Usage: run_l16_paper_one.sh DATASET SEED}"
case "$DS" in mcf|zsre|mquake|rwku) ;; *) echo "DATASET must be mcf|zsre|mquake|rwku" >&2; exit 2;; esac
ROUTER_LAYER="${ROUTER_LAYER:-19}"
if [[ "${ONEPASS:-0}" == 1 ]]; then   # ours: one pass at the router layer, no lens, 20 / 20 no-PCA routers
  WRITE_LAYER="${WRITE_LAYER:-$ROUTER_LAYER}"; LENS_TYPE="${LENS_TYPE:-none}"; ROUTER_RECIPE="${ROUTER_RECIPE:-ours}"
else                                  # the published configuration
  WRITE_LAYER="${WRITE_LAYER:-16}"; LENS_TYPE="${LENS_TYPE:-logit}"; ROUTER_RECIPE="${ROUTER_RECIPE:-regular_v1}"
fi
ONE_PASS=0; [[ "$WRITE_LAYER" == "$ROUTER_LAYER" ]] && ONE_PASS=1
RL="$(printf '%02d' "$ROUTER_LAYER")"
LENS_LAYER="${LENS_LAYER:-24}"
MP="${MODEL_TAG:+${MODEL_TAG}_}"   # router-dir prefix, empty for the 3B runs
[[ -n "${MODEL_PATH:-}" ]] && export MODEL_PATH   # the layer-sweep scripts pick it up
case "$ROUTER_RECIPE" in
  ours)
    FIT_EXTRA_ARGS="--pca-dims 0 --subject-match fuzzy --positives-per-fact 20 --negatives-per-fact 20"
    FIT_EXTRA_ARGS+=" --max-prefix-positives 8 --negative-answer-check --negative-plausibility-delta 1.5"
    ROUTER_REWRITES=1
    ZSRE_REWORD_TAG="${ZSRE_REWORD_TAG:-none}" ;;   # ZsRE uses this router, not the reworded_v2 one
  regular_v1)
    FIT_EXTRA_ARGS="--positives-per-fact 0 --negatives-per-fact 0 --pca-dims 0,64,256"
    ROUTER_REWRITES=0 ;;
  *) echo "ROUTER_RECIPE must be ours or regular_v1" >&2; exit 2 ;;
esac
export FIT_EXTRA_ARGS ROUTER_REWRITES            # read by run_<ds>_layer_sweep_one.sh's router fit
if [[ "$ONE_PASS" == 1 ]]; then
  TAG="${TAG:-onepass_l${ROUTER_LAYER}$([[ "$ROUTER_RECIPE" == regular_v1 ]] && echo _regular || true)_v1}"
  MCF_ROUTER="${MCF_ROUTER:-linear}"
elif [[ "$ROUTER_RECIPE" == regular_v1 ]]; then
  TAG="${TAG:-l16_paper_v1}"
  MCF_ROUTER="${MCF_ROUTER:-v22}"
else
  TAG="${TAG:-l16_ours_v1}"
  MCF_ROUTER="${MCF_ROUTER:-linear}"
fi
if [[ "$TAG" == l16_paper_v1 && "$ROUTER_RECIPE" != regular_v1 ]]; then
  # its rows were trained on the regular_v1 routers: the stale check would move them aside
  echo "TAG=l16_paper_v1 holds the regular_v1-router results: use ROUTER_RECIPE=regular_v1 or another TAG" >&2; exit 2
fi
case "$MCF_ROUTER" in v22|linear) ;; *) echo "MCF_ROUTER must be v22 or linear" >&2; exit 2;; esac
if [[ "$ONE_PASS" == 1 && "$DS" == mcf && "$MCF_ROUTER" == v22 ]]; then
  echo "one pass needs MCF_ROUTER=linear: v2.2 reads layers above the write layer (use WRITE_LAYER=16 for v2.2)" >&2; exit 2
fi
if [[ "$DS" == mcf && "$MCF_ROUTER" == linear && "$TAG" == l16_paper_v1 ]]; then
  # the stale check below would move the v2.2 ours/genie aside and retrain them
  echo "MCF_ROUTER=linear needs its own TAG (e.g. TAG=l16_paper_mcf_linear19)" >&2; exit 2
fi
if [[ "$DS" == rwku && "${RWKU_GATE:-threshold}" == subject && "$TAG" == l16_paper_v1 ]]; then
  # the stale check below would move the threshold-gate RWKU rows aside
  echo "RWKU_GATE=subject needs its own TAG (e.g. TAG=l16_paper_rwku_subject_v1)" >&2; exit 2
fi
HYBRID_GATE="${HYBRID_GATE:-}"
case "$HYBRID_GATE" in ""|learned|learned_or_exact) ;; *) echo "HYBRID_GATE must be learned or learned_or_exact" >&2; exit 2;; esac
if [[ -n "$HYBRID_GATE" && "$TAG" == l16_paper_v1 ]]; then
  echo "HYBRID_GATE needs its own TAG (e.g. TAG=l16_paper_hybrid_${HYBRID_GATE}_v1)" >&2; exit 2
fi
MOVE_INCOMPLETE="${MOVE_INCOMPLETE:-1}"
export RWKU_CHAT_DATE_STRING="${RWKU_CHAT_DATE_STRING:-26 Jul 2024}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OUT="$ROOT/outputs/$TAG/$DS/seed$SEED"
mkdir -p "$OUT"
log() { echo "===== [$DS s$SEED] $* | $(date)"; }

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

# ---------------------------------------------------------------- 1. L19 linear router
RTAG="${MP}multiseed_$([[ "$ROUTER_RECIPE" == regular_v1 ]] && echo regular_v1 || echo "${ROUTER_RECIPE}_v1")"
GATE="${RWKU_GATE:-threshold}"
[[ "$DS" == rwku && "$GATE" == subject ]] && RTAG="${MP}multiseed_subject_$([[ "$ROUTER_RECIPE" == regular_v1 ]] && echo v1 || echo "${ROUTER_RECIPE}_v1")"
LIN="$ROOT/outputs/${DS}_${RTAG}/seed$SEED/L$RL/router"
if [[ ! -f "$LIN/fact_association_embeddings.pt" ]]; then
  [[ "${BUILD_ROUTERS:-1}" == 1 ]] || { echo "L$ROUTER_LAYER router missing: $LIN (run the main job first)" >&2; exit 2; }
  log "1. fit L$ROUTER_LAYER linear router (STOP_AFTER=router)"
  SEED="$SEED" SWEEP_TAG="$RTAG" TRAINING_ROUTE=router NORM_SCALE=1 WITH_DECOMPOSITION=0 \
    STOP_AFTER=router GATE="$GATE" bash "scripts/run_${DS}_layer_sweep_one.sh" "$ROUTER_LAYER"
fi
test -f "$LIN/fact_association_embeddings.pt" || { echo "no L$ROUTER_LAYER router at $LIN" >&2; exit 2; }
log "1. L$ROUTER_LAYER router: $LIN"

# ---------------------------------------------------------------- 2. MCF: v2.2 router
ROUTER="$LIN"
CHOICE="$OUT/router_choice.json"
REF_MCF="$ROOT/outputs/mcf_fact_assoc_router_v2_seed1/association_manifest.json"
if [[ "$DS" == mcf && ( "$MCF_ROUTER" == linear || -n "$HYBRID_GATE" ) ]]; then
  # the hybrid needs per-fact linear heads; v2.2 has a relation softmax instead
  echo "{\"router\": \"linear_L$ROUTER_LAYER\", \"dir\": \"$LIN\", \"reason\": \"MCF_ROUTER=linear\"}" > "$CHOICE"
elif [[ "$DS" == mcf ]]; then
  V22="$OUT/router_v22"
  if [[ ! -f "$V22/summary.json" ]]; then
    [[ "${BUILD_ROUTERS:-1}" == 1 ]] || { echo "v2.2 router missing: $V22 (run the main job first)" >&2; exit 2; }
    log "2. fit factorized router v2.2 (subject layers ${V22_SUBJECT_LAYERS:-6})"
    mkdir -p "$V22"
    read -r -a SL <<< "${V22_SUBJECT_LAYERS:-6}"
    read -r -a RLS <<< "${V22_RELATION_LAYERS:-19 23 24 27}"
    python -u scripts/fit_factorized_router_v2.py --reference-dir "$LIN" --output-dir "$V22" \
      --subject-layers "${SL[@]}" --relation-layers "${RLS[@]}" \
      --relation-concat "${V22_RELATION_CONCAT:-19,23,27}" \
      --hard-rounds 1 --hard-fraction 0.1 --hard-weight 5 \
      --min-recall 0.95 --max-false-fire-same 0.10 --max-false-fire-other 0.01 \
      --near-miss-mode any_word --near-miss-cap 30 \
      --device cuda --local-files-only || echo "v2.2 fit exited non-zero (no deployable layer?)"
  fi
  SEL="$(jq -r '.selected_dir // empty' "$V22/summary.json" 2>/dev/null || true)"
  if [[ -n "$SEL" && -f "$V22/$SEL/fact_association_embeddings.pt" ]]; then
    ROUTER="$V22/$SEL"
    echo "{\"router\": \"factorized_v2.2\", \"dir\": \"$ROUTER\"}" > "$CHOICE"
  else
    echo "{\"router\": \"linear_L19_fallback\", \"dir\": \"$LIN\", \"reason\": \"v2.2 not deployable\"}" > "$CHOICE"
    echo "!!! v2.2 not deployable for seed $SEED: falling back to the L19 linear router" >&2
  fi
elif [[ "$DS" == zsre && "${ZSRE_REWORD_TAG:-reworded_v2}" != none ]]; then
  # ZsRE: the L19 router misses paraphrases (genie Gen ~2.5 vs ours ~12.5). Router trained on
  # base-model rewordings of each direct question (official rephrases never read):
  # zsre_router_rewordings.py; v2 keeps a rewording only if the base model's answer stays as
  # likely after it as after the direct question, and drops near-duplicates.
  RWTAG="${ZSRE_REWORD_TAG:-reworded_v2}"
  RW="$ROOT/outputs/zsre_${MP}multiseed_$RWTAG/seed$SEED"
  ROUTER="$RW/L$RL/router"
  if [[ ! -f "$ROUTER/fact_association_embeddings.pt" ]]; then
    [[ "${BUILD_ROUTERS:-1}" == 1 ]] || { echo "reworded ZsRE router missing: $ROUTER (run the main job first)" >&2; exit 2; }
    mkdir -p "$RW"
    if [[ ! -f "$RW/rewordings.json" ]]; then
      GEN=()
      [[ "$RWTAG" == reworded_v2 ]] && GEN=(--consistency-margin 1.0 --max-jaccard 0.8 --samples 16 --max-rounds 4)
      log "2. ZsRE rewordings ($RWTAG)"
      python -u scripts/zsre_router_rewordings.py generate \
        --prep-dir "$ROOT/outputs/zsre_${RTAG}/seed$SEED/L$RL/prep" \
        --out "$RW/rewordings.json.tmp" --device cuda --local-files-only "${GEN[@]}"
      mv "$RW/rewordings.json.tmp" "$RW/rewordings.json"
    fi
    log "2. ZsRE reworded L$ROUTER_LAYER router ($RWTAG)"
    SEED="$SEED" SWEEP_TAG="${MP}multiseed_$RWTAG" REWORDINGS="$RW/rewordings.json" TRAINING_ROUTE=router \
      NORM_SCALE=1 STOP_AFTER=router bash scripts/run_zsre_layer_sweep_one.sh "$ROUTER_LAYER"
  fi
  test -f "$ROUTER/fact_association_embeddings.pt" || { echo "no reworded router at $ROUTER" >&2; exit 2; }
  echo "{\"router\": \"linear_L${ROUTER_LAYER}_$RWTAG\", \"dir\": \"$ROUTER\"}" > "$CHOICE"
else
  LABEL="linear_L$ROUTER_LAYER"
  [[ "$DS" == rwku && "$GATE" == subject ]] && LABEL="${LABEL}_subject_gate"
  echo "{\"router\": \"$LABEL\", \"dir\": \"$LIN\"}" > "$CHOICE"
fi
if [[ -n "$HYBRID_GATE" ]]; then
  # ---------------------------------------------------------------- 2h. hybrid router
  # learned subject head (v2.2's MIL head, early layer) gating the linear heads chosen above
  HEADS="$ROUTER"
  HY="$ROOT/outputs/${DS}_${MP}hybrid_subject_v1/seed$SEED"
  if [[ -f "$HY/summary.json" ]]; then
    HREF="$(jq -r '.reference_dir // empty' "$HY/summary.json")"
    [[ "$(realpath "$HREF")" == "$(realpath "$HEADS")" ]] \
      || { echo "$HY was fit on $HREF, not $HEADS (move it aside)" >&2; exit 2; }
  fi
  if [[ ! -f "$HY/$HYBRID_GATE/fact_association_embeddings.pt" ]]; then
    [[ "${BUILD_ROUTERS:-1}" == 1 ]] || { echo "hybrid router missing: $HY (run the main job first)" >&2; exit 2; }
    log "2h. fit learned subject head (layers ${HYBRID_SUBJECT_LAYERS:-6}) over the heads of $HEADS"
    read -r -a HSL <<< "${HYBRID_SUBJECT_LAYERS:-6}"
    python -u scripts/fit_hybrid_subject_router.py --reference-dir "$HEADS" --output-dir "$HY" \
      --mcf-path "$(jq -r '.mcf_path' "$REF_MCF")" --subject-layers "${HSL[@]}" \
      --device cuda --local-files-only
  fi
  ROUTER="$HY/$HYBRID_GATE"
  test -f "$ROUTER/fact_association_embeddings.pt" || { echo "no hybrid router at $ROUTER" >&2; exit 2; }
  echo "{\"router\": \"hybrid_$HYBRID_GATE\", \"dir\": \"$ROUTER\", \"linear_heads_from\": \"$HEADS\"}" > "$CHOICE"
fi
log "2. router for ours/genie: $ROUTER"
# Never mix models: every run dir below inherits the router's model_path.
if [[ -n "${MODEL_PATH:-}" ]]; then
  for R in "$LIN" "$ROUTER"; do
    RM="$(jq -r '.model_path' "$R/association_manifest.json")"
    [[ "$(realpath "$RM")" == "$(realpath "$MODEL_PATH")" ]] \
      || { echo "router $R was fit on $RM, not MODEL_PATH=$MODEL_PATH" >&2; exit 2; }
  done
fi

# ---------------------------------------------------------------- evaluator per benchmark
evaluate() {  # run_dir [--genie]
  local dir="$1"; shift
  case "$DS" in
    mcf)
      python -u scripts/evaluate_static_overlap_fact_association_embeddings_official.py \
        --run-dir "$dir" --mcf-path "$(jq -r '.mcf_path' "$REF_MCF")" \
        --wikidata-dir "$ROOT/data/wikidata" --seed "$SEED" \
        --device cuda --dtype bfloat16 --local-files-only "$@" ;;
    zsre)
      python -u scripts/evaluate_zsre_fact_association_embeddings_official.py \
        --run-dir "$dir" --zsre-path "$ROOT/data/zsre_mend_eval.json" \
        --wikidata-dir "$ROOT/data/wikidata" --seed "$SEED" \
        --device cuda --dtype bfloat16 --batch-size 8 --local-files-only "$@" ;;
    mquake)
      python -u scripts/evaluate_mquake_fact_association_embeddings_official.py \
        --run-dir "$dir" --mquake-path "$ROOT/data/MQuAKE-CF-3k-v2.json" \
        --wikidata-dir "$ROOT/data/wikidata" --seed "$SEED" \
        --device cuda --dtype bfloat16 --batch-size 8 --local-files-only \
        --allow-imperfect-direct-routing "$@" ;;
    rwku)
      python -u scripts/evaluate_rwku_fact_association_embeddings_seed1.py \
        --run-dir "$dir" --data-root "$ROOT/data/rwku" --wikidata-dir "$ROOT/data/wikidata" \
        --seed "$SEED" --device cuda --dtype bfloat16 --local-files-only --no-download \
        --allow-imperfect-direct-routing --out "$dir/official_rwku_eval.json" "$@" ;;
  esac
}
EVAL_JSON="official_${DS}_eval.json"

# ---------------------------------------------------------------- 6. representation recoverability
RECOVER_JSON="representation_recoverability.json"
recover() {  # run_dir [extra args]: lens / tuned-lens / answer-probe readout per block
  local dir="$1"; shift
  local data=()
  case "$DS" in
    mcf)    data=(--mcf-path "$(jq -r '.mcf_path' "$REF_MCF")") ;;
    zsre)   data=(--zsre-path "$ROOT/data/zsre_mend_eval.json") ;;
    mquake) data=(--mquake-path "$ROOT/data/MQuAKE-CF-3k-v2.json") ;;
    rwku)   data=(--data-root "$ROOT/data/rwku" --no-download) ;;
  esac
  python -u scripts/representation_recoverability.py --run-dir "$dir" --dataset "$DS" "${data[@]}" \
    --seed "$SEED" --device cuda --dtype bfloat16 --local-files-only ${RECOVER_ARGS:-} "$@"
}
baseline_edit_args() {  # B run_dir -> EDIT_ARGS: the covariance the baseline's own eval recorded
  # (BASELINE_MOM2_DATASET overrides), so a recoverability pass recomputes the SAME edit
  EDIT_ARGS=()
  [[ " zerounlearn rome memit " == *" $1 "* ]] || return 0
  local name="${BASELINE_MOM2_DATASET:-}" text="${BASELINE_MOM2_TEXT:-}"
  if [[ -z "$name" && -f "$2/$EVAL_JSON" ]]; then
    name="$(jq -r '(.baseline // {}).mom2_dataset // empty' "$2/$EVAL_JSON")"
    text="$(jq -r '(.baseline // {}).mom2_text // empty' "$2/$EVAL_JSON")"
    [[ "$text" == *"vendored default"* ]] && text=""
  fi
  if [[ -n "$name" ]]; then
    EDIT_ARGS=(--mom2-dataset "$name")
    [[ -n "$text" ]] && EDIT_ARGS+=(--mom2-text "$text")
  fi
}
recover_base_and_baselines() {
  [[ "${RECOVERABILITY:-0}" == 1 ]] || return 0
  if [[ -f "$BASE/$EVAL_JSON" && ! -f "$BASE/$RECOVER_JSON" ]]; then
    log "6. recoverability: base"; recover "$BASE"
  fi
  local B BD
  for B in ${BASELINES:-}; do
    BD="$OUT/$B"
    [[ -f "$BD/$EVAL_JSON" && ! -f "$BD/$RECOVER_JSON" ]] || continue
    baseline_edit_args "$B" "$BD"
    log "6. recoverability: baseline $B (edit recomputed${EDIT_ARGS[0]:+, ${EDIT_ARGS[*]}})"
    recover "$BD" --baseline "$B" --baseline-views-from "$ROUTER" "${EDIT_ARGS[@]}" ${BASELINE_ARGS:-}
  done
}
recover_ours_genie() {
  [[ "${RECOVERABILITY:-0}" == 1 ]] || return 0
  if [[ -f "$OURS/$EVAL_JSON" && ! -f "$OURS/$RECOVER_JSON" ]]; then
    log "6. recoverability: ours"; recover "$OURS"
  fi
  if [[ -f "$GENIE/$EVAL_JSON" && ! -f "$GENIE/$RECOVER_JSON" ]]; then
    log "6. recoverability: genie (oracle routing)"; recover "$GENIE" --genie
  fi
}

link_run() {  # src dst: a run dir that points at src's artifact (so evals never touch src)
  mkdir -p "$2"
  for f in fact_association_embeddings.pt association_manifest.json linear_router_report.json \
           training_report.json association_examples.json; do
    [[ -f "$1/$f" && ! -e "$2/$f" ]] && ln -s "$1/$f" "$2/$f"
  done
  return 0
}

# ---------------------------------------------------------------- 3. base
BASE="$OUT/base"
SRC_BASE="$ROOT/outputs/${BASE_FROM:-}/$DS/seed$SEED/base"
if [[ -n "${BASE_FROM:-}" && ! -f "$BASE/$EVAL_JSON" && -f "$SRC_BASE/$EVAL_JSON" ]]; then
  log "3. base copied from $BASE_FROM (same stage-1 router, zero rows)"
  [[ -e "$BASE" ]] && mv "$BASE" "$BASE.incomplete_$(date +%Y%m%d_%H%M%S)"
  cp -a "$SRC_BASE" "$BASE"
fi
if [[ ! -f "$BASE/$EVAL_JSON" ]]; then
  log "3. base eval (L19 router, all-zero rows = unedited model)"
  link_run "$LIN" "$BASE"
  evaluate "$BASE"
fi

# ---------------------------------------------------------------- 3b. baselines (BASELINES="ga npo ...")
# Same training-visible forget views as ours (the router ours trains on), retain = base
# self-generated text, scored by the same official evaluator (scripts/paper_baselines.py).
for B in ${BASELINES:-}; do
  BD="$OUT/$B"
  # results from before early stopping (report has no stop_prob) are superseded
  if [[ -f "$BD/$EVAL_JSON" && " ga gd npo rmu " == *" $B "* ]] && \
     [[ "$(jq -r '(.baseline // {}) | has("stop_prob")' "$BD/$EVAL_JSON" 2>/dev/null)" != true ]]; then
    TS="$(date +%Y%m%d_%H%M%S)"
    echo "baseline $B predates early stopping: moving aside ($BD.superseded_$TS)" >&2
    mv "$BD" "$BD.superseded_$TS"
  fi
  # locate-and-edit baselines: covariance from BASELINE_MOM2_DATASET (scripts/build_edit_covariance.py);
  # a result built on another covariance (e.g. the 200-sentence data/wikidata fixture) is superseded
  EDIT_ARGS=()
  if [[ " zerounlearn rome memit " == *" $B "* && -n "${BASELINE_MOM2_DATASET:-}" ]]; then
    EDIT_ARGS=(--mom2-dataset "$BASELINE_MOM2_DATASET")
    [[ -n "${BASELINE_MOM2_TEXT:-}" ]] && EDIT_ARGS+=(--mom2-text "$BASELINE_MOM2_TEXT")
    if [[ -f "$BD/$EVAL_JSON" ]] && \
       [[ "$(jq -r '(.baseline // {}).mom2_dataset // "wikipedia"' "$BD/$EVAL_JSON" 2>/dev/null)" != "$BASELINE_MOM2_DATASET" ]]; then
      TS="$(date +%Y%m%d_%H%M%S)"
      echo "baseline $B used another covariance: moving aside ($BD.superseded_$TS)" >&2
      mv "$BD" "$BD.superseded_$TS"
    fi
  fi
  if [[ ! -f "$BD/$EVAL_JSON" ]]; then
    log "3b. baseline $B ${EDIT_ARGS[*]:-}"
    [[ -e "$BD" ]] && mv "$BD" "$BD.incomplete_$(date +%Y%m%d_%H%M%S)"
    link_run "$LIN" "$BD"
    evaluate "$BD" --baseline "$B" --baseline-views-from "$ROUTER" "${EDIT_ARGS[@]}" ${BASELINE_ARGS:-}
  fi
done
recover_base_and_baselines
if [[ "${SKIP_OURS:-0}" == 1 ]]; then log "COMPLETE (baselines only) -> $OUT"; exit 0; fi

# ---------------------------------------------------------------- 4. ours
OURS="$OUT/ours"
GENIE="$OUT/genie"
TF=()
[[ "$DS" != mcf ]] && TF=(--teacher-forced-views)
ORACLE=()
[[ "${ORACLE_UNROUTED:-1}" == 1 ]] && ORACLE=(--oracle-unrouted)
# A finished `ours` trained under another config (router, flags) is moved aside together
# with its genie, so a resubmit with new settings retrains instead of silently reusing it.
if [[ -f "$OURS/fact_association_embeddings.pt" ]]; then
  STALE="$(python - "$OURS" "$ROUTER" "${#TF[@]}" "${#ORACLE[@]}" "$WRITE_LAYER" "$LENS_TYPE" <<'EOF'
import json, sys, torch
from pathlib import Path
ours, router, tf, oracle, wl = Path(sys.argv[1]), str(Path(sys.argv[2]).resolve()), sys.argv[3] != "0", sys.argv[4] != "0", int(sys.argv[5])
lens_type = sys.argv[6]
m = json.loads((ours / "association_manifest.json").read_text())
fb = m.get("jlens_feedback") or {}
why = []
if str(Path(m.get("router_copied_from", "")).resolve()) != router: why.append("router")
if bool(fb.get("teacher_forced_views")) != tf: why.append("teacher_forced_views")
if int(m.get("write_layer", wl)) != wl: why.append("write_layer")
if str(fb.get("lens_type", "logit")) != lens_type: why.append("lens_type")
if oracle and not fb.get("oracle_unrouted"):
    rows = torch.load(ours / "fact_association_embeddings.pt", map_location="cpu", weights_only=False)["rows"]
    if int((rows.abs().sum(-1) == 0).sum()):   # untrained (zero) rows exist -> the flag matters
        why.append("zero_rows")
print(",".join(why))
EOF
)"
  if [[ -n "$STALE" && "${NO_RETRAIN:-0}" == 1 ]]; then
    echo "ours does not match this configuration ($STALE) and NO_RETRAIN=1: refusing to move it aside" >&2; exit 2
  fi
  if [[ -n "$STALE" ]]; then
    TS="$(date +%Y%m%d_%H%M%S)"
    echo "ours is stale ($STALE): moving ours/genie aside (*.superseded_$TS)" >&2
    mv "$OURS" "$OURS.superseded_$TS"
    [[ -e "$GENIE" ]] && mv "$GENIE" "$GENIE.superseded_$TS"
  fi
fi
if ! stage_ready "$OURS" fact_association_embeddings.pt; then
  LENS_DESC="no lens"; [[ "$LENS_TYPE" != none ]] && LENS_DESC="$LENS_TYPE lens L$LENS_LAYER"
  log "4. train ours (write L$WRITE_LAYER, query, $LENS_DESC, ${TF[*]:-no TF views} ${ORACLE[*]})"
  python -u scripts/train_mcf_jlens_feedback_rows.py --router-dir "$ROUTER" --output-dir "$OURS" \
    --write-layer "$WRITE_LAYER" --write-mode query --lens-type "$LENS_TYPE" --lens-layer "$LENS_LAYER" \
    "${TF[@]}" "${ORACLE[@]}" ${TRAIN_ARGS:-} --device cuda --local-files-only
fi
if [[ ! -f "$OURS/$EVAL_JSON" ]]; then
  log "4. ours eval"
  evaluate "$OURS"
fi

# ---------------------------------------------------------------- 5. genie
if [[ ! -f "$GENIE/$EVAL_JSON" ]]; then
  log "5. genie eval (same rows, perfect routing)"
  link_run "$OURS" "$GENIE"
  evaluate "$GENIE" --genie
fi
if [[ "$DS" == rwku ]]; then
  # entity-level genie: perfect PERSON routing (held-out probes about a target -> that person's row)
  GENIE_E="$OUT/genie_entity"
  if [[ ! -f "$GENIE_E/$EVAL_JSON" ]]; then
    log "5b. entity genie eval (same rows, every probe about a target person routed)"
    link_run "$OURS" "$GENIE_E"
    evaluate "$GENIE_E" --genie-entity
  fi
fi
recover_ours_genie
log "COMPLETE -> $OUT"
