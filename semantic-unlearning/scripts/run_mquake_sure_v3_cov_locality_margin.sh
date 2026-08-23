#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

V3_ROOT="${1:-outputs/mquake_pure_two_stage_directional_v3_3b}"
BASE_MODEL="${2:?Usage: bash scripts/run_mquake_sure_v3_cov_locality_margin.sh V3_ROOT BASE_MODEL}"
SEEDS_TEXT="${MQUAKE_COVLOCAL_V3_SEEDS:-2}"
DTYPE="${DTYPE:-bf16}"
DEVICE_MAP="${DEVICE_MAP:-single}"
CACHE_BATCH="${SURE_CACHE_BATCH_SIZE:-8}"
PROTECT_MARGIN="${MQUAKE_SURE_CONSTRAINT_MARGIN:-0.05}"
MAX_PKL="${SURE_MAX_PROTECTED_KL:-0.05}"
REPAIR_STEPS="${SURE_STAGE2_STEPS:-800}"
REPAIR_LR="${SURE_STAGE2_LR:-0.0005}"
REPAIR_BATCH="${SURE_STAGE2_BATCH_SIZE:-8}"
REPAIR_CHECK="${SURE_STAGE2_CHECK_EVERY:-25}"
REPAIR_L2="${SURE_STAGE2_L2_WEIGHT:-0.000001}"
BACKTRACK="${SURE_STAGE2_BACKTRACK_SCALES:-0.5,0.25,0.125,0.0625,0.03125,0.015625}"

WIKIDATA_DIR="${WIKIDATA_DIR:-data/wikidata}"
WIKI_DOCS="${SURE_WIKI_COV_DOCUMENTS:-1000}"
WIKI_STATES="${SURE_WIKI_COV_STATES:-100000}"
WIKI_EXCLUDE_FIRST="${SURE_WIKI_COV_EXCLUDE_FIRST:-20}"
WIKI_MAXLEN="${SURE_WIKI_COV_MAX_LENGTH:-1024}"
WIKI_BATCH="${SURE_WIKI_COV_BATCH_SIZE:-4}"
WIKI_SEED="${SURE_WIKI_COV_SEED:-1729}"
LOCALITY_PROMPTS="${SURE_RELATION_LOCALITY_PROMPTS:-1000}"
COV_FLOOR_RTOL="${SURE_WIKI_COV_EIGEN_FLOOR_RTOL:-1e-8}"
CONTRACTION_STEPS="${SURE_COV_CONTRACTION_STEPS:-40}"

if (( WIKI_STATES % WIKI_DOCS != 0 )); then
  echo "WIKI_STATES must be divisible by WIKI_DOCS" >&2
  exit 2
fi
if (( WIKI_EXCLUDE_FIRST < 20 )); then
  echo "SURE_WIKI_COV_EXCLUDE_FIRST must be >=20 to remain disjoint from the repository PPL probe" >&2
  exit 2
fi
STATES_PER_DOC=$((WIKI_STATES / WIKI_DOCS))

STATS_DIR="${V3_ROOT}/external_stats"
COV="${STATS_DIR}/wiki_lmhead_cov_${WIKI_DOCS}docs_${WIKI_STATES}states_pplx${WIKI_EXCLUDE_FIRST}.pt"
mkdir -p "${STATS_DIR}"
test -d "${WIKIDATA_DIR}"
test -d "${BASE_MODEL}"

if [[ ! -f "${COV}" ]]; then
  echo "===== BUILD PPL-DISJOINT EXTERNAL WIKIPEDIA LM-HEAD COVARIANCE ====="
  echo "documents=${WIKI_DOCS} states=${WIKI_STATES} states/doc=${STATES_PER_DOC} exclude_first=${WIKI_EXCLUDE_FIRST} corpus_seed=${WIKI_SEED}"
  echo "NO MQUAKE RETAIN / ATOMICGEN / TARGET_NEW / NEIGHBORHOOD DATA USED"
  python scripts/mquake_sure_build_wiki_lmhead_covariance.py \
    --model-path "${BASE_MODEL}" \
    --wikidata-dir "${WIKIDATA_DIR}" \
    --output "${COV}" \
    --documents "${WIKI_DOCS}" \
    --states-per-document "${STATES_PER_DOC}" \
    --exclude-first "${WIKI_EXCLUDE_FIRST}" \
    --max-length "${WIKI_MAXLEN}" \
    --batch-size "${WIKI_BATCH}" \
    --corpus-seed "${WIKI_SEED}" \
    --dtype "${DTYPE}" \
    --device-map "${DEVICE_MAP}"
else
  echo "===== REUSE CACHED PPL-DISJOINT WIKIPEDIA LM-HEAD COVARIANCE ====="
  echo "${COV}"
fi

read -r -a SEEDS <<< "${SEEDS_TEXT}"
for SEED in "${SEEDS[@]}"; do
  ROOT="${V3_ROOT}/seed${SEED}"
  PROTOCOL="${ROOT}/protocol"
  VISIBLE="${PROTOCOL}/training_visible_forget.json"
  MANIFEST="${PROTOCOL}/split_manifest.json"
  STAGE1="${ROOT}/level1_full_residual_directional_ga/checkpoint"
  OUT="${ROOT}/level2_cov_locality_traincal_margin_v3"
  SUMMARY="${OUT}/two_stage_summary.json"

  test -d "${STAGE1}"
  test -f "${VISIBLE}"
  test -f "${MANIFEST}"
  DIRECT_COUNT="$(python -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["sampling"]["forget_atomic_fact_count"])' "${MANIFEST}")"

  rm -rf "${OUT}"
  mkdir -p "${OUT}"

  echo "===== SURE v3 WIKIPEDIA-COVARIANCE + RELATION-LOCALITY STAGE2: seed ${SEED} ====="
  echo "Repair margin: max(${PROTECT_MARGIN}, median Stage1-success direct margin)"
  echo "Utility metric: ${WIKI_DOCS} PPL-disjoint Wikipedia docs / ${WIKI_STATES} LM-head states"
  echo "Relation locality: ${LOCALITY_PROMPTS} external Wikipedia-title controls; preserve Stage1 top1 exactly"
  echo "NO MQUAKE RETAIN / ATOMICGEN / TARGET_NEW / PARAPHRASE / NEIGHBORHOOD / MULTIHOP USED"

  python scripts/mquake_sure_stage2_cov_locality_v3.py \
    --model-path "${STAGE1}" \
    --training-visible-path "${VISIBLE}" \
    --split-manifest "${MANIFEST}" \
    --wiki-covariance "${COV}" \
    --output-dir "${OUT}" \
    --seed "${SEED}" \
    --forget-num "${DIRECT_COUNT}" \
    --repair-steps "${REPAIR_STEPS}" \
    --repair-lr "${REPAIR_LR}" \
    --batch-size "${REPAIR_BATCH}" \
    --cache-batch-size "${CACHE_BATCH}" \
    --constraint-margin "${PROTECT_MARGIN}" \
    --max-protected-kl "${MAX_PKL}" \
    --l2-weight "${REPAIR_L2}" \
    --check-every "${REPAIR_CHECK}" \
    --backtrack-scales "${BACKTRACK}" \
    --locality-prompts "${LOCALITY_PROMPTS}" \
    --cov-eigen-floor-rtol "${COV_FLOOR_RTOL}" \
    --contraction-steps "${CONTRACTION_STEPS}" \
    --dtype "${DTYPE}" \
    --device-map "${DEVICE_MAP}"

  python - "${SUMMARY}" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
l2 = p.get("level2", {})
wc = p.get("wiki_covariance", {})
loc = p.get("relation_locality", {})
print("===== COV-LOCALITY FINAL TRAINING DIGEST =====")
print("repair_margin:", p.get("repair_margin"))
print("wiki_documents:", wc.get("document_count"))
print("wiki_states:", wc.get("state_count"))
print("repair_basis_rank:", l2.get("repair_basis_rank"))
print("P_nullspace_leak64:", l2.get("P_nullspace_leak64"))
print("covariance_whitening_error:", wc.get("whitening_identity_max_abs_error"))
print("relation_count:", loc.get("relation_count"))
print("relation_locality_prompts:", loc.get("prompt_count"))
print("baseline_top1_in_edited_rows:", loc.get("baseline_top1_in_edited_rows"))
print("relation_locality_final_exact:", loc.get("final_exact_report"))
print("pre_contraction_covariance_cost:", l2.get("pre_contraction_covariance_cost"))
print("contraction_scale:", l2.get("contraction_scale"))
print("post_contraction_covariance_cost:", l2.get("post_contraction_covariance_cost"))
print("materialization_recovery_scale:", l2.get("materialization_recovery_scale"))
print("materialized_delta_norm:", l2.get("materialized_delta_norm"))
print("final_direct_gate:", p.get("final_gate"))
print("final_F_repair_margin_gate:", p.get("final_F_repair_margin_gate"))
print("stage1_successes_regressed:", p.get("stage1_successes_regressed"))
print("protected_kl:", p.get("protected_kl"))
print("frozen_non_head_exact:", p.get("frozen_non_head_exact"))
print("final_gates_pass:", p.get("final_gates_pass"))
print("checkpoint:", p.get("checkpoint"))
PY

done

echo "===== COVARIANCE + RELATION-LOCALITY v3 TRAINING COMPLETE ====="
echo "Held-out metrics were not evaluated by this runner."
