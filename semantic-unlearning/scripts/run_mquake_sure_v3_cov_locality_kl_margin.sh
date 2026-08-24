#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

V3_ROOT="${1:-outputs/mquake_pure_two_stage_directional_v3_3b}"
BASE_MODEL="${2:?Usage: bash scripts/run_mquake_sure_v3_cov_locality_kl_margin.sh V3_ROOT BASE_MODEL}"
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
# Deterministic dyadic feasibility line search.  The older ladder stopped at
# 2^-6, which can falsely freeze an otherwise feasible constrained trajectory
# when the current iterate lies close to the locality-KL boundary.  Extending
# to 2^-16 changes no model/utility hyperparameter; it only resolves the
# numerical step length more finely before declaring a proposal infeasible.
BACKTRACK="${SURE_STAGE2_BACKTRACK_SCALES:-0.5,0.25,0.125,0.0625,0.03125,0.015625,0.0078125,0.00390625,0.001953125,0.0009765625,0.00048828125,0.000244140625,0.0001220703125,0.00006103515625,0.000030517578125,0.0000152587890625}"
LOCALITY_PROMPTS="${SURE_RELATION_LOCALITY_PROMPTS:-1000}"
COV_FLOOR_RTOL="${SURE_WIKI_COV_EIGEN_FLOOR_RTOL:-1e-8}"
CONTRACTION_STEPS="${SURE_COV_CONTRACTION_STEPS:-40}"

WIKI_DOCS="${SURE_WIKI_COV_DOCUMENTS:-1000}"
WIKI_STATES="${SURE_WIKI_COV_STATES:-100000}"
WIKI_EXCLUDE_FIRST="${SURE_WIKI_COV_EXCLUDE_FIRST:-20}"
STATS_DIR="${V3_ROOT}/external_stats"
COV="${STATS_DIR}/wiki_lmhead_cov_req${WIKI_DOCS}docs_${WIKI_STATES}states_pplx${WIKI_EXCLUDE_FIRST}.pt"

test -d "${BASE_MODEL}"
test -f "${COV}" || {
  echo "Missing covariance cache: ${COV}" >&2
  echo "Run scripts/run_mquake_sure_v3_cov_locality_margin.sh once to build it." >&2
  exit 2
}

read -r -a SEEDS <<< "${SEEDS_TEXT}"
for SEED in "${SEEDS[@]}"; do
  ROOT="${V3_ROOT}/seed${SEED}"
  VISIBLE="${ROOT}/protocol/training_visible_forget.json"
  MANIFEST="${ROOT}/protocol/split_manifest.json"
  STAGE1="${ROOT}/level1_full_residual_directional_ga/checkpoint"
  OUT="${ROOT}/level2_cov_kl_locality_traincal_margin_v3"
  SUMMARY="${OUT}/two_stage_summary.json"

  test -d "${STAGE1}"
  test -f "${VISIBLE}"
  test -f "${MANIFEST}"
  DIRECT_COUNT="$(python -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["sampling"]["forget_atomic_fact_count"])' "${MANIFEST}")"

  rm -rf "${OUT}"
  mkdir -p "${OUT}"

  echo "===== SURE v3 COVARIANCE + KL-RELATION-LOCALITY: seed ${SEED} ====="
  echo "Reusing covariance cache: ${COV}"
  echo "Repair margin: max(${PROTECT_MARGIN}, median Stage1-success direct margin)"
  echo "Relation locality hard gate: exact sparse-head mean KL <= existing protected-KL budget ${MAX_PKL}"
  echo "Top1 locality changes are diagnostic only; they do not hard-freeze near-ties"
  echo "Feasibility line search: deterministic dyadic backtracking through 2^-16"
  echo "NO MQUAKE RETAIN / ATOMICGEN / TARGET_NEW / PARAPHRASE / NEIGHBORHOOD / MULTIHOP USED"

  python scripts/mquake_sure_stage2_cov_locality_kl_v3.py \
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

  python - "${SUMMARY}" "${MAX_PKL}" <<'PY'
import json, sys
path, budget = sys.argv[1], float(sys.argv[2])
p = json.load(open(path))
p["method"] = "MQuAKE SURE v3 Wikipedia-Covariance + KL-Relation-Locality Stage2"
p["protocol"] = "training-visible direct repair plus external PPL-disjoint Wikipedia covariance and KL-locality controls"
loc = p.setdefault("relation_locality", {})
loc["construction"] = (
    "training-visible relation templates x external Wikipedia titles; "
    "hard gate on exact sparse-head full-vocabulary mean KL from Stage1"
)
loc["hard_gate"] = "mean_full_vocabulary_KL <= max_protected_kl"
loc["kl_budget"] = budget
loc["top1_role"] = "diagnostic only; not a feasibility gate"
p["relation_locality_labels"] = "Stage1 distributions on external relation prompts; no benchmark retain labels"
with open(path, "w") as f:
    json.dump(p, f, indent=2)
    f.write("\n")
PY

  python - "${SUMMARY}" <<'PY'
import json, sys
p = json.load(open(sys.argv[1]))
l2 = p.get("level2", {})
wc = p.get("wiki_covariance", {})
loc = p.get("relation_locality", {})
print("===== COV + KL-LOCALITY FINAL TRAINING DIGEST =====")
print("repair_margin:", p.get("repair_margin"))
print("wiki_documents_actual:", wc.get("document_count"))
print("wiki_states:", wc.get("state_count"))
print("repair_basis_rank:", l2.get("repair_basis_rank"))
print("P_nullspace_leak64:", l2.get("P_nullspace_leak64"))
print("covariance_whitening_error:", wc.get("whitening_identity_max_abs_error"))
print("relation_count:", loc.get("relation_count"))
print("relation_locality_prompts:", loc.get("prompt_count"))
print("baseline_top1_in_edited_rows:", loc.get("baseline_top1_in_edited_rows"))
print("relation_locality_hard_gate:", loc.get("hard_gate"))
print("relation_locality_kl_budget:", loc.get("kl_budget"))
print("relation_locality_final_exact:", loc.get("final_exact_report"))
print("pre_contraction_covariance_cost:", l2.get("pre_contraction_covariance_cost"))
print("contraction_scale:", l2.get("contraction_scale"))
print("post_contraction_covariance_cost:", l2.get("post_contraction_covariance_cost"))
print("materialization_recovery_scale:", l2.get("materialization_recovery_scale"))
print("materialized_delta_norm:", l2.get("materialized_delta_norm"))
print("accepted_steps:", l2.get("accepted_steps"))
print("rolled_back_steps:", l2.get("rolled_back_steps"))
print("final_direct_gate:", p.get("final_gate"))
print("final_F_repair_margin_gate:", p.get("final_F_repair_margin_gate"))
print("stage1_successes_regressed:", p.get("stage1_successes_regressed"))
print("protected_kl:", p.get("protected_kl"))
print("frozen_non_head_exact:", p.get("frozen_non_head_exact"))
print("final_gates_pass:", p.get("final_gates_pass"))
print("checkpoint:", p.get("checkpoint"))
PY

done

echo "===== COVARIANCE + KL-RELATION-LOCALITY v3 COMPLETE ====="
