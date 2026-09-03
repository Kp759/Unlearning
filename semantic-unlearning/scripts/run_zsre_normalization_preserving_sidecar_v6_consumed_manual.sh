#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 CONSUMED_SEED_1 OUTPUT_ROOT" >&2
  exit 2
fi

SEED="$1"
OUTPUT_ROOT="$2"
if [[ "${SEED}" != "1" ]]; then
  echo "ZsRE V6 development is restricted to consumed seed 1" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd -P)}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH must be exported}"
ZSRE_PATH="${ZSRE_PATH:-${PROJECT_DIR}/data/zsre_mend_eval.json}"
WIKIDATA_DIR="${WIKIDATA_DIR:?WIKIDATA_DIR must be exported}"
REGISTRY="${PROJECT_DIR}/protocols/zsre_normalization_preserving_sidecar_v6_0_seed1_development_registry.json"

if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "ZsRE V6 consumed-development output already exists: ${OUTPUT_ROOT}" >&2
  exit 2
fi

SPLIT_DIR="${OUTPUT_ROOT}/protocol/direct_only_split"
CANDIDATE_DIR="${OUTPUT_ROOT}/method/candidate"
REPLAY_DIR="${OUTPUT_ROOT}/development_replay"

python "${PROJECT_DIR}/scripts/build_zsre_normalization_preserving_sidecar_v6_consumed_split.py" \
  --zsre-path "${ZSRE_PATH}" \
  --output-dir "${SPLIT_DIR}" \
  --seed "${SEED}"

env \
  -u ZSRE_PATH \
  -u OFFICIAL_ZSRE_PATH \
  -u OFFICIAL_EVAL_PATH \
  -u PARAPHRASE_PATH \
  -u NEIGHBORHOOD_PATH \
  -u RETAIN_EVAL_PATH \
  -u WIKIDATA_DIR \
  python "${PROJECT_DIR}/scripts/build_zsre_normalization_preserving_sidecar_v6_candidate.py" \
    --model-path "${MODEL_PATH}" \
    --training-visible-path "${SPLIT_DIR}/training_visible_direct_only.json" \
    --split-manifest "${SPLIT_DIR}/split_manifest.json" \
    --development-registry "${REGISTRY}" \
    --output-dir "${CANDIDATE_DIR}" \
    --seed "${SEED}"

python "${PROJECT_DIR}/scripts/evaluate_zsre_normalization_preserving_sidecar_v6_consumed.py" \
  --model-path "${MODEL_PATH}" \
  --candidate-run-dir "${CANDIDATE_DIR}" \
  --split-manifest "${SPLIT_DIR}/split_manifest.json" \
  --zsre-path "${ZSRE_PATH}" \
  --wikidata-dir "${WIKIDATA_DIR}" \
  --output-dir "${REPLAY_DIR}" \
  --seed "${SEED}"

jq '{
  seed,
  passed,
  base: {
    Eff: .arms.base.forget.Eff,
    Gen: .arms.base.forget.Gen,
    Spe: .arms.base.forget.Spe,
    PPL: .arms.base.PPL
  },
  v6: {
    Eff: .arms.v6_sidecar.forget.Eff,
    Gen: .arms.v6_sidecar.forget.Gen,
    Spe: .arms.v6_sidecar.forget.Spe,
    PPL: .arms.v6_sidecar.PPL
  },
  behavioral_checks,
  collision_summary: {
    full_input_overlap_count:
      .scorer_collision_audit.full_input_overlap_count,
    exact_scorer_collision_count:
      .scorer_collision_audit.exact_scorer_collision_count
  },
  route_summary: {
    positive_coverage:
      .route_audit.all_forget_rewrite_paraphrase_owner_routes_open,
    forget_neighborhood_closed:
      .route_audit.all_forget_neighborhood_routes_closed,
    retain_closed: .route_audit.all_retain_routes_closed,
    ppl_closed: .route_audit.ppl_route_closed
  },
  exact_preservation: .exact_preservation.checks
}' "${REPLAY_DIR}/development_replay.json"
