#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 CONSUMED_SEED_1_OR_2 OUTPUT_ROOT" >&2
  exit 2
fi

SEED="$1"
OUTPUT_ROOT="$2"
if [[ "${SEED}" != "1" && "${SEED}" != "2" ]]; then
  echo "V6 development is restricted to consumed seeds 1 and 2" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd -P)}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH must be exported}"
MCF_PATH="${MCF_PATH:-${PROJECT_DIR}/data/multi_counterfact.json}"
WIKIDATA_DIR="${WIKIDATA_DIR:?WIKIDATA_DIR must be exported}"
FRAME_LEXICON="${PROJECT_DIR}/protocols/mcf_frozen_two_sided_relation_frame_lexicon_v1.json"
REGISTRY="${PROJECT_DIR}/protocols/mcf_normalization_preserving_sidecar_v6_0_development_registry.json"

if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "V6 consumed-development output already exists: ${OUTPUT_ROOT}" >&2
  exit 2
fi

SPLIT_DIR="${OUTPUT_ROOT}/protocol/direct_only_split"
CANDIDATE_DIR="${OUTPUT_ROOT}/method/candidate"
REPLAY_DIR="${OUTPUT_ROOT}/development_replay"

python "${PROJECT_DIR}/scripts/build_mcf_normalization_preserving_sidecar_v6_consumed_split.py" \
  --mcf-path "${MCF_PATH}" \
  --frame-lexicon "${FRAME_LEXICON}" \
  --output-dir "${SPLIT_DIR}" \
  --seed "${SEED}"

env \
  -u MCF_PATH \
  -u OFFICIAL_MCF_PATH \
  -u OFFICIAL_EVAL_PATH \
  -u PARAPHRASE_PATH \
  -u NEIGHBORHOOD_PATH \
  -u RETAIN_EVAL_PATH \
  -u ADVERSARIAL_EVAL_PATH \
  -u WIKIDATA_DIR \
  python "${PROJECT_DIR}/scripts/build_mcf_normalization_preserving_sidecar_v6_candidate.py" \
    --model-path "${MODEL_PATH}" \
    --training-visible-path "${SPLIT_DIR}/training_visible_target_aware_direct.json" \
    --split-manifest "${SPLIT_DIR}/split_manifest.json" \
    --development-registry "${REGISTRY}" \
    --frame-lexicon "${FRAME_LEXICON}" \
    --output-dir "${CANDIDATE_DIR}" \
    --seed "${SEED}"

python "${PROJECT_DIR}/scripts/evaluate_mcf_normalization_preserving_sidecar_v6_consumed.py" \
  --model-path "${MODEL_PATH}" \
  --candidate-run-dir "${CANDIDATE_DIR}" \
  --mcf-path "${MCF_PATH}" \
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
  exact_preservation: .exact_preservation.checks,
  route_summary: {
    positive_owner_coverage_passed:
      .route_audit.positive_owner_coverage_passed,
    forget_neighborhood_nested_entity_safety_passed:
      .route_audit.forget_neighborhood_nested_entity_safety_passed,
    retain_semantic_overlap_route_cells:
      .route_audit.retain_semantic_overlap_route_cells
  }
}' "${REPLAY_DIR}/development_replay.json"
