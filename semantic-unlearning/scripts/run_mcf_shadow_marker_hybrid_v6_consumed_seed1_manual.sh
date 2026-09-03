#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 V3_6_2_TRAINING_RUN STAGE1_WRITER_STATE V6_SEED1_ROOT OUTPUT_DIR" >&2
  exit 2
fi

V3_TRAINING_RUN="$1"
STAGE1_WRITER_STATE="$2"
V6_SEED1_ROOT="$3"
OUTPUT_DIR="$4"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd -P)}"
MODEL_PATH="${MODEL_PATH:?MODEL_PATH must be exported}"
MCF_PATH="${MCF_PATH:-${PROJECT_DIR}/data/multi_counterfact.json}"
WIKIDATA_DIR="${WIKIDATA_DIR:?WIKIDATA_DIR must be exported}"

if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "hybrid output already exists: ${OUTPUT_DIR}" >&2
  exit 2
fi

python "${PROJECT_DIR}/scripts/evaluate_mcf_shadow_marker_hybrid_v6_consumed_seed1.py" \
  --model-path "${MODEL_PATH}" \
  --v3-6-2-training-run-dir "${V3_TRAINING_RUN}" \
  --stage1-writer-state "${STAGE1_WRITER_STATE}" \
  --v6-candidate-run-dir "${V6_SEED1_ROOT}/method/candidate" \
  --mcf-path "${MCF_PATH}" \
  --wikidata-dir "${WIKIDATA_DIR}" \
  --output-dir "${OUTPUT_DIR}"

jq '{
  completed,
  evaluation_status,
  arms: [
    .fixed_arm_order[] as $arm
    | {
        arm: $arm,
        switches: .arms[$arm].switches,
        Eff: .arms[$arm].forget.Eff,
        Gen: .arms[$arm].forget.Gen,
        Spe: .arms[$arm].forget.Spe,
        retain_Spe: .arms[$arm].retain.Spe,
        PPL: .arms[$arm].PPL,
        exact_preservation:
          (.exact_preservation_vs_base[$arm].passed // true)
      }
  ],
  mechanistic_conclusions,
  integrity
}' "${OUTPUT_DIR}/hybrid_replay.json"
