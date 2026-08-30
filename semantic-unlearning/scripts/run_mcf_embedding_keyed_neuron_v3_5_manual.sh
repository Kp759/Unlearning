#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 CLEAN_WRITER_OUTPUT_DIR REJECTED_V3_2_OUTPUT_DIR REJECTED_V3_4_OUTPUT_DIR OUTPUT_DIR" >&2
  exit 2
fi

CLEAN_WRITER_DIR="$1"
FROZEN_V3_2_DIR="$2"
FROZEN_V3_4_DIR="$3"
RUN_OUTPUT_DIR="$4"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

for required in \
  "${CLEAN_WRITER_DIR}/protocol/training_visible_target_aware_direct.json" \
  "${CLEAN_WRITER_DIR}/protocol/split_manifest.json" \
  "${CLEAN_WRITER_DIR}/method/context_manifest.json" \
  "${CLEAN_WRITER_DIR}/method/stage1_writer.pt" \
  "${CLEAN_WRITER_DIR}/method/stage1_writer_report.json" \
  "${CLEAN_WRITER_DIR}/method/stage1_writer_log.jsonl" \
  "${CLEAN_WRITER_DIR}/method/stage1_gradient_conflict_audit.json" \
  "${CLEAN_WRITER_DIR}/method/training_safe_portability_preflight.json" \
  "${CLEAN_WRITER_DIR}/method/clean_stage1_acceptance.json"
do
  if [[ ! -f "${required}" ]]; then
    echo "required clean Stage-1 artifact is missing: ${required}" >&2
    exit 2
  fi
done

for source in "${FROZEN_V3_2_DIR}" "${FROZEN_V3_4_DIR}"; do
  if [[ ! -d "${source}" ]]; then
    echo "required frozen source directory is missing: ${source}" >&2
    exit 2
  fi
done
if [[ -e "${RUN_OUTPUT_DIR}" ]]; then
  echo "output path already exists; choose a fresh V3.5 directory: ${RUN_OUTPUT_DIR}" >&2
  exit 2
fi

export PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd -P)}"
export CLEAN_WRITER_OUTPUT_DIR="${CLEAN_WRITER_DIR}"
export FROZEN_V3_2_OUTPUT_DIR="${FROZEN_V3_2_DIR}"
export FROZEN_V3_4_OUTPUT_DIR="${FROZEN_V3_4_DIR}"
export OUTPUT_DIR="${RUN_OUTPUT_DIR}"
exec bash "${PROJECT_DIR}/slurm/run_mcf_embedding_keyed_neuron_v3_5_isolated_threshold_seed1_3b.slurm"
