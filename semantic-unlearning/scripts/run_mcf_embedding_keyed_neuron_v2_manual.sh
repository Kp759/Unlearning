#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 V3_OUTPUT_DIR OUTPUT_DIR" >&2
  exit 2
fi

V3_ARTIFACT_DIR="$1"
RUN_OUTPUT_DIR="$2"
for required in \
  "${V3_ARTIFACT_DIR}/protocol/training_visible_target_aware_direct.json" \
  "${V3_ARTIFACT_DIR}/protocol/split_manifest.json" \
  "${V3_ARTIFACT_DIR}/method/context_manifest.json" \
  "${V3_ARTIFACT_DIR}/method/stage1_writer.pt" \
  "${V3_ARTIFACT_DIR}/method/stage1_writer_report.json"
do
  if [[ ! -f "${required}" ]]; then
    echo "required frozen V3 artifact is missing: ${required}" >&2
    exit 2
  fi
done

if [[ -e "${RUN_OUTPUT_DIR}" ]]; then
  echo "output path already exists; choose a fresh v2 directory: ${RUN_OUTPUT_DIR}" >&2
  exit 2
fi

export PROJECT_DIR="${PROJECT_DIR:-$(pwd -P)}"
export V3_OUTPUT_DIR="${V3_ARTIFACT_DIR}"
export OUTPUT_DIR="${RUN_OUTPUT_DIR}"
exec bash slurm/run_mcf_embedding_keyed_neuron_seed1_3b.slurm
