#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 11 ]]; then
  echo "usage: $0 CLEAN_WRITER_OUTPUT_DIR V3_2_OUTPUT_DIR V3_4_OUTPUT_DIR REJECTED_V3_5_OUTPUT_DIR V3_5_1_FORENSICS_OUTPUT_DIR REJECTED_V3_5_2_OUTPUT_DIR REJECTED_V3_5_3_OUTPUT_DIR REJECTED_V3_5_4_OUTPUT_DIR PASSED_V3_5_5_OUTPUT_DIR REJECTED_V3_6_OUTPUT_DIR OUTPUT_DIR" >&2
  exit 2
fi

CLEAN_WRITER_DIR="$1"
FROZEN_V3_2_DIR="$2"
FROZEN_V3_4_DIR="$3"
FROZEN_V3_5_DIR="$4"
FROZEN_V3_5_1_DIR="$5"
FROZEN_V3_5_2_DIR="$6"
FROZEN_V3_5_3_DIR="$7"
FROZEN_V3_5_4_DIR="$8"
FROZEN_V3_5_5_DIR="$9"
FROZEN_V3_6_DIR="${10}"
RUN_OUTPUT_DIR="${11}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

for source in \
  "${CLEAN_WRITER_DIR}" \
  "${FROZEN_V3_2_DIR}" \
  "${FROZEN_V3_4_DIR}" \
  "${FROZEN_V3_5_DIR}" \
  "${FROZEN_V3_5_1_DIR}" \
  "${FROZEN_V3_5_2_DIR}" \
  "${FROZEN_V3_5_3_DIR}" \
  "${FROZEN_V3_5_4_DIR}" \
  "${FROZEN_V3_5_5_DIR}" \
  "${FROZEN_V3_6_DIR}"
do
  test -d "${source}" || {
    echo "required frozen source directory is missing: ${source}" >&2
    exit 2
  }
done
if [[ -e "${RUN_OUTPUT_DIR}" ]]; then
  echo "output path already exists; choose a fresh V3.6.1 directory: ${RUN_OUTPUT_DIR}" >&2
  exit 2
fi

export PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd -P)}"
export CLEAN_WRITER_OUTPUT_DIR="${CLEAN_WRITER_DIR}"
export FROZEN_V3_2_OUTPUT_DIR="${FROZEN_V3_2_DIR}"
export FROZEN_V3_4_OUTPUT_DIR="${FROZEN_V3_4_DIR}"
export FROZEN_V3_5_OUTPUT_DIR="${FROZEN_V3_5_DIR}"
export FROZEN_V3_5_1_OUTPUT_DIR="${FROZEN_V3_5_1_DIR}"
export FROZEN_V3_5_2_OUTPUT_DIR="${FROZEN_V3_5_2_DIR}"
export FROZEN_V3_5_3_OUTPUT_DIR="${FROZEN_V3_5_3_DIR}"
export FROZEN_V3_5_4_OUTPUT_DIR="${FROZEN_V3_5_4_DIR}"
export FROZEN_V3_5_5_OUTPUT_DIR="${FROZEN_V3_5_5_DIR}"
export FROZEN_V3_6_OUTPUT_DIR="${FROZEN_V3_6_DIR}"
export OUTPUT_DIR="${RUN_OUTPUT_DIR}"
exec bash "${PROJECT_DIR}/slurm/run_mcf_embedding_keyed_neuron_v3_6_1_coherent_preservation_seed1_3b.slurm"
