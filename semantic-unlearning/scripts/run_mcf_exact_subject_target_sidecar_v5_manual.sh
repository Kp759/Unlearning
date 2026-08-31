#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 CLEAN_V6_2_WRITER_OUTPUT_DIR OUTPUT_DIR" >&2
  exit 2
fi

CLEAN_WRITER_OUTPUT_DIR="$1"
OUTPUT_DIR="$2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

test -d "${CLEAN_WRITER_OUTPUT_DIR}" || {
  echo "clean V6.2 writer output is missing: ${CLEAN_WRITER_OUTPUT_DIR}" >&2
  exit 2
}
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "output already exists; choose a fresh V5 directory: ${OUTPUT_DIR}" >&2
  exit 2
fi

export PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd -P)}"
export CLEAN_WRITER_OUTPUT_DIR
export OUTPUT_DIR
exec bash "${PROJECT_DIR}/slurm/run_mcf_exact_subject_target_sidecar_v5_seed1_3b.slurm"
