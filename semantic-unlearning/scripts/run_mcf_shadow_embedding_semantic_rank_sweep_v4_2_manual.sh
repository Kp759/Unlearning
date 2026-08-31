#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 CLEAN_V6_2_WRITER_OUTPUT_DIR FAILED_V4_OUTPUT_DIR FAILED_V4_1_OUTPUT_DIR OUTPUT_DIR" >&2
  exit 2
fi

CLEAN_WRITER_OUTPUT_DIR="$1"
FAILED_V4_OUTPUT_DIR="$2"
FAILED_V4_1_OUTPUT_DIR="$3"
OUTPUT_DIR="$4"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

for source in \
  "${CLEAN_WRITER_OUTPUT_DIR}" \
  "${FAILED_V4_OUTPUT_DIR}" \
  "${FAILED_V4_1_OUTPUT_DIR}"
do
  test -d "${source}" || {
    echo "required source output is missing: ${source}" >&2
    exit 2
  }
done
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "output already exists; choose a fresh V4.2 directory: ${OUTPUT_DIR}" >&2
  exit 2
fi

export PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd -P)}"
export CLEAN_WRITER_OUTPUT_DIR
export FAILED_V4_OUTPUT_DIR
export FAILED_V4_1_OUTPUT_DIR
export OUTPUT_DIR
exec bash "${PROJECT_DIR}/slurm/run_mcf_shadow_embedding_semantic_rank_sweep_v4_2_seed1_3b.slurm"
