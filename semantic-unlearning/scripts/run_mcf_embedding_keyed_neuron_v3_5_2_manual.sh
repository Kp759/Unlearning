#!/usr/bin/env bash
set -euo pipefail

echo "V3.5.2 is preserved as the duplicate-prompt label-contradiction rejection." >&2
echo "Use run_mcf_embedding_keyed_neuron_v3_5_5_manual.sh with this rejection and the preserved V3.5.3/V3.5.4 outputs." >&2
echo "To reproduce V3.5.2 exactly, use historical commit be48b9bc2319c9ca55a3b9413f7e7e75c952b839." >&2
exit 2

if [[ $# -ne 6 ]]; then
  echo "usage: $0 CLEAN_WRITER_OUTPUT_DIR V3_2_OUTPUT_DIR V3_4_OUTPUT_DIR REJECTED_V3_5_OUTPUT_DIR V3_5_1_FORENSICS_OUTPUT_DIR OUTPUT_DIR" >&2
  exit 2
fi

CLEAN_WRITER_DIR="$1"
FROZEN_V3_2_DIR="$2"
FROZEN_V3_4_DIR="$3"
FROZEN_V3_5_DIR="$4"
FROZEN_V3_5_1_DIR="$5"
RUN_OUTPUT_DIR="$6"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

for source in \
  "${CLEAN_WRITER_DIR}" \
  "${FROZEN_V3_2_DIR}" \
  "${FROZEN_V3_4_DIR}" \
  "${FROZEN_V3_5_DIR}" \
  "${FROZEN_V3_5_1_DIR}"
do
  if [[ ! -d "${source}" ]]; then
    echo "required frozen source directory is missing: ${source}" >&2
    exit 2
  fi
done
if [[ -e "${RUN_OUTPUT_DIR}" ]]; then
  echo "output path already exists; choose a fresh V3.5.2 directory: ${RUN_OUTPUT_DIR}" >&2
  exit 2
fi

export PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd -P)}"
export CLEAN_WRITER_OUTPUT_DIR="${CLEAN_WRITER_DIR}"
export FROZEN_V3_2_OUTPUT_DIR="${FROZEN_V3_2_DIR}"
export FROZEN_V3_4_OUTPUT_DIR="${FROZEN_V3_4_DIR}"
export FROZEN_V3_5_OUTPUT_DIR="${FROZEN_V3_5_DIR}"
export FROZEN_V3_5_1_OUTPUT_DIR="${FROZEN_V3_5_1_DIR}"
export OUTPUT_DIR="${RUN_OUTPUT_DIR}"
exec bash "${PROJECT_DIR}/slurm/run_mcf_embedding_keyed_neuron_v3_5_2_global_writer_off_repair_seed1_3b.slurm"
