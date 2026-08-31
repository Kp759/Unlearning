#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 CLEAN_WRITER_OUTPUT_DIR PASSED_V3_6_2_OUTPUT_DIR PRESERVED_FAILED_OFFICIAL_DIR FRESH_RECOVERY_OUTPUT_DIR" >&2
  exit 2
fi

CLEAN_WRITER_DIR="$1"
FROZEN_V3_6_2_DIR="$2"
FAILED_OFFICIAL_DIR="$3"
RECOVERY_OUTPUT_DIR="$4"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

test -d "${CLEAN_WRITER_DIR}" || {
  echo "clean writer output is missing: ${CLEAN_WRITER_DIR}" >&2
  exit 2
}
test -d "${FROZEN_V3_6_2_DIR}" || {
  echo "V3.6.2 output is missing: ${FROZEN_V3_6_2_DIR}" >&2
  exit 2
}
test -d "${FAILED_OFFICIAL_DIR}" || {
  echo "preserved failed official output is missing: ${FAILED_OFFICIAL_DIR}" >&2
  exit 2
}
if [[ -e "${RECOVERY_OUTPUT_DIR}" ]]; then
  echo "recovery output already exists; resume and retry are prohibited: ${RECOVERY_OUTPUT_DIR}" >&2
  exit 2
fi
if [[ "$(cd "${FAILED_OFFICIAL_DIR}" && pwd -P)" == "$(cd "$(dirname "${RECOVERY_OUTPUT_DIR}")" && pwd -P)/$(basename "${RECOVERY_OUTPUT_DIR}")" ]]; then
  echo "recovery output must not overwrite the preserved failed output" >&2
  exit 2
fi

export PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd -P)}"
export CLEAN_WRITER_OUTPUT_DIR="${CLEAN_WRITER_DIR}"
export FROZEN_V3_6_2_OUTPUT_DIR="${FROZEN_V3_6_2_DIR}"
export FAILED_OFFICIAL_OUTPUT_DIR="${FAILED_OFFICIAL_DIR}"
export RECOVERY_OUTPUT_DIR
exec bash "${PROJECT_DIR}/slurm/run_mcf_embedding_keyed_neuron_v3_6_2_official_recovery_seed1_3b.slurm"
