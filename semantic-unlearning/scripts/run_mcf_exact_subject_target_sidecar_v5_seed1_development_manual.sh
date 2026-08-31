#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 V5_CANDIDATE_OUTPUT_DIR DEVELOPMENT_OUTPUT_DIR" >&2
  exit 2
fi

V5_CANDIDATE_OUTPUT_DIR="$1"
DEVELOPMENT_OUTPUT_DIR="$2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"

test -d "${V5_CANDIDATE_OUTPUT_DIR}" || {
  echo "V5 candidate output is missing: ${V5_CANDIDATE_OUTPUT_DIR}" >&2
  exit 2
}
if [[ -e "${DEVELOPMENT_OUTPUT_DIR}" ]]; then
  echo "development output already exists: ${DEVELOPMENT_OUTPUT_DIR}" >&2
  exit 2
fi

export PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd -P)}"
export V5_CANDIDATE_OUTPUT_DIR
export DEVELOPMENT_OUTPUT_DIR
exec bash "${PROJECT_DIR}/slurm/run_mcf_exact_subject_target_sidecar_v5_seed1_development.slurm"
