#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 SEED V5_CANDIDATE_OUTPUT_DIR OFFICIAL_OUTPUT_DIR" >&2
  exit 2
fi

SEED="$1"
V5_CANDIDATE_OUTPUT_DIR="$2"
OFFICIAL_OUTPUT_DIR="$3"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd -P)}"

if [[ -e "${V5_CANDIDATE_OUTPUT_DIR}" ]]; then
  echo "candidate output already exists: ${V5_CANDIDATE_OUTPUT_DIR}" >&2
  exit 2
fi
if [[ -e "${OFFICIAL_OUTPUT_DIR}" ]]; then
  echo "official output already exists; retry/resume prohibited: ${OFFICIAL_OUTPUT_DIR}" >&2
  exit 2
fi

export PROJECT_DIR SEED V5_CANDIDATE_OUTPUT_DIR OFFICIAL_OUTPUT_DIR
export OUTPUT_DIR="${V5_CANDIDATE_OUTPUT_DIR}"
mkdir -p "${PROJECT_DIR}/slurm_logs"

bash "${PROJECT_DIR}/slurm/run_mcf_exact_subject_target_sidecar_v5_fresh_seed_build_3b.slurm"
bash "${PROJECT_DIR}/slurm/run_mcf_exact_subject_target_sidecar_v5_fresh_seed_official_3b.slurm"
