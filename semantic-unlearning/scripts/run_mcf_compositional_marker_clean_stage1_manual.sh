#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 OUTPUT_DIR" >&2
  exit 2
fi

RUN_OUTPUT_DIR="$1"
if [[ -e "${RUN_OUTPUT_DIR}" ]]; then
  echo "output path already exists; choose a fresh clean-writer directory: ${RUN_OUTPUT_DIR}" >&2
  exit 2
fi

export PROJECT_DIR="${PROJECT_DIR:-$(pwd -P)}"
export OUTPUT_DIR="${RUN_OUTPUT_DIR}"
exec bash slurm/run_mcf_compositional_marker_clean_stage1_seed1_3b.slurm
