#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/mcf_sure_target_aware_true_ga_new_gd_v7}"

exec bash "${SCRIPT_DIR}/run_mcf_sure_target_aware.sh" "$@"
