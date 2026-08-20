#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SURE_MCF_TARGET_AWARE_FS=1
export OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/mcf_sure_target_aware_direct_fs_v6}"

exec bash "${SCRIPT_DIR}/run_mcf_sure_minimal.sh" "$@"
