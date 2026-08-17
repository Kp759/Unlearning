#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_RUNNER="${SCRIPT_DIR}/run_mquake_sure_v7_rank0_rank256_locked.sh"
TMP_RUNNER="${SCRIPT_DIR}/.run_mquake_sure_v7_rank0_rank256_locked_compat.$$.sh"

if [[ ! -f "${BASE_RUNNER}" ]]; then
  echo "Missing base runner: ${BASE_RUNNER}" >&2
  exit 2
fi

cleanup() {
  rm -f "${TMP_RUNNER}"
}
trap cleanup EXIT

sed \
  -e 's/mquake_sure_sparse_lm_gagd_v7\.py/mquake_sure_sparse_lm_gagd_v7_entry.py/g' \
  -e 's/mquake_sure_active_hidden_repair_v7\.py/mquake_sure_active_hidden_repair_v7_entry.py/g' \
  "${BASE_RUNNER}" > "${TMP_RUNNER}"

bash "${TMP_RUNNER}" "$@"
