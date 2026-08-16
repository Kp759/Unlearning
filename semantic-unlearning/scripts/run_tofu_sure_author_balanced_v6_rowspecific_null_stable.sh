#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP="${SCRIPT_DIR}/.run_v6_rowspecific_null_stable_$$.sh"
trap 'rm -f -- "${TMP}"' EXIT
sed 's#python scripts/tofu_sure_rowspecific_null_v6.py#python scripts/tofu_sure_rowspecific_null_v6_stable.py#' \
  "${SCRIPT_DIR}/run_tofu_sure_author_balanced_v6_rowspecific_null.sh" > "${TMP}"
chmod +x "${TMP}"
bash "${TMP}" "$@"
