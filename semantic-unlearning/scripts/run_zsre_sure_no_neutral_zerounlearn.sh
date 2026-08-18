#!/usr/bin/env bash
set -euo pipefail

echo "[deprecated entry point] run_zsre_sure_no_neutral_zerounlearn.sh now forwards to the canonical shared SURE pipeline." >&2
exec bash "$(dirname "$0")/run_zsre_sure_canonical.sh" "$@"
