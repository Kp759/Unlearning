#!/usr/bin/env bash
set -euo pipefail

# Delete only bulky model checkpoint directories from the known MQuAKE V7,
# V7.1 and V7.2 experiment roots. JSON/JSONL diagnostics and eval results stay.
# The V7.3 output tree is intentionally excluded.

OUTPUTS_ROOT="${1:-outputs}"
MODE="${2:---dry-run}"

case "${MODE}" in
  --dry-run|--execute) ;;
  *)
    echo "Usage: bash scripts/cleanup_mquake_v7_old_checkpoints.sh [OUTPUTS_ROOT] [--dry-run|--execute]" >&2
    exit 2
    ;;
esac

ROOTS=(
  "${OUTPUTS_ROOT}/mquake_sure_v7_smoke_seed1_fix1"
  "${OUTPUTS_ROOT}/mquake_sure_v71_utility_safe_seed1"
  "${OUTPUTS_ROOT}/mquake_sure_v72_active_rows_seed1"
)

found=0
for root in "${ROOTS[@]}"; do
  [[ -d "${root}" ]] || continue
  while IFS= read -r -d '' checkpoint; do
    found=1
    if [[ "${MODE}" == "--dry-run" ]]; then
      echo "WOULD DELETE: ${checkpoint}"
      du -sh "${checkpoint}" 2>/dev/null || true
    else
      echo "DELETING: ${checkpoint}"
      rm -rf -- "${checkpoint}"
    fi
  done < <(find "${root}" -type d -name checkpoint -print0)
done

if [[ "${found}" == "0" ]]; then
  echo "No old MQuAKE V7/V7.1/V7.2 checkpoint directories found under ${OUTPUTS_ROOT}."
fi

if [[ "${MODE}" == "--dry-run" ]]; then
  echo "Dry run only. Re-run with --execute to delete the listed checkpoint directories."
else
  echo "Old MQuAKE V7/V7.1/V7.2 checkpoint cleanup complete. Diagnostics were preserved."
fi
