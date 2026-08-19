#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODE="${1:---dry-run}"
if [[ "${MODE}" != "--dry-run" && "${MODE}" != "--delete" ]]; then
  echo "Usage: bash scripts/cleanup_mcf_superseded_checkpoints.sh [--dry-run|--delete]" >&2
  exit 2
fi

# Preserve all JSON/CSV/MD/log/provenance artifacts.  Delete only large model
# checkpoint directories from seed-1 MCF ablations that are superseded by the
# canonical rank-8 / margin-1.0 configuration.
TARGETS=(
  "outputs/mcf_best_run_target_true_sensitive/seed1/repair_best_run_mirrored/checkpoint"
  "outputs/mcf_best_run_tt_rank8_seed1/repair/checkpoint"
  "outputs/mcf_best_run_tt_rank8_margin2_seed1/repair/checkpoint"
)

echo "Superseded MCF checkpoint cleanup (${MODE})"
for path in "${TARGETS[@]}"; do
  if [[ ! -e "${path}" ]]; then
    echo "MISSING  ${path}"
    continue
  fi
  size="$(du -sh "${path}" 2>/dev/null | awk '{print $1}')"
  echo "${size:-?}  ${path}"
  if [[ "${MODE}" == "--delete" ]]; then
    rm -rf -- "${path}"
    echo "DELETED  ${path}"
  fi
done

cat <<'EOF'

Preserved intentionally:
  outputs/mcf_best_run_tt_rank8_margin1_seed1/repair/checkpoint
    - current best seed-1 checkpoint until the new canonical run finishes.
  outputs/mcf_best_run_target_true_sensitive/seed1/setting5e_best_run_mirrored/.../checkpoint
    - reusable Stage-1 checkpoint / diagnostic provenance.
  all eval JSON, repair_summary.json, manifests, logs, CSV, and markdown files.

After outputs/mcf_sure_rome_target_true_r8m1/seed1 finishes successfully, the
old margin-1 and old Stage-1 checkpoints can also be removed if desired.
EOF
