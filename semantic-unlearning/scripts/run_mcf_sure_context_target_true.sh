#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "DEPRECATED: run_mcf_sure_context_target_true.sh used an MCF-only reference-answer CE/GD term." >&2
echo "Forwarding to the fixed MCF/ZsRE shared architecture instead." >&2
exec bash scripts/run_mcf_sure_fixed_shared.sh "$@"
