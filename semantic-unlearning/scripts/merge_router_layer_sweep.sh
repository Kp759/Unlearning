#!/bin/bash
# One table from every finished layer of mcf_router_layer_sweep_seed1.slurm.
set -euo pipefail
ROOT="${ROOT:-/scratch/yl258/kp759/Unlearning/semantic-unlearning}"
SWEEP="$ROOT/outputs/mcf_router_layer_sweep_seed1"
shopt -s nullglob
FILES=("$SWEEP"/L*/specificity.json)
if (( ${#FILES[@]} == 0 )); then echo "no finished layers yet"; exit 0; fi
python "$ROOT/scripts/mcf_router_specificity.py" merge "$SWEEP/summary.md" "${FILES[@]}"
