#!/usr/bin/env bash
# Router diagnostics for all four benchmarks (router_diag.slurm, one task per benchmark).
#   bash scripts/submit_router_diag.sh            # seed 1
#   SEED=2 bash scripts/submit_router_diag.sh
# -> outputs/router_diag_v1/<ds>/seed<S>/report/summary.md        L19 error analysis
#    outputs/router_diag_v1/<ds>/seed<S>/layer_sweep/layer_choice.md
#    outputs/router_diag_v1/ROUTER_LAYER_CHOICE.md (+ errors_vs_layer_all.png), written by
#    whichever task finishes last
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
SEED="${SEED:-1}"
ROOT_OUT="${ROOT_OUT:-outputs/router_diag_v1}"
for DS in mcf zsre mquake rwku; do
  case "$DS" in
    mcf) R="outputs/mcf_multiseed_regular_v1/seed$SEED/L19/router" ;;
    zsre) R="outputs/zsre_multiseed_reworded_v2/seed$SEED/L19/router" ;;
    mquake) R="outputs/mquake_multiseed_regular_v1/seed$SEED/L19/router" ;;
    rwku) R="outputs/rwku_multiseed_subject_v1/seed$SEED/L19/router" ;;
  esac
  [[ -f "$R/fact_association_embeddings.pt" ]] || echo "WARNING: $DS router missing: $R (that task will fail)" >&2
done
sbatch --export=ALL,SEED="$SEED",ROOT_OUT="$ROOT_OUT" router_diag.slurm
