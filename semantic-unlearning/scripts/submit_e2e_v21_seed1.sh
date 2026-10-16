#!/bin/bash
# End-to-end run, MCF seed 1: the v2.1 factorized router (learned subject at layer 8 x
# relation at layers 19+23+27) in front of NEWLY TRAINED question-side Δe rows.
# Same training recipe as E_L16_query_logit (write layer 16, query mode, output + logit-lens
# feedback), so the only change is the router: it decides which training views are routed
# and when the rows fire at test time.
#
#   bash scripts/submit_e2e_v21_seed1.sh                 (v2.1, subject layer 8, the default)
#   SUBJECT_LAYER=6 bash scripts/submit_e2e_v21_seed1.sh
#   ROUTER_ROOT=outputs/mcf_factorized_router_v22_seed1 TAG=v22 SUBJECT_LAYER=8 bash scripts/submit_e2e_v21_seed1.sh
# Jobs: 1) train + official MCF eval + routed generation + held-out leak eval
#       2) same-subject collateral   3) lens eval        (both after 1)
# Then: bash scripts/e2e_summary_v21.sh
set -euo pipefail
ROOT=/scratch/yl258/kp759/Unlearning/semantic-unlearning
cd "$ROOT"
L="${SUBJECT_LAYER:-8}"
TAG="${TAG:-v21}"
ROUTER_ROOT="${ROUTER_ROOT:-outputs/mcf_factorized_router_v21_seed1}"
[[ "$ROUTER_ROOT" = /* ]] || ROUTER_ROOT="$ROOT/$ROUTER_ROOT"
ROUTER="$(ls -d "$ROUTER_ROOT"/S"$(printf '%02d' "$L")"_R* | head -1)"
test -f "$ROUTER/fact_association_embeddings.pt" || { echo "no $TAG router for subject layer $L in $ROUTER_ROOT" >&2; exit 2; }
SUFFIX="_query_logit_${TAG}S$L"
echo "router: $ROUTER -> arm E_L16$SUFFIX"
J=$(sbatch --parsable --array=0 \
  --export=ALL,LAYERS=16,WRITE_MODE=query,NAME_SUFFIX="$SUFFIX",ROUTER_SRC="$ROUTER",EXTRA_TRAIN_ARGS="--lens-type logit" \
  mcf_early_write_seed1.slurm)
C=$(sbatch --parsable --dependency=afterok:"$J" mcf_subject_collateral_seed1.slurm)
E=$(sbatch --parsable --dependency=afterok:"$J" mcf_lens_eval_seed1.slurm)
echo "train+eval $J | collateral $C | lens $E"
