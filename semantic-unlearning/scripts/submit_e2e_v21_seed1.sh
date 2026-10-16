#!/bin/bash
# End-to-end run, MCF seed 1: a factorized router (learned subject x relation) in front of
# NEWLY TRAINED question-side Δe rows. Same training recipe as E_L16_query_logit (query mode,
# output + logit-lens feedback at layer 24), so the router and the write layer are the only
# things that change.
#
#   bash scripts/submit_e2e_v21_seed1.sh                 (v2.1, subject layer 8, write layer 16)
#   SUBJECT_LAYER=6 bash scripts/submit_e2e_v21_seed1.sh
#   ROUTER_ROOT=outputs/mcf_factorized_router_v22_seed1 TAG=v22 SUBJECT_LAYER=6 bash scripts/submit_e2e_v21_seed1.sh
# Write-layer sweep (one array task per layer; arms E_L<w>_query_logit_<TAG>S<L>; an arm that
# already finished is skipped by every step, so the existing L16 arm is reused):
#   ROUTER_ROOT=outputs/mcf_factorized_router_v22_seed1 TAG=v22 SUBJECT_LAYER=6 \
#     WRITE_LAYERS="12 16 19" bash scripts/submit_e2e_v21_seed1.sh
# Jobs: 1) train + official MCF eval + routed generation + held-out leak eval (array)
#       2) same-subject collateral   3) lens eval        (both after all of 1)
#       4) sweep only: lens eval of base + the swept arms at layers 8..27
#          -> outputs/mcf_write_layer_sweep_<TAG>S<L>_seed1/lens
# Then: bash scripts/e2e_summary_v21.sh                      (single write layer)
#       python scripts/write_layer_sweep_summary.py --tag <TAG> --subject-layer <L> --layers 12 16 19
set -euo pipefail
ROOT=/scratch/yl258/kp759/Unlearning/semantic-unlearning
cd "$ROOT"
L="${SUBJECT_LAYER:-8}"
TAG="${TAG:-v21}"
read -r -a WL <<< "${WRITE_LAYERS:-16}"
ROUTER_ROOT="${ROUTER_ROOT:-outputs/mcf_factorized_router_v21_seed1}"
[[ "$ROUTER_ROOT" = /* ]] || ROUTER_ROOT="$ROOT/$ROUTER_ROOT"
ROUTER="$(ls -d "$ROUTER_ROOT"/S"$(printf '%02d' "$L")"_R* | head -1)"
test -f "$ROUTER/fact_association_embeddings.pt" || { echo "no $TAG router for subject layer $L in $ROUTER_ROOT" >&2; exit 2; }
SUFFIX="_query_logit_${TAG}S$L"
ARMS=""
for W in "${WL[@]}"; do ARMS+="E_L$W$SUFFIX "; done
echo "router: $ROUTER -> arms: $ARMS"
J=$(sbatch --parsable --array=0-$(( ${#WL[@]} - 1 )) \
  --export=ALL,LAYERS="${WL[*]}",WRITE_MODE=query,NAME_SUFFIX="$SUFFIX",ROUTER_SRC="$ROUTER",EXTRA_TRAIN_ARGS="--lens-type logit" \
  mcf_early_write_seed1.slurm)
C=$(sbatch --parsable --dependency=afterok:"$J" mcf_subject_collateral_seed1.slurm)
E=$(sbatch --parsable --dependency=afterok:"$J" mcf_lens_eval_seed1.slurm)
echo "train+eval $J | collateral $C | lens $E"
if (( ${#WL[@]} > 1 )); then
  S=$(sbatch --parsable --dependency=afterok:"$J" \
    --export=ALL,ARMS="$ARMS",OUT_DIR="outputs/mcf_write_layer_sweep_${TAG}S${L}_seed1/lens",LENS_ARGS="--logit-layers 8 10 12 14 16 18 19 20 22 24 26 27" \
    mcf_lens_eval_seed1.slurm)
  echo "sweep lens (layers 8..27) $S"
  echo "then: python scripts/write_layer_sweep_summary.py --tag $TAG --subject-layer $L --layers ${WL[*]}"
fi
