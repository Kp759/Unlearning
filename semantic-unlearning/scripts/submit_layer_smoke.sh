#!/usr/bin/env bash
# Layer smoke test on another model, before its 4-dataset x 5-seed paper run (layer_smoke.slurm).
#
#   bash scripts/submit_layer_smoke.sh MODEL router            # 1. classifier quality at every block
#   python scripts/layer_smoke_table.py MODEL                   #    -> pick the router layer R
#   bash scripts/submit_layer_smoke.sh MODEL write R [W,W,...]  # 2. rows at W under the R router:
#                                                               #    Eff / Gen / Spe / PPL per W
#   python scripts/layer_smoke_table.py MODEL                   #    -> pick the write layer W
#   ONEPASS=1 ROUTER_LAYER=R WRITE_LAYER=W bash scripts/submit_paper_model.sh MODEL ours
#
# MODEL = qwen3_4b | llama31_8b. MCF seed 1 by default (DATASETS="mcf zsre" adds ZsRE; SEED=1).
# Default write candidates: R, R-3, R-5 (the 3B model uses router 19 and Δe at 16 = R-3).
# The router stage fits at the model's default router layer (scripts/paper_model_config.sh;
# ROUTER_LAYER in the environment overrides it); the refit at every block does not depend on it.
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="${1:?Usage: submit_layer_smoke.sh MODEL router | write R [W,W,...]}"
MODE="${2:?Usage: submit_layer_smoke.sh MODEL router | write R [W,W,...]}"
DATASETS="${DATASETS:-mcf}"
read -r -a DS_LIST <<< "$DATASETS"
N_DS=${#DS_LIST[@]}
export MODEL
source scripts/paper_model_config.sh
paper_model_check

case "$MODE" in
  router)
    J=$(sbatch --parsable --export=ALL,MODEL="$MODEL",STAGE=router,DATASETS="$DATASETS" \
          --array="0-$(( N_DS - 1 ))" layer_smoke.slurm)
    echo "$MODEL router sweep ($DATASETS; router fit at L$ROUTER_LAYER, refit at every block): $J"
    echo "then: python scripts/layer_smoke_table.py $MODEL" ;;
  write)
    R="${3:?write needs the router layer R}"
    W="${4:-$R,$(( R - 3 )),$(( R - 5 ))}"
    IFS=, read -r -a W_LIST <<< "$W"
    (( R > 0 && R < N_LAYERS )) || { echo "R=$R outside 1..$(( N_LAYERS - 1 ))" >&2; exit 2; }
    for w in "${W_LIST[@]}"; do
      (( w > 0 && w <= R )) || { echo "write layer $w must be in 1..R=$R" >&2; exit 2; }
    done
    A=$(sbatch --parsable --export=ALL,MODEL="$MODEL",STAGE=base,ROUTER_LAYER="$R",DATASETS="$DATASETS" \
          --array="0-$(( N_DS - 1 ))" layer_smoke.slurm)
    echo "$MODEL router L$R + base ($DATASETS): $A"
    B=$(sbatch --parsable --dependency=afterok:"$A" \
          --export=ALL,MODEL="$MODEL",STAGE=write,ROUTER_LAYER="$R",WRITE_LAYERS="$W",DATASETS="$DATASETS" \
          --array="0-$(( N_DS * ${#W_LIST[@]} - 1 ))" layer_smoke.slurm)
    echo "$MODEL rows at L{$W} under the L$R router, ours + genie ($DATASETS): $B"
    echo "then: python scripts/layer_smoke_table.py $MODEL" ;;
  *) echo "MODE must be router or write" >&2; exit 2 ;;
esac
