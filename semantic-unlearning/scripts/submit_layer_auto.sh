#!/usr/bin/env bash
# One submission: layer smoke test, automatic layer choice, then the full paper run.
#
#   bash scripts/submit_layer_auto.sh qwen3_4b
#   bash scripts/submit_layer_auto.sh llama31_8b
#
# Chain (each step waits for the previous one to finish OK; a failure leaves the rest pending,
# Slurm then cancels them as DependencyNeverSatisfied):
#   1. router sweep      GPU   layer_smoke.slurm STAGE=router   MCF seed 1 (DATASETS to add more)
#   2. decide router     CPU   layer_decide.slurm STAGE=router  -> decision.json (rule in layer_decide.py)
#   3. base              GPU   layer_smoke.slurm STAGE=base     router at the chosen R + base eval
#   4. write candidates  GPU   layer_smoke.slurm STAGE=write    rows at {R, R-3, R-5, default W}
#   5. decide write      CPU   layer_decide.slurm STAGE=write   -> decision.json: write_layer, tag
#   6. full run          GPU   paper_model_multiseed.slurm      base | ours | genie, 4 datasets x 5 seeds
#   7. table             CPU   layer_decide.slurm STAGE=table   -> outputs/<tag>/paper_table.md
# Decision file: outputs/layer_smoke_<MODEL>/decision.json. Inspect after step 2 or 5 with
#   python scripts/layer_decide.py MODEL show
# Finished stages are skipped on a resubmit, so rerunning the chain after a fix is cheap.
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="${1:?Usage: submit_layer_auto.sh MODEL}"
DATASETS="${DATASETS:-mcf}"
read -r -a DS_LIST <<< "$DATASETS"
N_DS=${#DS_LIST[@]}
export MODEL ONEPASS=1
export LAYER_DECISION="outputs/layer_smoke_${MODEL}/decision.json"
source scripts/paper_model_config.sh
paper_model_check
MAX_W=4   # write candidates: at most {R, R-3, R-5, default}; extra array tasks exit at once

GPU="ALL,MODEL=$MODEL,ONEPASS=1,DATASETS=$DATASETS,LAYER_DECISION=$LAYER_DECISION"
CPU="ALL,MODEL=$MODEL"
# the router sweep must not read a stale decision.json from an earlier chain: no LAYER_DECISION there
J1=$(sbatch --parsable --export="ALL,MODEL=$MODEL,ONEPASS=1,DATASETS=$DATASETS,STAGE=router" \
       --array="0-$(( N_DS - 1 ))" layer_smoke.slurm)
J2=$(sbatch --parsable --dependency=afterok:"$J1" --export="$CPU,STAGE=router" layer_decide.slurm)
J3=$(sbatch --parsable --dependency=afterok:"$J2" --export="$GPU,STAGE=base" \
       --array="0-$(( N_DS - 1 ))" layer_smoke.slurm)
J4=$(sbatch --parsable --dependency=afterok:"$J3" --export="$GPU,STAGE=write" \
       --array="0-$(( N_DS * MAX_W - 1 ))" layer_smoke.slurm)
J5=$(sbatch --parsable --dependency=afterok:"$J4" --export="$CPU,STAGE=write" layer_decide.slurm)
J6=$(sbatch --parsable --dependency=afterok:"$J5" --export="ALL,MODEL=$MODEL,ONEPASS=1,LAYER_DECISION=$LAYER_DECISION" \
       --array=0-19 paper_model_multiseed.slurm)
J7=$(sbatch --parsable --dependency=afterok:"$J6" --export="$CPU,STAGE=table" layer_decide.slurm)
cat <<EOF
$MODEL automatic layer chain submitted ($DATASETS seed 1 for the smoke test; default router L$ROUTER_LAYER, write L$EARLY_WRITE_LAYER):
  1 router sweep      $J1
  2 decide router     $J2   -> $LAYER_DECISION
  3 base              $J3
  4 write candidates  $J4
  5 decide write      $J5
  6 full run          $J6   -> outputs/paper_${MODEL}_r<R>_w<W>_ours_v1 (4 datasets x 5 seeds)
  7 paper table       $J7
watch: squeue -u \$USER -n layer_smoke,layer_decide,paper_model ; decision: python scripts/layer_decide.py $MODEL show
EOF
