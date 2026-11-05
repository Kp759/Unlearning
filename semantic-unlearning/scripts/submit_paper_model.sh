#!/bin/bash
# Paper table on another model (run from semantic-unlearning/ on the login node).
#
#   bash scripts/submit_paper_model.sh MODEL fetch   # 1. download the weights (login node has internet)
#   bash scripts/submit_paper_model.sh MODEL smoke   # 2. MCF seed 1: base | ours | genie + 5 baselines
#   bash scripts/submit_paper_model.sh MODEL full    # 3. 4 datasets x 5 seeds (+ 100 baseline tasks)
#   ONEPASS=1 ROUTER_LAYER=R WRITE_LAYER=W bash scripts/submit_paper_model.sh MODEL ours
#                                                    # 3'. base | ours | genie only, with the layers
#                                                    #     picked by scripts/submit_layer_smoke.sh
#
# MODEL = llama31_8b | qwen3_4b (settings: scripts/paper_model_config.sh).
# Baselines wait for the main array (they reuse its routers and base runs). ZeroUnlearn task 4
# (MCF seed 1) runs first and caches the model's Wikipedia covariance; the other ZeroUnlearn
# tasks wait for it. Finished stages are skipped, so `full` after `smoke` redoes nothing.
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="${1:?Usage: submit_paper_model.sh MODEL fetch|smoke|full}"
MODE="${2:?Usage: submit_paper_model.sh MODEL fetch|smoke|full}"
export MODEL
source scripts/paper_model_config.sh

if [[ "$MODE" == fetch ]]; then
  # Llama is gated: accept the licence on huggingface.co and `hf auth login` once.
  if command -v hf >/dev/null; then DL=(hf download); else DL=(huggingface-cli download); fi
  "${DL[@]}" "$HF_ID" --cache-dir "$HF_CACHE" --exclude "original/*"
  unset MODEL_PATH; source scripts/paper_model_config.sh
  SLURM_JOB_ID="" paper_model_check
  exit 0
fi
paper_model_check

case "$MODE" in
  smoke) MAIN=0; SETS=(0) ;;
  full)  MAIN=0-19; SETS=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19) ;;
  ours)  # base | ours | genie only, 4 datasets x 5 seeds, no baselines (the paper's own-method rows)
    M=$(sbatch --parsable --export=ALL,MODEL="$MODEL" --array=0-19 paper_model_multiseed.slurm)
    echo "$MODEL base|ours|genie, 4 datasets x 5 seeds (router L$ROUTER_LAYER, Δe L$WRITE_LAYER): $M"
    echo "when done: python scripts/l16_paper_table.py --root outputs/$TAG --ours-label \"Ours (L$ROUTER_LAYER router, Δe L$WRITE_LAYER)\""
    exit 0 ;;
  *) echo "MODE must be fetch, smoke, ours or full" >&2; exit 2 ;;
esac
# baseline task ids for the chosen (dataset, seed) sets: 5 * set + {0..3} train, + 4 ZeroUnlearn
# (the main array's set s = 5 * dataset + seed - 1 maps to baseline tasks 5s..5s+4)
TRAIN=(); ZERO=()
for s in "${SETS[@]}"; do
  TRAIN+=("$(( 5 * s ))-$(( 5 * s + 3 ))"); ZERO+=("$(( 5 * s + 4 ))")
done
join() { local IFS=,; echo "$*"; }

M=$(sbatch --parsable --export=ALL,MODEL="$MODEL" --array="$MAIN" paper_model_multiseed.slurm)
echo "$MODEL base|ours|genie (array $MAIN): $M"
B=$(sbatch --parsable --export=ALL,MODEL="$MODEL" --dependency=afterany:"$M" \
      --array="$(join "${TRAIN[@]}")" paper_model_baselines.slurm)
echo "$MODEL GA/GD/NPO/RMU: $B"
Z=$(sbatch --parsable --export=ALL,MODEL="$MODEL" --dependency=afterany:"$M" \
      --array=4 paper_model_baselines.slurm)
echo "$MODEL ZeroUnlearn MCF seed 1 (builds the covariance cache): $Z"
if (( ${#ZERO[@]} > 1 )); then
  Z2=$(sbatch --parsable --export=ALL,MODEL="$MODEL" --dependency=afterany:"$Z" \
         --array="$(join "${ZERO[@]:1}")" paper_model_baselines.slurm)
  echo "$MODEL ZeroUnlearn, other sets: $Z2"
fi
echo "when done: python scripts/l16_paper_table.py --root outputs/$TAG"
