#!/bin/bash
# mcf seed 1: routing improves at the penultimate layer (router_diag_staged.slurm, 2026-10-23 08:56:42)
set -euo pipefail
cd /scratch/yl258/kp759/Unlearning/semantic-unlearning
TAG=l16_paper_router26_v1 MCF_ROUTER=linear ROUTER_LAYER=26 bash scripts/run_l16_paper_one.sh mcf 1
