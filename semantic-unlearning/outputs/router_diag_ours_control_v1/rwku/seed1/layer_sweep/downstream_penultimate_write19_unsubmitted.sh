#!/bin/bash
# rwku seed 1: routing improves at the penultimate layer (router_diag_staged.slurm, 2026-11-01 18:50:15)
set -euo pipefail
cd /scratch/yl258/kp759/Unlearning/semantic-unlearning
TAG=onepass_router26_write19_v1 ONEPASS=1 WRITE_LAYER=19 ROUTER_LAYER=26 bash scripts/run_l16_paper_one.sh rwku 1
