#!/bin/bash
# rwku seed 1: routing improves at the penultimate layer; the paper's method with the router at 26 (pass 1) and Δe at 16 (pass 2)
set -euo pipefail
cd /scratch/yl258/kp759/Unlearning/semantic-unlearning
TAG=l16_ours_router26_v1 ONEPASS=1 WRITE_LAYER=16 ROUTER_LAYER=26 bash scripts/run_l16_paper_one.sh rwku 1
