#!/usr/bin/env bash
set -euo pipefail
export ZSRE_STEPS=450
export ZSRE_BATCH_SIZE=1
export ZSRE_CACHE_BATCH_SIZE=8
export ZSRE_EMB_LM_LR=0.000075
export ZSRE_GA_WEIGHT=1.75
export ZSRE_GD_WEIGHT=1.0
export OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/aws_zsre_sure_no_neutral_gagd_mid_3b}
exec bash "$(dirname "$0")/run_zsre_sure_no_neutral_gagd.sh" "$@"
