#!/bin/bash
export ZSRE_STEPS=600
export ZSRE_BATCH_SIZE=1
export ZSRE_CACHE_BATCH_SIZE=8
export ZSRE_EMB_LM_LR=0.0001
export ZSRE_GA_WEIGHT=2.0
export ZSRE_GD_WEIGHT=1.0
export OUTPUT_ROOT=${OUTPUT_ROOT:-outputs/aws_zsre_sure_no_neutral_gagd_aggressive_3b}
exec bash "$(dirname "$0")/run_zsre_sure_no_neutral_gagd.sh" "$@"
