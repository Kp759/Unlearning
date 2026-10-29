#!/usr/bin/env bash
# Representation recoverability only (stage 6 of run_l16_paper_one.sh) on finished run dirs of a
# paper-table TAG: base, baselines, ours, genie. Every other stage is skipped because its output
# exists; a run dir without its official eval is left alone (nothing to measure yet).
#
#   bash scripts/run_recoverability_one.sh TAG DATASET SEED          TAG e.g. onepass_l19_v1
#   BASELINES="ga npo" bash scripts/run_recoverability_one.sh onepass_l19_v1 mcf 1
#   RECOVER_ARGS="--layers 8,12,16,19,24,27" ...                       # a subset of blocks
# Other models: MODEL=qwen3_4b ONEPASS=1 bash scripts/run_recoverability_one.sh paper_qwen3_4b_onepass_v1 mcf 1
#   (scripts/paper_model_config.sh is sourced for MODEL_PATH / ROUTER_LAYER / BASELINE_ARGS)
# The TAG decides the configuration the stage-4 staleness check expects:
#   onepass_l19_v1 -> ONEPASS=1 ; l16_ours_v1 -> ONEPASS=1 WRITE_LAYER=16 ; l16_paper_v1 -> defaults.
set -euo pipefail
TAG="${1:?Usage: run_recoverability_one.sh TAG DATASET SEED}"
DS="${2:?Usage: run_recoverability_one.sh TAG DATASET SEED}"
SEED="${3:?Usage: run_recoverability_one.sh TAG DATASET SEED}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${MODEL:-}" ]]; then
  # shellcheck source=paper_model_config.sh
  source "$HERE/paper_model_config.sh"
else
  case "$TAG" in
    onepass_l*_v1)  export ONEPASS=1 ;;
    l16_ours_v1)    export ONEPASS=1 WRITE_LAYER=16 ;;
    l16_paper_v1)   ;;
    *) [[ -n "${ONEPASS:-}" ]] || { echo "unknown TAG $TAG: set ONEPASS / WRITE_LAYER / ROUTER_RECIPE yourself" >&2; exit 2; } ;;
  esac
fi
export TAG RECOVERABILITY=1 BUILD_ROUTERS=0 MOVE_INCOMPLETE=0
exec bash "$HERE/run_l16_paper_one.sh" "$DS" "$SEED"
