#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL="${1:?Usage: bash scripts/run_mcf_sure_v9_gfs_recovery.sh MODEL MCF_JSON REAL_WIKIPEDIA_DIR}"
MCF="${2:?Pass the CounterFact JSON path}"
REAL_WIKIPEDIA_DIR="${3:?Pass the prepared real-Wikipedia DatasetDict path}"
DOCS="${SURE_V9_WIKIPEDIA_DOCS:-10000}"

# Hold the successful W10K locality treatment fixed.  The only change from the
# v9 legacy augmentation is the locked prompt-view profile: four answer-cued
# same-subject GA/GD views paired with structure-matched, external-Wikipedia
# subject views under exact KL.  Official GFS and Spe prompts remain unavailable
# until the checkpoint has been frozen by the underlying runner.
export OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/mcf_sure_v9_gfs_recovery_w${DOCS}}"
export SURE_V9_WIKIPEDIA_DOCS="${DOCS}"
export SURE_EXTERNAL_CONTEXT_PROFILE="paired_answer_cue_v1"

bash scripts/run_mcf_sure_v9_aug.sh \
  "${MODEL}" \
  "${MCF}" \
  "${REAL_WIKIPEDIA_DIR}"
