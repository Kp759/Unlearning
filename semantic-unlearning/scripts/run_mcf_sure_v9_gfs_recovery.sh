#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL="${1:?Usage: bash scripts/run_mcf_sure_v9_gfs_recovery.sh MODEL MCF_JSON REAL_WIKIPEDIA_DIR}"
MCF="${2:?Pass the CounterFact JSON path}"
REAL_WIKIPEDIA_DIR="${3:?Pass the prepared real-Wikipedia DatasetDict path}"
DOCS="${SURE_V9_WIKIPEDIA_DOCS:-10000}"
PPL_WIKIPEDIA_DIR="${WIKIDATA_DIR:-data/wikidata}"
SEEDS_TEXT="${MCF_SEEDS:-1}"
CANDIDATES="${SURE_UTILITY_PROMPT_COUNT:-100000}"

# Hold the successful W10K locality treatment fixed.  The only change from the
# v9 legacy augmentation is the locked prompt-view profile: four answer-cued
# same-subject GA/GD views paired with structure-matched, external-Wikipedia
# subject views under exact KL.  Official GFS and Spe prompts remain unavailable
# until the checkpoint has been frozen by the underlying runner.
if [[ "${DOCS}" != "10000" ]]; then
  echo "paired_context_recovery is locked to 10000 Wikipedia documents" >&2
  exit 2
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/mcf_sure_v9_gfs_recovery_w${DOCS}}"
read -r -a SEEDS <<< "${SEEDS_TEXT}"
UTILITY_CACHE_ARGS=()
if [[ -n "${SURE_UTILITY_CACHE:-}" ]]; then
  UTILITY_CACHE_ARGS=(--utility-cache "${SURE_UTILITY_CACHE}")
fi

python -u scripts/MCF_Scripts/run_mcf_sure_two_stage.py \
  --model-path "${MODEL}" \
  --mcf-path "${MCF}" \
  --wikipedia-dir "${PPL_WIKIPEDIA_DIR}" \
  --utility-wikipedia-dir "${REAL_WIKIPEDIA_DIR}" \
  --treatment paired_context_recovery \
  --utility-docs "${DOCS}" \
  --utility-prompts "${CANDIDATES}" \
  --require-corpus-protocol sure_external_wikipedia_corpus_v1 \
  --output-root "${OUTPUT_ROOT}" \
  --seeds "${SEEDS[@]}" \
  "${UTILITY_CACHE_ARGS[@]}"
