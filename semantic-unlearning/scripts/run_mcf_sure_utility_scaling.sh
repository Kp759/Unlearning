#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL="${1:?Usage: bash scripts/run_mcf_sure_utility_scaling.sh MODEL MCF_JSON REAL_WIKIPEDIA_DIR}"
MCF="${2:?Pass the CounterFact JSON path}"
REAL_WIKIPEDIA_DIR="${3:?Pass the prepared real-Wikipedia DatasetDict path}"
DOCS_TEXT="${SURE_SCALING_DOCS:-1000 10000}"
SEEDS_TEXT="${MCF_SEEDS:-1}"
CANDIDATES="${SURE_UTILITY_PROMPT_COUNT:-100000}"
MIN_PROMPTS="${SURE_SCALING_MIN_UTILITY_PROMPTS:-90000}"
CACHE_ROOT="${SURE_SCALING_CACHE_ROOT:-outputs/sure_wikipedia_stats/real_wikipedia}"
RUN_ROOT="${SURE_SCALING_OUTPUT_ROOT:-outputs/mcf_sure_utility_scaling}"
MODEL_TAG="$(basename "${MODEL}")"

test -d "${MODEL}"
test -f "${MCF}"
test -d "${REAL_WIKIPEDIA_DIR}"

read -r -a DOC_COUNTS <<< "${DOCS_TEXT}"
for DOCS in "${DOC_COUNTS[@]}"; do
  if [[ ! "${DOCS}" =~ ^[0-9]+$ ]] || (( DOCS <= 0 )); then
    echo "Invalid Wikipedia document count: ${DOCS}" >&2
    exit 2
  fi
  LABEL="W${DOCS}"
  OUTPUT_ROOT="${RUN_ROOT}/v8-${LABEL}" \
  MCF_SEEDS="${SEEDS_TEXT}" \
  SURE_ENABLE_EXTERNAL_CONTEXTS=0 \
  SURE_UTILITY_WIKIPEDIA_DIR="${REAL_WIKIPEDIA_DIR}" \
  SURE_UTILITY_SAMPLE_SIZE="${DOCS}" \
  SURE_UTILITY_PROMPT_COUNT="${CANDIDATES}" \
  SURE_MIN_UTILITY_DOCUMENTS="${DOCS}" \
  SURE_MIN_UTILITY_PROMPTS="${MIN_PROMPTS}" \
  SURE_REQUIRED_UTILITY_CORPUS_PROTOCOL="sure_external_wikipedia_corpus_v1" \
  SURE_UTILITY_CACHE="${CACHE_ROOT}/${MODEL_TAG}_realwiki_docs${DOCS}_candidates${CANDIDATES}_v3.pt" \
    bash scripts/run_mcf_sure_target_aware_direct_only.sh "${MODEL}" "${MCF}"
done

echo "Pure-v8 Wikipedia scaling experiments complete: ${RUN_ROOT}"
