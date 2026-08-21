#!/usr/bin/env bash
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL="${1:?Usage: bash scripts/run_mcf_sure_v9_aug.sh MODEL MCF_JSON REAL_WIKIPEDIA_DIR}"
MCF="${2:?Pass the CounterFact JSON path}"
REAL_WIKIPEDIA_DIR="${3:?Pass the prepared real-Wikipedia DatasetDict path}"
DOCS="${SURE_V9_WIKIPEDIA_DOCS:-10000}"
CANDIDATES="${SURE_UTILITY_PROMPT_COUNT:-100000}"
MIN_PROMPTS="${SURE_V9_MIN_UTILITY_PROMPTS:-90000}"
MODEL_TAG="$(basename "${MODEL}")"

export OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/mcf_sure_v9_aug_w${DOCS}}"
export SURE_ENABLE_EXTERNAL_CONTEXTS=1
export SURE_UTILITY_WIKIPEDIA_DIR="${REAL_WIKIPEDIA_DIR}"
export SURE_UTILITY_SAMPLE_SIZE="${DOCS}"
export SURE_UTILITY_PROMPT_COUNT="${CANDIDATES}"
export SURE_MIN_UTILITY_DOCUMENTS="${DOCS}"
export SURE_MIN_UTILITY_PROMPTS="${MIN_PROMPTS}"
export SURE_REQUIRED_UTILITY_CORPUS_PROTOCOL="sure_external_wikipedia_corpus_v1"
export SURE_UTILITY_CACHE="${SURE_UTILITY_CACHE:-outputs/sure_wikipedia_stats/real_wikipedia/${MODEL_TAG}_realwiki_docs${DOCS}_candidates${CANDIDATES}_v3.pt}"

bash scripts/run_mcf_sure_target_aware_direct_only.sh "${MODEL}" "${MCF}"
