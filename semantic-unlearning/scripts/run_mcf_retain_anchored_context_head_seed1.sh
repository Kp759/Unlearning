#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

: "${MODEL_PATH:?Set MODEL_PATH to the local Llama-3.2-3B-Instruct snapshot}"

MCF_PATH="${MCF_PATH:-${ROOT}/data/multi_counterfact.json}"
WIKIDATA_DIR="${WIKIDATA_DIR:-${ROOT}/data/wikidata}"
OUT_DIR="${OUT_DIR:-${ROOT}/results/retain_anchored_context_head/mcf/seed1_context_only}"
DTYPE="${DTYPE:-bf16}"
DESCRIPTOR_DIM="${DESCRIPTOR_DIM:-32}"
RADIUS="${RADIUS:-1.0}"
LOGIT_PENALTY="${LOGIT_PENALTY:-12.0}"
RETAIN_EVENTS_PER_RECORD="${RETAIN_EVENTS_PER_RECORD:-1}"
EXTRACT_BATCH_SIZE="${EXTRACT_BATCH_SIZE:-16}"
ALPHA_CHUNK_SIZE="${ALPHA_CHUNK_SIZE:-256}"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${ROOT}:${ROOT}/scripts${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${OUT_DIR}"

python scripts/mcf_retain_anchored_context_head_seed1.py \
  --model-path "${MODEL_PATH}" \
  --mcf-path "${MCF_PATH}" \
  --wikidata-dir "${WIKIDATA_DIR}" \
  --output-dir "${OUT_DIR}" \
  --seed 1 \
  --forget-num 50 \
  --retain-num 1000 \
  --dtype "${DTYPE}" \
  --device cuda \
  --descriptor-dim "${DESCRIPTOR_DIM}" \
  --radius "${RADIUS}" \
  --logit-penalty "${LOGIT_PENALTY}" \
  --retain-events-per-record "${RETAIN_EVENTS_PER_RECORD}" \
  --extract-batch-size "${EXTRACT_BATCH_SIZE}" \
  --alpha-chunk-size "${ALPHA_CHUNK_SIZE}" \
  --eval-base

echo
echo "Seed-1 summary: ${OUT_DIR}/seed1_summary.json"
