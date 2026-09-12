#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/scratch/yl258/kp759/hf/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95}"
OURS_RUN="${OURS_RUN:-$PWD/outputs/static_overlap_fact_association_embeddings_v1_hierarchical_seed1}"
MCF_PATH="${MCF_PATH:-$PWD/data/multi_counterfact.json}"
WIKIDATA_DIR="${WIKIDATA_DIR:-$PWD/data/wikidata}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/outputs/mcf_seed1_fact_assoc_matched_ours_only}"

python -u scripts/evaluate_mcf_seed1_fact_association_matched.py \
  --model-path "${MODEL_PATH}" \
  --ours-run-dir "${OURS_RUN}" \
  --mcf-path "${MCF_PATH}" \
  --wikidata-dir "${WIKIDATA_DIR}" \
  --output-dir "${OUTPUT_DIR}"
