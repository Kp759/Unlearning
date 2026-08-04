#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${script_dir}/.." && pwd)"

python_bin="${PYTHON_BIN:-python}"
model_path="${MODEL_PATH:-/scratch/yl258/kp759/hf/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95}"
output_root="${RWKU_OUTPUT_ROOT:-${project_dir}/outputs/rwku_v2}"
data_root="${RWKU_DATA_ROOT:-${project_dir}/data/rwku}"
mcf_path="${MCF_PATH:-${project_dir}/data/multi_counterfact.json}"
wikidata_dir="${WIKIDATA_DIR:-${project_dir}/data/wikidata}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

"${python_bin}" "${script_dir}/rwku_experiment.py" \
  --seed 0 \
  --methods representation \
  --model-path "${model_path}" \
  --output-root "${output_root}" \
  --data-root "${data_root}" \
  --mcf-path "${mcf_path}" \
  --wikidata-dir "${wikidata_dir}" \
  --no-download \
  --save-checkpoints \
  --dtype bf16 \
  --representation-steps 1800 \
  --representation-rank 24 \
  --representation-alpha 48 \
  --representation-last-n-layers 12 \
  --representation-qa-tasks-per-step 4 \
  --representation-positive-tasks-per-phase 2 \
  --representation-constraint-polish-fraction 0.15 \
  --representation-constraint-polish-limit 96 \
  --representation-answer-probability-target 1e-6 \
  --representation-answer-probability-weight 2 \
  --representation-frozen-head-weight 2 \
  --representation-frozen-head-demotion-margin 1 \
  --representation-layerwise-concept-erasure-weight 1 \
  --representation-layerwise-concept-orthogonal-weight 0.25 \
  --representation-concept-anchor-layer-stride 2 \
  --representation-positive-proxy-weight 1 \
  --representation-retain-kl-weight 10 \
  --representation-retain-answer-weight 2 \
  --representation-retain-hidden-weight 2 \
  --representation-max-retain-kl 0.02 \
  --representation-max-retain-p95-kl 0.05 \
  --representation-min-retain-probability-ratio 0.995 \
  --representation-max-retain-probability-ratio 1.005 \
  --representation-min-retain-p05-probability-ratio 0.95 \
  --representation-max-retain-p95-probability-ratio 1.05 \
  --representation-min-retain-top1-agreement 0.99 \
  --representation-min-retain-hidden-cosine 0.995 \
  --representation-min-retain-p05-hidden-cosine 0.99 \
  --representation-max-retain-hidden-relative-l2 0.10 \
  --representation-max-retain-p95-hidden-relative-l2 0.15 \
  --representation-max-proxy-mia-advantage 0.05 \
  --representation-max-frozen-head-chance-ratio 1 \
  --representation-min-frozen-head-normalized-rank 0.90 \
  --representation-selection-calibration-limit 192 \
  --representation-selection-generation-limit 32
