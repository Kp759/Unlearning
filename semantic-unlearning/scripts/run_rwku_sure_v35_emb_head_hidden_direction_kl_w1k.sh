#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${script_dir}/.." && pwd)"
cd "${project_dir}"

if [[ "$#" -ne 4 ]]; then
  echo "Usage: bash scripts/run_rwku_sure_v35_emb_head_hidden_direction_kl_w1k.sh MODEL STEPHEN_KING_ATOMIC_CORPUS_DIR REAL_WIKIPEDIA_DIR SOURCE_HEAD_ONLY_RUN_DIR" >&2
  exit 2
fi

model="$1"
corpus_dir="$2"
real_wikipedia_dir="$3"
source_head_only_run="$4"
python_bin="${PYTHON_BIN:-python}"
experiment_id="rwku-h-w1k-stephen-king-emb-head-hidden-direction-seed0-v35-kl"
configuration="config/rwku/sure_v35_emb_head_hidden_direction_kl_w1k_seed0.json"
training_bundle="${corpus_dir}/generated_training_bundle.json"
generator_receipt="${corpus_dir}/generator_receipt.json"
output_root="${RWKU_V35_OUTPUT_ROOT:-outputs/rwku_v35_emb_head_kl_w1k}"
cache_root="${RWKU_H_W1K_CACHE_ROOT:-outputs/sure_wikipedia_stats/real_wikipedia}"
rwku_data_root="${RWKU_DATA_ROOT:-data/rwku}"
model_tag="$(basename "${model}")"
utility_cache="${RWKU_H_W1K_UTILITY_CACHE:-${cache_root}/${model_tag}_rwku_stephen_king_excluded_docs1000_candidates100000_v1.pt}"
run_dir="${output_root}/${experiment_id}"

test -d "${model}"
test -d "${corpus_dir}"
test -f "${training_bundle}"
test -f "${generator_receipt}"
test -d "${real_wikipedia_dir}"
test -d "${source_head_only_run}"
test -f "${source_head_only_run}/sure_head_only_w1k/stage1_delta.pt"
test -f "${source_head_only_run}/sure_head_only_w1k/infeasible.json"
test -f "${utility_cache}"
test -f "${configuration}"

if [[ -e "${run_dir}" ]]; then
  echo "Refusing to overwrite v3.5 run: ${run_dir}" >&2
  echo "Choose a new RWKU_V35_OUTPUT_ROOT for a new run." >&2
  exit 2
fi

download_args=()
if [[ "${RWKU_NO_DOWNLOAD:-0}" == "1" ]]; then
  download_args=(--no-download)
elif [[ "${RWKU_NO_DOWNLOAD:-0}" != "0" ]]; then
  echo "RWKU_NO_DOWNLOAD must be 0 or 1" >&2
  exit 2
fi

"${python_bin}" scripts/rwku_experiment.py \
  --seed 0 \
  --stage prepare \
  --training-source target_only_generated_entity_corpus \
  --experiment-id "${experiment_id}" \
  --model-path "${model}" \
  --output-root "${output_root}" \
  --data-root "${rwku_data_root}" \
  --generated-entity-fact-bundle "${training_bundle}" \
  --generator-receipt "${generator_receipt}" \
  "${download_args[@]}"

"${python_bin}" scripts/rwku_sure_v35_emb_head_hidden_direction_kl_w1k.py \
  --model-path "${model}" \
  --training-bundle "${training_bundle}" \
  --generator-receipt "${generator_receipt}" \
  --utility-cache "${utility_cache}" \
  --wikipedia-dir "${real_wikipedia_dir}" \
  --source-head-only-run "${source_head_only_run}" \
  --output-root "${output_root}" \
  --experiment-id "${experiment_id}" \
  --configuration "${configuration}" \
  --save-checkpoint

echo "RWKU v3.5 development run complete: ${run_dir}"
echo "Trainable: input embeddings + untied LM head + final-layer down_proj LoRA."
echo "Frozen probe: immutable original W0."
echo "Selection excludes the already-opened v3.2 1K guard; fresh 1K confirmatory utility opens only after selection."
