#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${script_dir}/.." && pwd)"
cd "${project_dir}"

if [[ "$#" -ne 3 ]]; then
  echo "Usage: bash scripts/run_rwku_sure_head_only_w1k.sh MODEL STEPHEN_KING_ATOMIC_CORPUS_DIR REAL_WIKIPEDIA_DIR" >&2
  exit 2
fi

model="$1"
corpus_dir="$2"
real_wikipedia_dir="$3"
python_bin="${PYTHON_BIN:-python}"
experiment_id="rwku-h-w1k-stephen-king-atomic-seed0-v1"
configuration="config/rwku/sure_head_only_w1k_seed0.json"
training_bundle="${corpus_dir}/generated_training_bundle.json"
generator_receipt="${corpus_dir}/generator_receipt.json"
output_root="${RWKU_H_W1K_OUTPUT_ROOT:-outputs/rwku_h_w1k}"
cache_root="${RWKU_H_W1K_CACHE_ROOT:-outputs/sure_wikipedia_stats/real_wikipedia}"
rwku_data_root="${RWKU_DATA_ROOT:-data/rwku}"
wikidata_dir="${WIKIDATA_DIR:-${real_wikipedia_dir}}"
model_tag="$(basename "${model}")"
utility_cache="${RWKU_H_W1K_UTILITY_CACHE:-${cache_root}/${model_tag}_rwku_stephen_king_excluded_docs1000_candidates100000_v1.pt}"
run_dir="${output_root}/${experiment_id}"

test -d "${model}"
test -d "${corpus_dir}"
test -f "${training_bundle}"
test -f "${generator_receipt}"
test -d "${real_wikipedia_dir}"
test -f "${configuration}"

if [[ -e "${run_dir}" ]]; then
  echo "Refusing to overwrite the atomic RWKU-H-W1K run: ${run_dir}" >&2
  echo "Choose a new RWKU_H_W1K_OUTPUT_ROOT; never reuse an experiment receipt." >&2
  exit 2
fi

if [[ ! -f "${utility_cache}" ]]; then
  mkdir -p "$(dirname "${utility_cache}")"
  "${python_bin}" scripts/build_sure_wikipedia_stats.py \
    --model-path "${model}" \
    --wikidata-dir "${real_wikipedia_dir}" \
    --output-path "${utility_cache}" \
    --sample-size 1000 \
    --require-min-documents 1000 \
    --require-min-prompts 90000 \
    --require-corpus-protocol sure_external_wikipedia_corpus_v1 \
    --utility-seed 1 \
    --exclude-first 20 \
    --exclude-casefold-substring "Stephen King" \
    --utility-max-length 4096 \
    --utility-batch-size 1 \
    --utility-prompt-count 100000 \
    --utility-logit-batch-size 64 \
    --dtype bf16 \
    --device-map single
fi

download_args=()
if [[ "${RWKU_NO_DOWNLOAD:-0}" == "1" ]]; then
  download_args=(--no-download)
elif [[ "${RWKU_NO_DOWNLOAD:-0}" != "0" ]]; then
  echo "RWKU_NO_DOWNLOAD must be 0 or 1" >&2
  exit 2
fi

# Preparation records immutable target-only bundle identities and creates only
# a descriptor of official RWKU data; it does not open official record files.
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

# The learner has no RWKU data-root, official-probe, neighbor, MIA, or utility
# benchmark argument.  Its only utility input is target-excluded Wikipedia.
"${python_bin}" scripts/rwku_sure_head_only_w1k.py \
  --model-path "${model}" \
  --training-bundle "${training_bundle}" \
  --generator-receipt "${generator_receipt}" \
  --utility-cache "${utility_cache}" \
  --output-root "${output_root}" \
  --experiment-id "${experiment_id}" \
  --configuration "${configuration}"

# Only the generic evaluator can cross the frozen receipt boundary.  It opens
# the native Level 1/2/3, adversarial, MIA, neighbor, utility, fluency, and PPL
# inputs after verifying every checkpoint and implementation identity.
"${python_bin}" scripts/rwku_experiment.py \
  --seed 0 \
  --stage evaluate \
  --training-source target_only_generated_entity_corpus \
  --experiment-id "${experiment_id}" \
  --model-path "${model}" \
  --output-root "${output_root}" \
  --data-root "${rwku_data_root}" \
  --wikidata-dir "${wikidata_dir}" \
  --dtype bf16 \
  --eval-batch-size "${RWKU_EVAL_BATCH_SIZE:-4}" \
  "${download_args[@]}"

echo "RWKU-H-W1K Stephen King head-only feasibility run complete: ${run_dir}"
