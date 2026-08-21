#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${script_dir}/.." && pwd)"
cd "${project_dir}"
if [[ "$#" -ne 2 ]]; then
  echo "Usage: bash scripts/run_rwku_sure_hidden_direction_v31_base_probe.sh MODEL STEPHEN_KING_ATOMIC_CORPUS_DIR" >&2
  exit 2
fi
model="$1"
corpus_dir="$2"
python_bin="${PYTHON_BIN:-python}"
training_bundle="${corpus_dir}/generated_training_bundle.json"
generator_receipt="${corpus_dir}/generator_receipt.json"
configuration="config/rwku/sure_head_hidden_direction_v31_w1k_seed0.json"
probe_output="${RWKU_HD_V31_BASE_PROBE:-outputs/rwku_hd_v31_base_probe/stephen_king_seed0_base_answer_probe.json}"
test -d "${model}"
test -f "${training_bundle}"
test -f "${generator_receipt}"
test -f "${configuration}"
"${python_bin}" scripts/rwku_sure_hidden_direction_v31_base_probe.py \
  --model-path "${model}" \
  --training-bundle "${training_bundle}" \
  --generator-receipt "${generator_receipt}" \
  --configuration "${configuration}" \
  --output "${probe_output}"
