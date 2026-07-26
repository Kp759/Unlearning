#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "${script_dir}/.." && pwd)"

python_bin="${PYTHON_BIN:-python}"
model_path="${MODEL_PATH:-/scratch/yl258/kp759/hf/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95}"
output_root="${RWKU_OUTPUT_ROOT:-${project_dir}/outputs/rwku}"
data_root="${RWKU_DATA_ROOT:-${project_dir}/data/rwku}"
mcf_path="${MCF_PATH:-${project_dir}/data/multi_counterfact.json}"
wikidata_dir="${WIKIDATA_DIR:-${project_dir}/data/wikidata}"

for seed in 0 1 2 3 4 5 6 7 8 9; do
  "${python_bin}" "${script_dir}/rwku_experiment.py" \
    --seed "${seed}" \
    --model-path "${model_path}" \
    --output-root "${output_root}" \
    --data-root "${data_root}" \
    --mcf-path "${mcf_path}" \
    --wikidata-dir "${wikidata_dir}" \
    "$@"
done

"${python_bin}" "${script_dir}/aggregate_rwku_results.py" \
  --input-root "${output_root}" \
  --output-dir "${output_root}/aggregate"
