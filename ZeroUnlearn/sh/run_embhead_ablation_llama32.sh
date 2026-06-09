#!/usr/bin/env bash
set -euo pipefail

model_name="Llama-3.2-3B-Instruct"
model_path="/scratch/yl258/kp759/hf/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95"
data_set=(zsre mquake mcf)
seeds=(1 2 3 4 5)
unlearn_num="${UNLEARN_NUM:-50}"
retain_num="${RETAIN_NUM:-1000}"
eval_edited_glue="${EVAL_EDITED_GLUE:-False}"

downstream_eval_steps=0
if [[ "${eval_edited_glue}" == "True" || "${eval_edited_glue}" == "true" || "${eval_edited_glue}" == "1" ]]; then
  downstream_eval_steps=1
fi

for ds_name in "${data_set[@]}"; do
  for seed in "${seeds[@]}"; do
    for alg_name in ZeroUnlearn ZeroUnlearn_EmbHead_All ZeroUnlearn_EmbHead_TouchedRows; do
      edit_layer_nums=0
      if [[ "${alg_name}" == "ZeroUnlearn" ]]; then
        edit_layer_nums=3
      fi

      python experiments/evaluate.py \
        --alg_name "${alg_name}" \
        --model_name "${model_name}" \
        --model_path "${model_path}" \
        --hparams_fname "${model_name}.json" \
        --ds_name "${ds_name}" \
        --ratio_or_num \
        --unlearn_num "${unlearn_num}" \
        --retain_num "${retain_num}" \
        --edit_layer_nums "${edit_layer_nums}" \
        --eval_retain \
        --downstream_eval_steps "${downstream_eval_steps}" \
        --seed "${seed}"
    done
  done
done
