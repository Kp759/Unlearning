# Embedding / LM Head ZeroUnlearn Ablations

This repository now includes two embedding-and-output-head-only ablations that use the same retain/unlearn split and evaluation flow as `ZeroUnlearn`.

## Algorithms

### `ZeroUnlearn_EmbHead_All`

`ZeroUnlearn_EmbHead_All` freezes every model parameter first, then unfreezes only `model.get_input_embeddings().weight` and `model.get_output_embeddings().weight`. The optimizer updates the full input-embedding and LM-head/output-embedding matrices. If the embeddings are tied, the shared parameter is passed to the optimizer only once.

### `ZeroUnlearn_EmbHead_TouchedRows`

`ZeroUnlearn_EmbHead_TouchedRows` also freezes transformer blocks, attention, MLPs, and all hidden-layer parameters. It updates only embedding/output-head rows for tokens that appear in each request's `subject`, `target_true["str"]`, or `target_new["str"]`. Non-touched row gradients are zeroed during backpropagation, and non-touched rows are restored from original copies after each optimizer step.

## Llama 3.2 3B Instruct snapshot

Use the logical model alias for hyperparameter selection, result directory naming, and logs:

```bash
model_name="Llama-3.2-3B-Instruct"
model_path="/scratch/yl258/kp759/hf/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95"
```

### Baseline ZeroUnlearn

```bash
python experiments/evaluate.py \
  --alg_name ZeroUnlearn \
  --model_name "Llama-3.2-3B-Instruct" \
  --model_path "/scratch/yl258/kp759/hf/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95" \
  --hparams_fname "Llama-3.2-3B-Instruct.json" \
  --ds_name zsre \
  --ratio_or_num \
  --unlearn_num 50 \
  --retain_num 1000 \
  --edit_layer_nums 3 \
  --eval_retain \
  --seed 1
```

### EmbHead all rows

```bash
python experiments/evaluate.py \
  --alg_name ZeroUnlearn_EmbHead_All \
  --model_name "Llama-3.2-3B-Instruct" \
  --model_path "/scratch/yl258/kp759/hf/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95" \
  --hparams_fname "Llama-3.2-3B-Instruct.json" \
  --ds_name zsre \
  --ratio_or_num \
  --unlearn_num 50 \
  --retain_num 1000 \
  --edit_layer_nums 0 \
  --eval_retain \
  --seed 1
```

### EmbHead touched rows

```bash
python experiments/evaluate.py \
  --alg_name ZeroUnlearn_EmbHead_TouchedRows \
  --model_name "Llama-3.2-3B-Instruct" \
  --model_path "/scratch/yl258/kp759/hf/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95" \
  --hparams_fname "Llama-3.2-3B-Instruct.json" \
  --ds_name zsre \
  --ratio_or_num \
  --unlearn_num 50 \
  --retain_num 1000 \
  --edit_layer_nums 0 \
  --eval_retain \
  --seed 1
```

For the full comparison over `zsre`, `mquake`, and `mcf` with seeds 1 through 5, run:

```bash
bash sh/run_embhead_ablation_llama32.sh
```

## Output files to compare

Each run preserves the existing evaluation outputs:

- `forget_metrics.jsonl`
- `retain_metrics.jsonl`
- `forget_summarize_results.json`
- `retain_summarize_results.json`
- `glue_eval/*.json` when downstream GLUE evaluation is enabled

## Locality warning

These EmbHead ablations change lexical/token-level parameters instead of hidden-layer modules. Even the touched-row variant can alter many contexts that reuse the same tokens, so locality may be worse than hidden-layer ZeroUnlearn edits.
