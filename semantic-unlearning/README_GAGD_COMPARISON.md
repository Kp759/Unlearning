# Semantic-Unlearning GA/GD Comparison

`semantic-unlearning/scripts/gagd_compare.py` is a self-contained comparison runner for gradient-ascent/gradient-descent (GA/GD) unlearning. It does not depend on ZeroUnlearn or optional editing baselines such as MEMIT, ROME, NSE, AlphaEdit, or MEND.

## Four settings

The runner compares four modes:

1. `full_all_tokens`
   - Trains all model parameters.
   - Uses answer-only cross-entropy over every answer token.
2. `full_selective_tokens`
   - Trains all model parameters.
   - Uses answer-only cross-entropy only on selected answer token positions.
3. `emb_lm_all_tokens`
   - Freezes all model parameters except input embeddings and untied output embeddings / `lm_head`.
   - Uses answer-only cross-entropy over every answer token.
4. `emb_lm_selective_tokens`
   - Freezes all model parameters except input embeddings and untied output embeddings / `lm_head`.
   - Uses answer-only cross-entropy only on selected answer token positions.
   - Masks embedding / lm_head gradients to selected token rows and restores non-selected rows exactly after each optimizer step.

For all modes, each GA/GD step samples a forget batch and a retain batch, then optimizes:

```text
total_loss = -forget_weight * forget_loss + retain_weight * retain_loss
```

Optional retain KL regularization is available with `--kl-retain-weight`, but defaults to `0.0` for pure GA/GD.

## MCF command

```bash
cd semantic-unlearning
python scripts/gagd_compare.py \
  --dataset mcf \
  --model-path /scratch/yl258/kp759/hf/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95 \
  --mode all \
  --forget-num 50 \
  --retain-num 1000 \
  --steps 100 \
  --batch-size 1 \
  --retain-batch-size 1 \
  --lr 1e-5 \
  --dtype bf16
```

A small smoke run is supported:

```bash
cd semantic-unlearning
python scripts/gagd_compare.py \
  --dataset mcf \
  --model-path /scratch/yl258/kp759/hf/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95 \
  --mode all \
  --forget-num 2 \
  --retain-num 4 \
  --steps 1 \
  --max-eval-examples 2
```

## TOFU command

```bash
cd semantic-unlearning
python scripts/gagd_compare.py \
  --dataset tofu \
  --forget-split forget05 \
  --retain-split retain95 \
  --model-path /scratch/yl258/kp759/hf/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95 \
  --mode all \
  --forget-num 50 \
  --retain-num 1000 \
  --steps 100 \
  --batch-size 1 \
  --retain-batch-size 1 \
  --lr 1e-5 \
  --dtype bf16
```

The convenience wrapper runs MCF first and TOFU only when requested. It sets `CUDA_VISIBLE_DEVICES=0` when that variable is not already set, so HPC jobs can override GPU placement by exporting `CUDA_VISIBLE_DEVICES` before launching it:

```bash
cd semantic-unlearning
bash scripts/run_gagd_compare_mcf_tofu.sh /scratch/yl258/kp759/hf/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95
RUN_TOFU=1 bash scripts/run_gagd_compare_mcf_tofu.sh
```


## MCF target choice and official-style metrics

By default, MCF training and the generic GA/GD loss metrics use `--mcf-answer-field target_new`. This matches the CounterFact / ZeroUnlearn rewrite target convention and makes the default `forget_loss_*` and `retain_loss_*` columns comparable to ZeroUnlearn/CounterFact-style metrics.

`--mcf-answer-field target_true` is available only as an optional diagnostic when you want the GA/GD objective and generic loss columns to use the original true answer instead of the rewrite target. Official-style MCF rewrite metrics are still computed for both `target_new` and `target_true` whenever both targets are available.

For MCF runs, the script also writes official-style rewrite metrics for both forget and retain splits:

- `forget_rewrite_target_new_nll`
- `forget_rewrite_target_true_nll`
- `forget_rewrite_new_over_true_success`
- `forget_rewrite_prob_diff_new_minus_true`
- `retain_rewrite_target_new_nll`
- `retain_rewrite_target_true_nll`
- `retain_rewrite_new_over_true_success`
- `retain_rewrite_prob_diff_new_minus_true`

If paraphrase prompts are available, it also writes:

- `forget_paraphrase_new_over_true_success`
- `retain_paraphrase_new_over_true_success`

Here `new_over_true_success` is the mean indicator that `target_new_nll < target_true_nll`, and `prob_diff_new_minus_true` is `mean(exp(-target_new_nll) - exp(-target_true_nll))`. For official-style MCF success metrics, lower `forget_rewrite_new_over_true_success` after unlearning means the rewrite target is less preferred, matching the ZeroUnlearn table direction where Eff/Gen are ↓.

## Metric directions

- Lower `forget_match_after` is better for forgetting.
- Higher `forget_loss_after` is better for forgetting for the GA/GD loss metrics.
- Lower `retain_loss_after` is better for retention.
- Higher `retain_match_after` is better for retention.
- `forget_loss_delta = forget_loss_after - forget_loss_before`; larger positive values usually indicate stronger forgetting.
- `retain_loss_delta = retain_loss_after - retain_loss_before`; values near zero or negative are better for retention.

For MCF, the generic `forget_loss_*` and `retain_loss_*` columns correspond to the selected `--mcf-answer-field` and default to `target_new`. The script additionally reports the official-style rewrite and paraphrase metrics listed above. For TOFU, it reports `forget_answer_nll` and `retain_answer_nll`.

## Expected output files

For an `--output-dir` such as `outputs/gagd_compare/mcf`, the script writes:

- `base_metrics.json`
- `selected_token_ids.json`
- `config_used.json`
- `<mode>/metrics.json`
- `<mode>/train_log.jsonl`
- `comparison.csv`
- `comparison.md`

Models are not saved by default. Pass `--save-model` to save each mode under `<mode>/checkpoint/`.

## OOM notes

The `full_*` modes train every model parameter and can require much more GPU memory than the `emb_lm_*` modes. The default optimizer is therefore SGD for `full_*` modes to avoid AdamW optimizer-state OOM on a 3B model. The `emb_lm_*` modes default to AdamW. You can override this with `--optimizer {sgd,adamw,adamw8bit}`; if `adamw8bit` is requested but bitsandbytes is unavailable, the script prints a warning and falls back to AdamW.

Other memory controls:

- Use `--gradient-checkpointing` for full-model modes.
- Keep `--batch-size 1 --retain-batch-size 1` on constrained GPUs.
- Use `--dtype bf16` or `--dtype fp16` when supported.
- The default training load is a single CUDA device; `--device-map auto` is available if your environment supports it.

## NJIT HPC notes

The runner is path-stable: you can launch it from the repository root or from `semantic-unlearning` because it changes into the `semantic-unlearning` directory before running Python. Example from the HPC checkout:

```bash
cd /scratch/yl258/kp759/Unlearning
bash semantic-unlearning/scripts/run_gagd_compare_mcf_tofu.sh \
  /scratch/yl258/kp759/hf/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95
```

For an interactive GPU smoke test, request a GPU allocation first, then run:

```bash
cd /scratch/yl258/kp759/Unlearning/semantic-unlearning
python scripts/gagd_compare.py \
  --dataset mcf \
  --model-path /scratch/yl258/kp759/hf/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95 \
  --mode all \
  --forget-num 2 \
  --retain-num 4 \
  --steps 1 \
  --max-eval-examples 2
```

Relative `--output-dir` and `--mcf-cache-path` values are resolved under `semantic-unlearning/`, so outputs and the MCF cache stay inside that project folder even when the wrapper is launched from `/scratch/yl258/kp759/Unlearning`.
