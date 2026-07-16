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

## JSON-LMHead-Zero + Target-True Emb Restore comparison

For a fair comparison with the saved `lm_head_zero_true_restore150` results,
GA/GD must train and evaluate on the same MCF records. The default
`--mcf-sample-mode official` now matches the JSON-LMHead and ZeroUnlearn split:

- retain pool: first half of MCF;
- forget pool: second half of MCF;
- both pools sampled with the same seed;
- 50 forget and 1000 retain records by default.

Run all four scopes for seeds 0-4 and aggregate them with the existing baseline:

```bash
cd semantic-unlearning
bash scripts/run_gagd_vs_json_lmhead.sh /path/to/Llama-3.2-3B-Instruct
```

The comparison wrapper now uses an MCF margin objective on forget records and
answer-only GD on retain `target_new` tokens. The margin directly trains the
official decision rule while retaining the same quantity protected by the
JSON-LMHead run. It runs official evaluation in memory, so checkpoints are not
written unless `--save-model` is explicitly used. Final outputs are:

```text
outputs/gagd_vs_json_lmhead/comparison/per_seed.md
outputs/gagd_vs_json_lmhead/comparison/aggregate.md
outputs/gagd_vs_json_lmhead/comparison/comparison.json
```

To aggregate existing results without training, use:

```bash
python scripts/compare_gagd_to_json_lmhead.py \
  --seeds 0 1 2 3 4 \
  --gagd-pattern 'outputs/gagd_vs_json_lmhead/seed{seed}/official_eval/{method}_official_eval.json' \
  --baseline-pattern 'outputs/official_eval_lmhead_zero_true_restore150_seed{seed}_spefix.json' \
  --base-pattern 'outputs/official_eval_base_seed{seed}_spefix.json'
```

The aggregator rejects mismatched seed, split mode, forget count, retain count,
or dataset metadata rather than silently mixing incomparable runs.

For plain `answer_nll` GA/GD, each step optimizes:

```text
total_loss = -forget_weight * forget_loss + retain_weight * retain_loss
```

Optional retain KL regularization is available with `--kl-retain-weight`, but defaults to `0.0` for pure GA/GD.

The JSON-LMHead comparison wrapper uses the stronger settings below by default:

```text
forget_loss_type = mcf_margin
forget_margin = 1.0
steps = 250
full-model SGD lr = 1e-3
embedding/LM-head AdamW lr = 1e-4
forget_weight = 2.0
retain_weight = 1.0
forget batch size = 1
retain batch size = 4
sampling_strategy = epoch
```

These are intentionally scope-aware: `1e-5` is especially weak for plain SGD,
while using the same larger rate for AdamW embedding rows can be too aggressive.
All values remain overridable with the wrapper environment variables
`STEPS`, `FULL_LR`, `EMB_LM_LR`, `FORGET_WEIGHT`, `RETAIN_WEIGHT`,
`FORGET_MARGIN`, `RETAIN_BATCH_SIZE`, and `SAMPLING_STRATEGY`.

## MCF command

```bash
cd semantic-unlearning
python scripts/gagd_compare.py \
  --dataset mcf \
  --model-path /scratch/yl258/kp759/hf/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95 \
  --mode all \
  --forget-num 50 \
  --retain-num 1000 \
  --forget-loss-type mcf_margin \
  --forget-margin 1.0 \
  --steps 250 \
  --batch-size 1 \
  --retain-batch-size 4 \
  --full-lr 1e-3 \
  --emb-lm-lr 1e-4 \
  --forget-weight 2.0 \
  --retain-weight 1.0 \
  --sampling-strategy epoch \
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

For MCF runs, `comparison.csv`, `comparison.md`, and each mode `metrics.json` also include these official-style before/after metrics for both forget and retain splits:

- `forget_target_new_nll_before`
- `forget_target_new_nll_after`
- `forget_target_true_nll_before`
- `forget_target_true_nll_after`
- `forget_new_over_true_success_before`
- `forget_new_over_true_success_after`
- `retain_target_new_nll_before`
- `retain_target_new_nll_after`
- `retain_target_true_nll_before`
- `retain_target_true_nll_after`
- `retain_new_over_true_success_before`
- `retain_new_over_true_success_after`

Here `new_over_true_success` is the mean indicator that `target_new_nll < target_true_nll`. Lower `forget_new_over_true_success_after` means stronger unlearning in the ZeroUnlearn/CounterFact target_new-vs-target_true sense.



## MCF margin forget loss

The default forget objective is `--forget-loss-type answer_nll`, which keeps the original GA behavior: maximize the selected forget answer NLL and minimize retain answer NLL. For MCF, you can instead use `--forget-loss-type mcf_margin` to align the training objective more directly with official CounterFact/ZeroUnlearn Eff/Gen success.

`mcf_margin` computes answer-only NLL for both `target_new` and `target_true`
on the same forget prompt, excluding EOS to match the official Eff/Gen target
comparison, and minimizes:

```text
softplus(target_true_nll - target_new_nll + forget_margin)
```

Minimizing this margin pushes
`target_new_nll >= target_true_nll + forget_margin`, which lowers
`target_new_nll < target_true_nll` success and therefore aligns with reducing
official Eff/Gen. A positive `--forget-margin` keeps training past the fragile
zero-gap boundary; the comparison wrapper uses `1.0`. This option is only valid
with `--dataset mcf`; TOFU runs raise a clear error if it is selected. Retain
loss remains answer-only CE using `--mcf-answer-field`, which defaults to
`target_new`.

`--sampling-strategy epoch` is the default and reshuffles only after every
record has been used. Use `--sampling-strategy with_replacement` to reproduce
the legacy random sampler. With 50 forget records and batch size 1, epoch
sampling guarantees five visits per record in 250 steps.

Selective-token modes apply the selected-token mask to both `target_new` and `target_true` NLLs. If an example has zero selected `target_true` tokens, it falls back to the full target-true answer labels and logs that fallback in `train_log.jsonl`. The log also records `forget_loss_type`, `forget_margin_loss`, `forget_target_new_nll`, and `forget_target_true_nll` when margin loss is used.


## ZeroUnlearn-style GA baseline in semantic-unlearning

`--forget-loss-type zerounlearn_ga` implements the official ZeroUnlearn GA baseline objective inside `semantic-unlearning` without importing ZeroUnlearn code. It differs from the default `answer_nll` GA/GD objective in several important ways:

- It always uses MCF `target_true` for the forget loss, ignoring `--mcf-answer-field` for forget examples.
- It prepends a leading space to `target_true` when needed.
- It computes log-probabilities from the model logits at the last prompt token position and gathers the log-probabilities of the separately tokenized `target_true` token IDs.
- The minimized loss is the mean `target_true` log-probability, not NLL, and is **not** multiplied by `-1`. Lower values mean stronger GA because the model assigns lower probability to `target_true`.
- It does not require retain GD for the official baseline; run with `--retain-weight 0.0` for the exact baseline, though retain-loss variants remain available for experiments.

The official Llama-3.2-3B GA defaults are `--steps 25 --lr 5e-3 --optimizer adam --weight-decay 0.0 --retain-weight 0.0 --forget-weight 1.0`. The helper script runs this objective across all four parameter scopes, using `adamw8bit` by default for full-model modes to reduce OOM risk and `adam` for embedding/lm-head modes:

```bash
cd semantic-unlearning
bash scripts/run_zerounlearn_ga_4scope_mcf.sh \
  /scratch/yl258/kp759/hf/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95
```

The checkpoints are written under:

- `outputs/zerounlearn_ga_4scope_mcf/full_all_tokens_run/full_all_tokens/checkpoint`
- `outputs/zerounlearn_ga_4scope_mcf/full_selective_tokens_run/full_selective_tokens/checkpoint`
- `outputs/zerounlearn_ga_4scope_mcf/emb_lm_all_tokens_run/emb_lm_all_tokens/checkpoint`
- `outputs/zerounlearn_ga_4scope_mcf/emb_lm_selective_tokens_run/emb_lm_selective_tokens/checkpoint`

After all four runs, the script evaluates the base model plus all checkpoints with `run_same_mcf_eval.py` and writes the final ZeroUnlearn-compatible table to:

```text
outputs/zerounlearn_ga_4scope_mcf/same_eval_with_base/official_eval_comparison.md
```

Metric directions for this baseline:

- Training diagnostic `forget_loss_after`: lower is stronger GA because it is target_true log-probability.
- Official `Eff`: ↓ is better unlearning.
- Official `Gen`: ↓ is better unlearning.
- Official `Spe`: ↑ is better specificity.
- Official `PPL`: ↓ or stable is better fluency.

## ZeroUnlearn-compatible official MCF evaluation

`comparison.md` from `gagd_compare.py` is a GA/GD training-diagnostic loss table. Use it to inspect whether the optimization objective moved in the expected direction, but do **not** use it as the final ZeroUnlearn comparison table.

For final MCF comparisons against ZeroUnlearn, use the official-compatible evaluator and compare `official_eval_comparison.md`. It reports:

- `Eff`: rewrite success; ↓ is better unlearning.
- `Gen`: paraphrase success; ↓ is better unlearning.
- `Spe`: neighborhood probability-difference specificity; ↑ is better specificity.
- `Spe_success`: neighborhood success rate, reported separately.
- `PPL`: Wikidata-style perplexity; ↓ or stable is better fluency.

To make `gagd_compare.py` run the official-compatible evaluator after every mode, add `--run-official-mcf-eval`. Evaluation runs on the trained model in memory; add `--save-model` separately only when checkpoints are needed. The evaluator writes:

- `outputs/gagd_compare/.../official_eval/<mode>_official_eval.json`
- `outputs/gagd_compare/.../official_eval_comparison.csv`
- `outputs/gagd_compare/.../official_eval_comparison.md`

Example:

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
  --dtype bf16 \
  --run-official-mcf-eval
```

You can also compare arbitrary checkpoints, including a ZeroUnlearn checkpoint/output model and the four semantic-unlearning mode checkpoints, with:

```bash
cd semantic-unlearning
python scripts/run_same_mcf_eval.py \
  --model-dirs \
    ZeroUnlearn=/path/to/zerounlearn/checkpoint \
    full_all_tokens=outputs/gagd_compare/mcf/full_all_tokens/checkpoint \
    full_selective_tokens=outputs/gagd_compare/mcf/full_selective_tokens/checkpoint \
    emb_lm_all_tokens=outputs/gagd_compare/mcf/emb_lm_all_tokens/checkpoint \
    emb_lm_selective_tokens=outputs/gagd_compare/mcf/emb_lm_selective_tokens/checkpoint \
  --mcf-path data/mcf/multi_counterfact.json \
  --wikidata-dir data/wikidata \
  --out-dir outputs/gagd_compare/same_mcf_eval \
  --unlearn-num 50 \
  --retain-num 1000 \
  --seed 0 \
  --sample-mode official \
  --dtype bf16
```

If Wikidata is unavailable and you do not pass `--skip-ppl`, the evaluator warns and sets PPL to `null` rather than crashing.

## Metric directions

- Lower `forget_match_after` is better for forgetting.
- Higher `forget_loss_after` is better for forgetting for the GA/GD loss metrics.
- Lower `retain_loss_after` is better for retention.
- Higher `retain_match_after` is better for retention.
- `forget_loss_delta = forget_loss_after - forget_loss_before`; larger positive values usually indicate stronger forgetting.
- `retain_loss_delta = retain_loss_after - retain_loss_before`; values near zero or negative are better for retention.

For MCF, the generic `forget_loss_*` and `retain_loss_*` columns correspond to the selected `--mcf-answer-field` and default to `target_new`. The script additionally reports the official-style target_new-vs-target_true metrics listed above. For TOFU, it reports `forget_answer_nll` and `retain_answer_nll`.

## Expected output files

For an `--output-dir` such as `outputs/gagd_compare/mcf`, the script writes:

- `base_metrics.json`
- `selected_token_ids.json`
- `config_used.json`
- `<mode>/metrics.json`
- `<mode>/train_log.jsonl`
- `comparison.csv`
- `comparison.md` (GA/GD diagnostic only)
- `official_eval_comparison.md` when `--run-official-mcf-eval` is set

Models are not saved by default. Pass `--save-model` to save each mode under `<mode>/checkpoint/`.

## OOM notes

The `full_*` modes train every model parameter and can require much more GPU memory than the `emb_lm_*` modes. The default optimizer is therefore SGD for `full_*` modes to avoid AdamW optimizer-state OOM on a 3B model. The `emb_lm_*` modes default to AdamW. `--full-lr` and `--emb-lm-lr` provide scope-specific learning rates. `--full-optimizer` and `--emb-lm-optimizer` provide scope-specific optimizers, while `--optimizer` overrides both. If `adamw8bit` is requested but bitsandbytes is unavailable, the script prints a warning and falls back to AdamW.

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
