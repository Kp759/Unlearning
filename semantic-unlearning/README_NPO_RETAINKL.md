# NPO + Retain CE + Retain KL for MCF

This is a standalone semantic-unlearning experiment for Multi-CounterFact (MCF). It does not depend on `scripts/gagd_compare.py`.

## Objective

For each forget record, the script formats `requested_rewrite.prompt` with `requested_rewrite.subject` and treats `requested_rewrite.target_new.str` as the negative answer. It computes answer-only log probabilities under the trainable policy and a frozen base/reference model:

```text
log_ratio = log p_current(target_new | prompt) - log p_base(target_new | prompt)
NPO loss = (2 / beta) * softplus(beta * log_ratio)
```

Minimizing this loss lowers the probability of the forget-set new target relative to the base model while saturating as the policy moves below the reference.

For retain records, the target is also `requested_rewrite.target_new.str`. Retain CE is ordinary answer-only negative log likelihood. Retain KL compares the frozen reference distribution and current model distribution on retain tokens, defaulting to answer-only `KL(p_ref || p_current)`. Use `--retain-kl-direction current_ref` for the reverse direction, or `--no-retain-kl-answer-only` to include all non-padding positions.

The total training objective is:

```text
total_loss = forget_weight * npo_loss
           + retain_weight * retain_ce_loss
           + kl_retain_weight * retain_kl_loss
```

## How this differs from GA/GD and margin GA

Gradient ascent / gradient descent (GA/GD) directly pushes forget examples up in loss and retain examples down in loss. Margin GA variants usually stop pushing once a chosen forget margin is reached. NPO instead compares the current model against a frozen base model and optimizes a softplus of the log-probability ratio, giving a reference-relative and smoothly saturating forget penalty rather than unbounded ascent on cross entropy.

## Four scopes

1. `full_all_tokens`: all model parameters are trainable, and all answer tokens contribute.
2. `full_selective_tokens`: all model parameters are trainable, but losses are restricted to selected MCF answer token positions.
3. `emb_lm_all_tokens`: only input embeddings and the output LM head are trainable, and all answer tokens contribute.
4. `emb_lm_selective_tokens`: only input embeddings and the output LM head are trainable; only selected answer token positions contribute; gradients for non-selected embedding / LM-head rows are masked and those rows are restored after each optimizer step.

Selected token IDs are collected from the forget-set subject, `target_new`, and `target_true`, excluding tokenizer special tokens.

## Smoke test

```bash
python semantic-unlearning/scripts/npo_retainkl_compare.py \
  --model-path /path/to/model \
  --output-dir outputs/npo_retainkl_smoke \
  --mode emb_lm_selective_tokens \
  --forget-num 2 \
  --retain-num 4 \
  --steps 1 \
  --batch-size 1 \
  --retain-batch-size 1 \
  --reference-device cpu \
  --optimizer adam \
  --skip-ppl
```

## Full four-scope MCF runner

```bash
MODEL_PATH=/path/to/model \
STEPS=300 \
SEED=1 \
bash semantic-unlearning/scripts/run_npo_retainkl_4scope_mcf.sh
```

The runner writes checkpoints, training logs, aggregate comparisons, and an official-compatible comparison with the base model under:

```text
outputs/npo_retainkl_4scope_mcf/
outputs/npo_retainkl_4scope_mcf/same_eval_with_base/official_eval_comparison.md
```

## OOM workaround

The script loads a frozen reference/base model for NPO and retain KL. If GPU memory is tight, keep the trainable model on CUDA and place the reference model on CPU:

```bash
python semantic-unlearning/scripts/npo_retainkl_compare.py \
  --model-path /path/to/model \
  --output-dir outputs/npo_retainkl_cpu_ref \
  --mode emb_lm_selective_tokens \
  --reference-device cpu
```

The default `--reference-device auto` tries CUDA first and falls back to CPU on CUDA OOM with a warning.
