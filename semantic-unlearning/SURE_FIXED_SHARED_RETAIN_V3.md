# SURE-LM fixed shared retain-protected v3

This is the cross-benchmark architecture contract for MCF and ZsRE.

## Data exposure

Both datasets expose the same kinds of information during optimization:

- 50 direct forget prompts plus the sensitive answer only.
- 1,000 direct retain-train prompts only; no retain answer labels.
- 1,000 official retain-eval records, strictly disjoint from retain-train and never opened before final evaluation.
- No paraphrase/rephrase, neighborhood/locality, generation, or PPL inputs before checkpoint freeze.

MCF maps original `target_true` into canonical `target_new` as the sensitive slot. Original MCF `target_new` (counterfactual/reference) is not present in the training-visible forget artifact. ZsRE uses original `target_true` as sensitive and exposes no replacement/neutral answer.

## Trainable architecture

For both datasets:

```text
frozen transformer
frozen Base input embeddings
untied LM head
only sensitive output rows editable
row-specific direct-forget hidden-state bases
```

Stage-1 context rank cap: 2.
Stage-2 context-rank candidates: 2 -> 8 -> full observed row-specific context rank (`0`).

## Shared objective

For both Stage 1 and Stage 2:

```text
L = 4.0 * L_GA_sensitive
  + 1.0 * L_KL_forget_non_sensitive
  + 1.0 * L_KL_retain_full_distribution
  + lambda_2 * ||Delta W||^2
```

`L_GA_sensitive` minimizes sensitive-token log probability.
`L_KL_forget_non_sensitive` preserves the frozen-Base distribution after removing and renormalizing around the current sensitive token.
`L_KL_retain_full_distribution` is `KL(Base || current)` over the complete next-token vocabulary on disjoint retain-train direct prompts.

There is no benchmark-specific reference-answer CE, pairwise MCF objective, hinge, or ReLU training loss.

## Shared direct constraints

Candidate/scale selection is identical across datasets and requires both, per teacher-forced sensitive token:

```text
best_non_sensitive_logit - sensitive_logit >= 0.25
sensitive_NLL_post - sensitive_NLL_base >= 4.0
```

These constraints use only direct forget sensitive cases and the frozen Base teacher. They do not use benchmark-specific replacement answers.

## Shared defaults

```text
forget records       = 50
retain-train records = 1000
retain-eval records  = 1000, official and disjoint
forget batch         = 1 (Stage 1), 8 (Stage 2)
retain batch         = 4
Stage-1 steps        = 600
Stage-1 lr           = 1e-4
GA weight            = 4.0
forget GD weight     = 1.0
retain KL weight     = 1.0
Stage-1 context rank = 2
Stage-2 ranks        = 2,8,0
Stage-2 steps        = 800
Stage-2 lr           = 5e-3
Stage-2 L2           = 1e-6
logit margin         = 0.25
minimum NLL increase = 4.0
```

## Runners

```bash
bash scripts/run_mcf_sure_fixed_shared_retain_v3.sh MODEL [MCF_JSON]
bash scripts/run_zsre_sure_fixed_shared_retain_v3.sh MODEL [ZSRE_JSON]
```
