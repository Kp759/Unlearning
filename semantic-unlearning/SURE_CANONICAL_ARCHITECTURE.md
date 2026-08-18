# Canonical SURE-LM architecture for MCF and ZsRE

This document defines the canonical implementation introduced after the earlier benchmark-specific MCF and ZsRE paths diverged.

## Pipeline

```text
locked per-seed direct-only split
    -> common Stage 1 GA/GD
    -> restore all non-sensitive vocabulary rows to base
    -> direct-only residual detection
    -> common sparse sensitive-row LM-head repair
    -> direct-only rank selection and scale selection
    -> freeze checkpoint
    -> held-out benchmark evaluation + canonical PPL
```

## Benchmark adapter

Only two semantics differ by benchmark.

| Item | MCF | ZsRE |
|---|---|---|
| Sensitive answer | `target_new` | original `target_true` |
| Direct success constraint | `NLL(target_new)-NLL(target_true) >= margin` | sensitive token is below best non-sensitive token by margin |

Everything else is shared.

## Stage 1

Entrypoint: `scripts/sure_stage1_gagd.py`.

For every teacher-forced sensitive token `y` on a direct training-visible forget request:

```text
L_GA = mean(log p_theta(y | x, y_<t))
L_GD = KL(p_base(-y) || p_theta(-y))
L_stage1 = 2.0 * L_GA + 1.0 * L_GD
```

The sensitive token is removed from both distributions for GD and the remaining vocabulary is renormalized. The base teacher cache uses only the same direct forget PredictionCases.

Canonical defaults:

```text
steps = 600
batch = 1 PredictionCase
base-cache batch = 8
lr = 1e-4
optimizer = AdamW
weight decay = 0
grad clip = 1.0
GA weight = 2.0
GD weight = 1.0
```

All transformer blocks are frozen. On tied-vocabulary models, the input embedding and LM head are one shared trainable vocabulary matrix.

## Restoration

The old MCF `1.25 * base` factual-row boost is not part of the canonical method.

Canonical restoration is:

```text
sensitive rows -> keep trained Stage-1 values
all other rows -> exact base snapshot
```

Thus MCF protected `target_true`-only rows and all unrelated rows return exactly to base. ZsRE uses the same rule.

## Stage 2

Entrypoint: `scripts/sure_stage2_sparse_repair.py`.

Stage 2 clones/unties the LM head, freezes the transformer and input embeddings, and modifies only sensitive-answer output rows belonging to residual direct failures.

The same ordered candidate ranks are used for both benchmarks:

```text
2 -> 8 -> 0
```

where `0` means an unrestricted full selected-row delta. Positive ranks use a fixed SVD hidden-direction basis and train only row coefficients.

Selection uses direct training-visible requests only:

1. evaluate rank 2;
2. if it leaves direct failures, evaluate rank 8;
3. if needed, evaluate unrestricted rows;
4. choose the lowest-complexity candidate with zero direct failures;
5. if none reaches zero, minimize `(direct_failures, candidate_order, delta_norm)`;
6. run the shared scale sweep;
7. choose the smallest scale preserving zero direct failures, otherwise minimize `(direct_failures, scale)`.

No paraphrase/rephrase, neighborhood/locality, benchmark retain, or PPL input enters Stage 2 selection.

## Locked artifacts

MCF now uses `scripts/build_mcf_sure_canonical_split.py`, which writes one seed-specific `training_visible_forget.json` and `split_manifest.json`, matching the ZsRE locked artifact layout.

Both benchmarks sample 50 forget records from the official second-half pool and reserve 1,000 first-half retain records for final evaluation.

## Canonical PPL

Both runners use:

```text
data/wikidata
```

The final evaluation JSON is annotated by `scripts/annotate_ppl_provenance.py` with the exact truncated token IDs, the token-ID SHA-256, and the joined-text SHA-256.

Do not compare canonical results against historical ZsRE PPL values produced from `data/wikidata_aws_diag`.

## Aggregation

`scripts/aggregate_sure_canonical.py` uses sample standard deviation (`n-1`) for both MCF and ZsRE.

## Canonical runners

```bash
bash scripts/run_mcf_sure_canonical.sh MODEL_PATH [MCF_JSON]
bash scripts/run_zsre_sure_canonical.sh MODEL_PATH [ZSRE_JSON]
```

The old `run_zsre_sure_no_neutral_zerounlearn.sh` entry point is deprecated and forwards to the canonical ZsRE runner.

## Historical results

Existing best-run MCF and ZsRE records remain valid descriptions of their historical checkpoints, but they are not canonical cross-benchmark architecture results because they used different Stage-1/restoration/Stage-2 implementations. New canonical claims must come from rerunning the canonical runners.
