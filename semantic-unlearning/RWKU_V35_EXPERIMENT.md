# RWKU SURE v3.5 — Embeddings + LM Head + Representation Repair

## Status

Development-only, post-hoc Stephen King seed-0 experiment. This does **not** replace frozen v3.2 and does not reopen official RWKU evaluation.

## Motivation

v3.2 showed that decoder-aware representation repair can drive frozen-original-decoder answer recovery from 100% to 0% while strongly preserving held-out Wikipedia utility, but its successful physical candidate required a 1.4117% final-layer `down_proj` intervention, above the predeclared 1% budget.

v3.5 tests whether the same forgetting pressure can be distributed across the input embeddings, deployed LM head, and final representation path so that the required `down_proj` intervention becomes smaller.

## Architecture

```text
input token ids
    |
    v
E' = trainable input embeddings
    |
    v
frozen transformer backbone
    |
    v
final MLP down_proj + rank-r LoRA       <-- trainable
    |
    v
edited final hidden state h'
    |
    +--------------------+
    |                    |
    v                    v
immutable original W0    trainable deployed LM head W'
(frozen recovery probe)  (starts from Stage-1 sparse head)
```

Trainable in v3.5:

- full input embedding matrix,
- full untied deployed LM head,
- final-layer MLP `down_proj` LoRA, rank ladder {1,2,4}.

Frozen:

- all attention modules,
- all earlier MLPs,
- final-layer `gate_proj` and `up_proj`,
- immutable original decoder copy `W0`.

## Starting point

Every rank starts from exactly the same source used by v3.2:

1. base Llama-3.2-3B-Instruct,
2. output head untied from input embeddings,
3. frozen v1 Stage-1 sparse LM-head delta materialized,
4. input embeddings still equal to Base,
5. `W0` copied before any v3.5 optimization and never updated.

## Objective

v3.5 preserves the v3.2 loss:

```text
8 * frozen-W0 answer hinge
+ 2 * deployed edited-head answer hinge
+ 2 * Wikipedia hidden-state preservation
+ 50 * exact full-vocabulary Base||Edited Wikipedia KL
+ 1e-4 * down_proj LoRA L2
```

The only substantive method change from v3.2 is the trainable parameter set: embeddings and deployed LM head are now optimized jointly with the final-layer `down_proj` LoRA.

## Learning rates

- embeddings: 1e-4
- deployed LM head: 1e-4
- down_proj LoRA: 5e-4
- steps: 300
- answer batch: 8
- exact-KL utility contexts per optimizer step: 4
- grad clip: 1.0
- weight decay: 0

## Selection

v3.5 evaluates every checkpoint (25, 50, ..., 300) against each physical `down_proj` scale:

```text
0.125, 0.25, 0.5, 0.75, 1.0
```

The embedding and LM-head state stays fixed at that checkpoint while only the `down_proj` LoRA materialization scale changes.

A candidate must pass before optimization-Wiki ranking:

- direct atomic success = 100%,
- generated-subject atomic success = 100%,
- frozen-W0 recovery = 0%,
- frozen-W0 minimum demotion margin >= 0.05,
- total LM-head delta Frobenius from original Base <= 1.5,
- `down_proj` relative Frobenius <= 1%.

Among pre-safe candidates, selection minimizes optimization-pool Wiki KL mean, then p95, then max, then total reported intervention drift.

## Utility boundary

- 1,000 external-Wikipedia contexts: optimization only.
- first 1,000 guard contexts: already opened by v3.2 and explicitly excluded from v3.5 training and selection.
- next disjoint 1,000 guard contexts: fresh v3.5 confirmatory utility set, opened only after selection.
- official RWKU rows: never loaded by the v3.5 learner.

Fresh confirmatory utility thresholds remain:

- mean KL <= 0.01,
- p95 KL <= 0.05,
- max KL <= 0.5.

## Primary question

Does joint embedding + LM-head training preserve 0% frozen-W0 recovery and strong utility while reducing the physical `down_proj` intervention from the v3.2 value of 1.4117% to <=1%?

## Interpretation rule

A v3.5 success is a development result, not a fresh RWKU benchmark result. Stephen King remains a post-hoc method-development target because official metrics were observed before v3.x design.
