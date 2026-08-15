# SURE-TOFU v3: Progressive Sparse Sensitive-Row Protocol

## Purpose

V3 isolates whether the large sensitive-row footprint in v2 caused collateral
utility loss.  It keeps the same Full-TOFU starting checkpoint, author-balanced
50-QA training-visible forget split, Stage1A GA/GD, direct-forget requirements,
rank-0 optimizer, boundary bisection, and final locked evaluator.  Only the
Stage1B sensitive-row selection policy changes.

## Data access

Before the final checkpoint is frozen, v3 may access only:

- the same 50 training-visible direct forget QAs;
- the protected Full-TOFU checkpoint, restricted to vocabulary-row restoration;
- quantities derived from those 50 QAs (answer-token document frequency,
  direct answer NLLs, direct constraint violations).

It does not access retain95, paraphrases, same-author heldout QAs, real-authors,
world-facts, PPL, or any final locked-evaluation metric.

## Stage1A

Identical to the existing SURE protocol:

- 600 steps;
- batch size 1;
- embedding + LM-head LR `1e-4`;
- GA weight 2.0;
- GD weight 1.0;
- `sensitive_both` post-training restoration mode.

## Stage1B-v3 progressive sparse selection

Let `A` be the union of all answer-token vocabulary rows in the 50 visible QAs.
For every answer, construct a complete deterministic ranking using only those
50 answers:

1. content-bearing tokens before punctuation/symbol-only tokens;
2. ascending answer-document frequency (rarer rows first);
3. original answer-token order;
4. token ID as a deterministic final tie-break.

Sensitivity is global by vocabulary row: once a token ID is selected anywhere,
its shared row is sensitive everywhere.

### Initial set

For every currently violating direct QA, add its top `K` ranked rows to the
global sensitive set `S`; default `K=3`.

Set `N=A-S` and restore every row in `N` exactly to Full-TOFU Base in both:

- input embedding;
- LM head.

Rows in `S` retain their Stage1A embedding/LM-head values before rank-0.

### Restricted rank-0 solve

Run unrestricted selected-row LM-head rank-0 repair only on `S` against all 50
fixed direct-forget constraints.  The transformer and input embeddings remain
frozen during the rank-0 optimizer.

At the first feasible optimizer crossing, binary-search between the last
infeasible and first feasible deltas and keep a near-boundary feasible delta.

### Progressive promotion

If the full restricted rank-0 budget cannot satisfy all direct constraints:

1. identify only the still-failing direct QAs from that attempted solution;
2. for each failing QA, add its next `P` ranked rows not already globally
   sensitive; default `P=1`;
3. restore all remaining non-sensitive rows to Base again;
4. restart the restricted rank-0 solve from zero delta on the enlarged row set.

No rows are promoted for QAs that are already feasible.  The process stops at
the first feasible sensitive-row set or fails closed if all ranked rows are
exhausted / the promotion-round limit is reached.

### Final materialization

After the first feasible sparse solve:

- reconstruct the exact final row policy;
- materialize the boundary-refined rank-0 LM-head delta on sensitive rows;
- snap every non-sensitive answer row to Full-TOFU Base again in both embedding
  and LM head;
- audit every one of the 50 direct constraints after model-dtype materialization;
- save only if every constraint passes.

## V3 first experiment

The first v3 experiment evaluates the frozen Stage1B-v3 R0 checkpoint directly,
without adding Stage2 restoration.  This isolates whether sparse sensitivity
alone improves utility relative to v2 (`542/613` sensitive rows, retain ratio
about `0.291`).

If v3 R0 materially improves retain utility while preserving direct forgetting,
Stage2 restoration can be added later as a separate predeclared ablation.

## Storage policy

Before the v3 run, delete only model-weight directories named `checkpoint` under
`outputs/tofu_sure*`.  Preserve all parent JSON/JSONL/log/evaluation files from
v1, v2, and R512/R1024 experiments.  The protected Full-TOFU epoch-5 checkpoint
must never be deleted.

After locked v3 evaluation, delete the temporary v3 Stage1A and Stage1B
`checkpoint` directories while keeping all v3 reports and evaluation JSON.
