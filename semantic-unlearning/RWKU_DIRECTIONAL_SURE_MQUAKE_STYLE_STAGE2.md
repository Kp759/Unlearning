# RWKU Directional SURE: MQuAKE-style protected Stage 2

This is a post-hoc Stephen King development experiment. Official RWKU evaluation remains locked until a development checkpoint passes all generated-atomic and external-Wikipedia gates.

## Level 1

Level 1 is deliberately unchanged from the previously tested content-sensitive embedding-GA-only configuration:

- content-sensitive input-embedding rows and the same untied LM-head rows;
- transformer exactly frozen;
- embedding update: `2 * GA` only; embedding GD is not applied;
- LM-head update: `2 * GA -> B_S + GD -> B_P(external Wikipedia)`;
- 600 steps; embedding LR `5e-5`; head LR `1e-4`;
- `rank(B_S)=8`, `rank(B_P)=32`;
- unchanged RWKU margin `0.01` and Wiki KL budgets `0.01 / 0.05 / 0.5`.

The learner reuses the exact previous Level-1 implementation and captures its selected external-Wikipedia-safe anchor immediately before the old residual Level 2 would begin.

## Stage 2

Let `F` be the Level-1 pairwise-margin-failed generated prompts and `P` all Level-1 pairwise-margin-passing prompts.

Stage 2 freezes the Level-1 input-embedding delta and the transformer. Only an incremental LM-head repair is trainable.

1. Collect final prediction-position hidden states `H_P` and `H_F` at the frozen Level-1 anchor.
2. Build `B_P = rowspace(H_P)` with protected rank at most 32.
3. Residualize failures: `R_F = H_F - Proj_{B_P}(H_F)`.
4. Build `B_F = rowspace(R_F)` with repair rank at most 4.
5. Restrict output rows to content-sensitive rows implicated by `F`.
6. Parameterize the repair as `Delta W_AF = C_F B_F`.

Because embeddings and the transformer are frozen in Stage 2, `B_P` and `B_F` are fixed rather than refreshed.

### Loss

For the residual batch:

`mean([max(0, 0.01 - (max_other_logit - sensitive_logit))]^2)`

plus non-sensitive KL on sampled Level-1-success cases relative to the Level-1-anchor logits, plus `1e-6` repair L2.

The MQuAKE margin value `0.05` is **not** imported. RWKU keeps its predeclared margin `0.01` so this experiment isolates the Stage-2 architecture rather than changing the behavioral gate.

### Hard protection and backtracking

Every AdamW proposal is checked on **all** Level-1 success cases. The full optimizer proposal (`scale=1`) is attempted first. It is accepted only if:

- Level-1 success-prompt regressions = 0; and
- mean non-sensitive KL from the Level-1 anchor on `P` <= `0.05`.

If the full step violates protection, the coefficient step is backtracked through:

`0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625`

If no scale passes, the proposal is rolled back. After a backtracked or rolled-back proposal, AdamW is reset so optimizer momentum from the forbidden larger step cannot leak forward.

If any proposed/backtracked step reaches zero residual failures, the unchanged external-Wikipedia selection utility gate is evaluated immediately at that exact scale. A behavior-complete but Wiki-unsafe scale is not accepted; smaller scales are tried.

## Final acceptance

A development candidate must simultaneously satisfy:

- generated direct success = 100%;
- generated other-atomic success = 100%;
- zero pairwise-margin failures at margin `0.01`;
- hard `P` regressions = 0;
- hard `P` non-sensitive KL <= `0.05`;
- external-Wikipedia selection KL mean/p95/max <= `0.01 / 0.05 / 0.5`.

Only after checkpoint selection is fixed is the disjoint fresh-1000 Wikipedia gate opened. No official RWKU paraphrase, neighborhood, retain, or PPL artifacts are used by the learner or selector.

## Explicitly absent

- no Stage-2 embedding training;
- no transformer update;
- no MLP/attention edit;
- no LoRA;
- no Level 3;
- no representation repair;
- no utility-budget relaxation.
