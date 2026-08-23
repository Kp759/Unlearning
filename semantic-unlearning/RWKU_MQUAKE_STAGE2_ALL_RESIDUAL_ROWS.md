# RWKU MQuAKE-style Stage 2: all residual target rows

This development ablation changes exactly the Stage-2 editable output-row policy relative to `claude/rwku-directional-sure-mquake-style-stage2`.

## Level 1

Level 1 is unchanged and is executed by the previously tested content-sensitive embedding-GA-only learner:

- content-sensitive input embedding rows and LM-head rows only;
- embeddings receive `2 * GA` only;
- LM-head receives `2 * GA -> B_S + GD -> B_P(wiki)`;
- transformer frozen;
- 600 steps, B_S rank 8, external protected rank 32;
- RWKU generated-atomic margin 0.01 and external-Wikipedia utility budgets unchanged.

## Stage 2 row-policy change

Previous MQuAKE-style Stage 2 used:

`A_F = L1_content_rows ∩ residual_target_rows`

The current ablation uses:

`A_F = all non-special sensitive target rows occurring in Level-1 residual prompts`

This is faithful to the intended MQuAKE-style residual-row definition and removes the content-filter controllability bottleneck seen in the previous run (30 residual token cases but only 6 editable output rows).

The runtime sparse output table is the union of the original L1 content rows and the all-residual Stage-2 rows. The L1 anchor is copied only into its original content rows; every newly admitted Stage-2-only row is initialized with exactly zero delta.

## Stage 2 unchanged mechanics

- LM-head only; L1 embeddings frozen;
- transformer frozen;
- P = L1 margin-passing generated prompts;
- F = L1 residual prompts;
- B_P = rowspace(H_P), rank <= 32;
- B_F = rowspace(H_F - Proj_BP(H_F)), rank <= 4;
- repair parameterization `Delta W_AF = C_F B_F`;
- squared 0.01-margin hinge + P-anchor non-sensitive KL + 1e-6 increment L2;
- hard P protection: zero prompt regressions and P KL <= 0.05;
- backtrack scales 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, otherwise rollback;
- unchanged external-Wikipedia selection utility gate;
- no Level 3, MLP, attention, LoRA, or representation repair.

## Utility-reference correction

For this run, external-Wikipedia Base hidden references are cached before the L1 anchor is installed in the reloaded runtime. This restores strict Base-vs-edited utility measurement and avoids the prior runtime's accidental L1-hidden reference.

Official RWKU evaluation remains locked.
