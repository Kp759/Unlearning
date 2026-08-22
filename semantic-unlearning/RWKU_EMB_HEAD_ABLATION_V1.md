# RWKU embedding/head ablation v1

Development-only Stephen King seed-0 ablation derived from frozen RWKU v3.2 commit `dc30b024ca092583f40a7e2ee2b15f236e9f449d`.

## Variants

1. `emb_head`: start from the same sparse Stage-1 head edit as v3.2; train the full input-embedding matrix and untied LM head. Transformer blocks remain frozen.
2. `emb_head_downproj`: same as (1), plus rank-1 LoRA on the final transformer MLP `down_proj`.

Both variants keep an immutable copy of the original tied vocabulary matrix `W0`. Frozen-decoder recovery is always evaluated with this untouched `W0`, even though the deployed input embeddings and LM head are trainable.

## Locked optimization

- 300 steps, seed 0, BF16 single device.
- embedding/head AdamW LR: `1e-4` (the original SURE embedding/head scale).
- final `down_proj` LoRA LR: `5e-4`, rank 1, alpha 1 for the composite variant.
- answer-level frozen-`W0` hinge: weight 8, target margin 0.5.
- edited-head answer hinge: weight 2, target margin 0.5.
- Wikipedia hidden preservation: weight 2.
- exact full-vocabulary `KL(P_base || P_edit)`: weight 50.
- 1000 optimization-only Wikipedia contexts; 4 utility contexts per step.

## Utility boundary

The 1000-context held-out Wiki gate already opened for v3.2 is excluded from checkpoint selection. The ablations skip those first 1000 guard indices and use the next predeclared 1000 guard contexts only after selecting a checkpoint from generated sensitive views + the optimization Wiki pool.

The launcher fixes both variant implementations and hyperparameters before either run; do not alter the second variant based on the first variant's confirmatory result.

## Reported norms

- embedding relative Frobenius drift from Base;
- LM-head relative Frobenius drift from the inherited Stage-1 head and from Base;
- final `down_proj` relative Frobenius drift.

Embedding/head drift is descriptive in this exploratory ablation. The composite variant retains the existing `down_proj <= 1%` gate.

## Output comparison

`scripts/summarize_rwku_emb_head_ablation_v1.py` prints:

- frozen-`W0` recovery;
- minimum frozen-`W0` demotion margin;
- fresh confirmatory Wiki KL mean/p95/max;
- embedding, LM-head, and `down_proj` relative Frobenius drift.
