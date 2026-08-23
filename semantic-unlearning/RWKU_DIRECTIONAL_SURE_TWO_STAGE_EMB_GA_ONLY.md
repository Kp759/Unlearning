# RWKU Two-Stage Directional SURE: Embedding-GA-Only

This is a post-hoc Stephen King development ablation. Official RWKU evaluation remains locked.

## Level 1

The row policy is the original Directional SURE v2 content-sensitive vocabulary selector for both input embeddings and the untied LM head. The transformer is exactly frozen.

Embedding update:

\[
g_E^{(1)} = 2 g_{GA,E}.
\]

Embedding-side GD is computed and logged for audit, but is not applied.

LM-head update:

\[
g_W^{(1)} = 2\Pi_{B_S}(g_{GA,W}) + \Pi_{B_P}(g_{GD,W}).
\]

Among checkpoints passing the unchanged external-Wikipedia selection KL budgets, the Level-1 anchor minimizes margin/behavior failures before utility KL and norm tie-breakers.

## Level 2

Level 2 is used only when Level 1 has a utility-safe anchor but has not reached 100% direct + 100% generated-other atomic acceptance with zero margin failures.

The residual prompt set is frozen from that Level-1 anchor. The editable Level-2 rows are the intersection of rows implicated by those residual prompts and the original content-sensitive Level-1 row set.

Embedding update:

\[
g_E^{(2)} = M_F \odot 2 g_{GA,E}.
\]

LM-head update:

\[
g_W^{(2)} = M_F \odot \left[2\Pi_{B_F}(g_{GA,W}) + \Pi_{B_P}(g_{GD,W})\right].
\]

The optimizer is reset at Level 2. The existing half-rate Level-2 learning rates, 300-step budget, B_F rank 8, B_P rank 32, and all utility/acceptance budgets are unchanged.

## What does not change

- Transformer, MLP, attention, and LoRA parameters remain frozen.
- No Level 3 or representation repair exists.
- Non-selected vocabulary rows remain exact Base.
- External Wiki protected/selection/fresh slices remain disjoint and target-excluded.
- Official RWKU paraphrase, neighborhood, retain, and PPL artifacts are not available to the learner.
