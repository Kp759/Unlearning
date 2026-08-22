# Directional SURE v2: embedding-GA-only ablation

This is a one-factor post-hoc development ablation from the original content-sensitive Directional SURE v2 configuration.

## Unchanged from Directional SURE v2

- target-only generated atomic training views;
- content-sensitive sensitive-answer row selector on both input embeddings and untied LM head;
- all transformer parameters exactly frozen;
- non-selected embedding and LM-head rows exactly Base;
- embedding LR `5e-5` and LM-head LR `1e-4`;
- 600 steps, batch size 1, AdamW, zero weight decay, gradient clip 1;
- GA weight 2 and GD weight 1;
- dynamic sensitive-exclusive basis `B_S` rank 8;
- protected basis `B_P` rank 32;
- basis refresh every 25 steps;
- external-Wikipedia basis/selection/fresh slices 256/256/1000;
- utility budgets KL mean/p95/max `0.01/0.05/0.5`;
- official RWKU evaluation artifacts unavailable to learning and checkpoint selection.

## Single changed factor

Original v2 embedding gradient:

\[
g_E = 2 g_{GA,E} + g_{GD,E}.
\]

This ablation:

\[
\boxed{g_E = 2 g_{GA,E}}.
\]

The embedding GD gradient is still computed and logged for diagnosis, but it is not assigned to `input_delta.grad` and therefore never reaches `optimizer.step()`.

The LM-head update remains exactly the original directional rule:

\[
\boxed{g_W = 2\Pi_{B_S}(g_{GA,W}) + \Pi_{B_P}(g_{GD,W})}.
\]

Thus GD is removed only from the embedding branch, not from the LM-head branch.

## Scientific question

The experiment asks whether unconstrained embedding-side GD was opposing sensitive suppression or otherwise degrading the optimization tradeoff, while preserving the original output-side locality mechanism `GD -> B_P`.

Interpretation should compare the entire generated-atomic forgetting versus external-Wikipedia KL trajectory to the original v2 baseline. A utility-safe plateau below 100% is still a useful development result; it is not evidence of knowledge erasure or benchmark-robust unlearning.
