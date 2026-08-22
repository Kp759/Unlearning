# Directional SURE Embedding Ablations A/B

This branch isolates whether input-embedding updates are necessary once the LM-head update is already directional.

Both variants retain the same pure two-stage architecture and never modify transformer, MLP, attention, or LoRA parameters.

## Common LM-head rule

For every Level-1 sensitive prediction case, canonical SURE GA and same-prompt non-sensitive GD KL are computed. The LM-head gradient is

\[
g_W^{(1)} = 2\Pi_{B_S}(g_{GA,W}) + \Pi_{B_P}(g_{GD,W}).
\]

Level 2 is triggered only from a Level-1 checkpoint that passes the unchanged external-Wikipedia selection utility budgets. On residual failures it uses

\[
g_W^{(2)} = M_F \odot [2\Pi_{B_F}(g_{GA,W}) + \Pi_{B_P}(g_{GD,W})].
\]

The LM head always exposes every non-special token row observed in teacher-forced sensitive answers.

## Variant A — frozen input embeddings

Input embedding weights remain exact Base. The learner preserves the same sparse-master shape as the canonical two-stage implementation so existing residual-row invariants remain unchanged, but an exact all-zero gradient mask is registered on the input sparse delta. With zero weight decay, the input delta stays identically zero throughout Level 1 and Level 2 and materializes back to Base exactly.

Thus the effective trainable object is only the all-sensitive LM-head sparse delta.

## Variant B — content-safe input rows

The LM head is identical to Variant A. Input embeddings may change only for the existing locked content-sensitive answer-token subset used by the earlier content-row policy. This excludes punctuation-only pieces and the locked common-function-token set while retaining content-bearing sensitive answer pieces.

The sparse input master still has the full sensitive-row shape, but a rowwise gradient mask permits nonzero updates only on the content-safe subset. Embedding gradients use ordinary weighted GA+GD and are not projected into the final-hidden B_S/B_P bases.

## Controlled factors

The following are identical between A and B:

- generated target-only atomic training corpus;
- all-sensitive LM-head row policy;
- 600 Level-1 steps;
- LM-head LR 1e-4;
- GA/GD weights 2/1;
- rank(B_S)=8;
- rank(B_P)=32;
- 25-step basis/checkpoint refresh;
- Level-2 trigger and residual logic;
- 300 Level-2 steps;
- Level-2 LM-head LR 5e-5;
- rank(B_F)=8;
- exact external-Wikipedia utility budgets mean/p95/max = 0.01/0.05/0.5;
- transformer exact freeze;
- no Level 3 / SURE-R;
- no official RWKU evaluation access during learning or selection.

The intended comparison is therefore whether embedding intervention improves forgetting enough to justify the additional utility exposure.

## Expected interpretation

If A preserves utility and reaches strong forgetting, input-embedding editing is unnecessary for the primary method.

If A is too weak but B reaches substantially stronger forgetting while staying within the same utility budgets, content-safe embeddings provide justified additional capacity.

If B again causes early utility failure, the evidence favors keeping input embeddings frozen and improving only the output-side directional/residual solver rather than broadening input-row edits.
