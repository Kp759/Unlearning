# MCF Private Vocabulary Rewiring V1.2 — True-Target Suppression

V1.2 is an objective-only ablation of V1.1. The position-preserving private-subject architecture is unchanged: every original subject token is replaced by one private reserved token at the same position, private rows are initialized as exact copies of the original rows, and the Transformer, LM head, and all non-private input rows remain frozen.

## Forget objective

For forget case `i`, let `b_i` be the frozen-Base mean answer-token log-probability of `target_true`, and let `q_i(theta)` be the current mean answer-token log-probability under the private route. V1.2 minimizes

`relu(q_i(theta) - (b_i - delta))^2`

with default `delta = 4.0` nats per answer token. `target_new` never contributes gradient. A registered development suppression success is a Base-relative true-target log-probability drop of at least `2.0` nats per answer token.

The old target-new minus target-true margin is still measured, but only as a diagnostic so V1.2 can be compared with V1.1 and the existing official MCF evaluator.

## Retain objective and constraints

V1.2 keeps V1.1's training-safe same-subject different-relation Base-KL preservation bank, top-64 teacher KL, anchor term, and hard relative private-row movement cap of `0.5`.

## Claim boundary

This remains behavioral unlearning / materialized tokenizer-and-embedding rewiring, not proven latent knowledge deletion. Restoring the original subject token IDs bypasses the private route. There is no relation-aware inference router.

Seed 1 is architecture-development evidence because its official paraphrases were already opened during V1.1 evaluation. A later clean seed is required for a held-out Gen claim.
