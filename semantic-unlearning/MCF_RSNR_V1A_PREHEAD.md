# MCF RSNR-V1A-PreHead

Development-only controlled ablation of Relation-Scoped Null Routing (RSNR).

## Question

Does RSNR require an internal decoder-layer intervention, or is a gated residual
adapter on the final hidden state immediately before the frozen LM head enough?

## Fixed comparison contract

RSNR-V1A-PreHead keeps the layer-24 RSNR-V1A experiment fixed except for the
intervention site:

- Base model: Llama-3.2-3B-Instruct.
- Consumed seed: 1 (development only).
- Forget cases: 50.
- Training views: exact locked V1.3 five-view corpus, 5 views/case.
- Oracle gate: exact `(subject, relation_id)` forget membership.
- Adapter: rank 16, alpha 16, 98,304 trainable parameters.
- Base embeddings: frozen.
- All Transformer blocks: frozen.
- Final norm: frozen.
- LM head weights: frozen.
- Loss: abstention + true-answer unlikelihood + adapter anchor.
- `target_new`: never used for training.
- Training gates: all five views per fact must satisfy
  - `log P(IDK) - log P(true) >= 0.1`, and
  - `log P_base(true) - log P_prehead(true) >= 2.0`.
- Gate-off path must have zero logit drift.

## Intervention

Let `h_L` be the final normalized hidden state passed to the LM head.  For a
sensitive query:

```math
\tilde h_L = h_L + A_{NULL}(h_L)
```

with

```math
A_{NULL}(h) = W_{up}\tanh(W_{down}h),
```

where `rank(W_down)=16`.  The frozen LM head then computes

```math
z = W_{LM}\tilde h_L.
```

For a non-sensitive query the gate is zero, so

```math
\tilde h_L = h_L
```

and the exact Base output path is recovered.

## Evaluation

Primary method-aligned metrics:

- `Eff_IDK`: canonical prompts where `P(true) >= P(IDK)`; lower is better.
- `Gen_IDK`: unseen official paraphrases where `P(true) >= P(IDK)`; lower is better.
- `Sensitive_Eff`: canonical greedy true/alias leakage rate; lower is better.
- `Sensitive_Gen`: unseen-paraphrase greedy true/alias leakage rate; lower is better.
- Base-to-RSNR true-answer log-probability drop.
- Greedy abstention rate.
- Adversarial retrieval leakage under refusal-suppression prompts.
- Stochastic leakage at multiple temperatures.
- Fresh-disjoint-retain utility and Wikidata PPL.

Legacy CounterFact Eff/Gen are retained only for historical comparability; they
compare `target_true` with CounterFact `target_new` and are not the primary RSNR
nondisclosure metrics.

## Claim boundary

This protocol tests conditional behavioral suppression / nondisclosure.  It does
not claim latent knowledge deletion because disabling the gate restores the
frozen Base model.
