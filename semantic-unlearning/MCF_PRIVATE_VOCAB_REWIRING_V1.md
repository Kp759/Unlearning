# MCF Private-Vocabulary Rewiring V1

## Hypothesis

Instead of searching for an already-disentangled physical embedding row shared by
forget and retain prompts, create a private parameter address for each forget
subject. V1 repurposes pre-existing Llama reserved tokenizer slots so that the
model shape and output vocabulary do not grow.

For a forget subject `s`, a reserved token id `r_s` is renamed to the literal
subject string by the saved tokenizer. Its trainable input row is initialized
from the mean of the Base token embeddings that previously represented `s`:

`e_private(s, 0) = mean(E_base[tokenize_base(s)])`.

Every original lexical embedding row remains frozen. The Transformer and LM
head remain frozen. During training a compact private-row bank is injected at
the reserved ids; after training the bank is copied into those input rows and
the hook is removed.

## Why reserved slots rather than rare lexical rows

Low frequency is not proof of irrelevance. Reusing a real lexical token could
silently destroy unrelated utility. V1 therefore requires explicitly named
`<|reserved_special_token_N|>` slots and refuses to run if there are not enough.

## Routing rule

The tokenizer itself maps the literal forget subject to its private token id.
There is no relation-aware detector and no per-request sidecar. The same private
row is used for every context containing that subject. Relation selectivity must
therefore come from the frozen Transformer.

This is intentionally different from a query-time rule of the form “if this is a
forget relation, use the edited vector.” Such a rule would reduce the problem to
runtime detection rather than weight-level behavioral unlearning.

## Stage 0: Base-equivalent clone initialization

Before forgetting, each private subject row is distilled toward Base behavior on
training-visible subject contexts. This separates two questions:

1. can a one-row lexical clone reproduce the Base subject sufficiently well?;
2. after that, can the clone be moved to suppress one factual association while
   retaining the subject's other behavior?

The run refuses to continue if the clone-equivalence KL gate is missed.

## Stage 1: private-row unlearning

Only the private subject rows are optimized. The objective combines:

- a direct CounterFact target-new minus target-true margin hinge;
- Base distillation on generic subject contexts and any training-visible
  protection-fit records with the same subject;
- an L2 anchor to the Base-equivalent clone;
- a hard relative row-norm cap.

The registered final direct margin remains `>= 0.1`.

## Materialization and integrity

At completion V1 copies the private vectors into only the chosen reserved input
rows, removes the training hook, and optionally saves a standard Transformers
checkpoint plus the patched tokenizer. It verifies:

- LM-head SHA-256 is identical before and after;
- every non-private input-embedding row is bit-identical before and after;
- the tokenizer vocabulary size is unchanged;
- sampled protection prompts whose subjects are not in the forget set tokenize
  identically under Base and private tokenizers.

## Claim boundary

A successful result supports **materialized behavioral unlearning by private
vocabulary rewiring**. It does not prove deletion of latent knowledge from the
frozen Transformer. Restoring the original tokenizer can expose the original
subject route, so tokenizer-bypass, aliases, misspellings, and token-disjoint
references remain explicit robustness tests rather than assumed coverage.
