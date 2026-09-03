# MCF Private Vocabulary Rewiring V1.1

## Decision

V1 failed before unlearning because a complete multi-token subject was compressed
into one private token.  For example, a five-token subject became one token,
shifting all following token positions and RoPE indices.  Clone distillation
reduced KL but saturated the 0.5 row cap far above the registered equivalence
threshold.

V1.1 removes that confound.  If Base tokenizes a subject as

```text
[t1, t2, ..., tk]
```

V1.1 allocates the same number of reserved private ids

```text
[p1, p2, ..., pk]
```

and initializes

```text
E[p1] = E[t1]
E[p2] = E[t2]
...
E[pk] = E[tk]
```

exactly.

## What is keyed

The private capacity belongs to the **subject**, not to a subject-relation pair.
The same private subject sequence is used in every context.  The lexical rewrite
never inspects relation text and never decides whether a prompt is a forget
query.

```text
private subject + forgotten relation -> forgetting objective
private subject + other relation     -> Base-preservation objective
```

The frozen Transformer must provide relation selectivity.

## Position preservation

The rewrite is a deterministic longest-match replacement over Base token ids:

```text
[t1, ..., tk] -> [p1, ..., pk]
```

Because both sequences have the same length, V1.1 preserves:

- the number of subject positions;
- the positions of every following relation/context token;
- RoPE indices of all following tokens;
- the initial embedding vector at every subject position.

Before any optimization, V1.1 requires near-zero Base/private top-k KL and
near-zero direct-margin drift.  There is no clone-distillation stage.

## Trainable parameters

Only the compact bank of private subject rows is trainable.  The following stay
frozen:

- every original lexical input-embedding row;
- the complete Transformer;
- the untied LM head / unembedding, which is hash-verified bit-identical.

Private rows are materialized into reserved vocabulary slots after training.

## Routing boundary

V1.1 currently implements the same-length subject rewrite as a deterministic
token-sequence tokenizer wrapper plus a saved `private_subject_routing.json`
manifest.  This routing is part of the method and must be disclosed.  It is
subject-only and relation-agnostic, but restoring the original subject token ids
can bypass the behavioral edit.  Therefore V1.1 supports a behavioral-unlearning
claim only, not latent knowledge deletion.

A later packaging revision may internalize the same deterministic rewrite into a
standalone tokenizer artifact if the experiment succeeds; that packaging change
must not alter the learning protocol.

## Registered first run

- seed: 1 (architecture-development only)
- forget facts: 50
- official retain: 1000 reserved and invisible
- protection fit/development/certification: 2000/500/1000
- direct forget success margin: >= 0.1 for every fact
- initial equivalence top-k KL: <= 1e-7
- initial direct-margin drift: <= 1e-5
- retain mean top-k KL after training: <= 1e-4
- private relative row cap: 0.5
- steps: 600

The first scientific question is whether exact position-preserving private
capacity can move the forgotten subject-relation association while preserving
other behavior of the same subject through the frozen Transformer.
