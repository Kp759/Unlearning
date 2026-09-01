# MCF sparse bi-endpoint nullspace rewiring V2

## Status

Implemented replacement for the rejected contextual V1 lineage, awaiting its
first training-only run. This is a transformer-frozen weight-editing framework. It has no detector,
semantic gate, external router, sidecar, adapter, MLP edit, or inference-time
dependency. Only selected rows of the untied input embedding and LM head may
change.

## What the history established

The repository contains more than 260 MCF-specific commits. The decisive
results are consistent:

- global target-answer LM-head edits couple forget efficacy to neighborhood
  specificity when another fact legitimately uses the same answer token;
- full subject-embedding coverage reached Eff 0 on all ten seeds and Gen
  `1.7 +/- 2.0`, but Spe was bimodal because common subject subwords were also
  present in protected prompts;
- context-composed markers and learned contextual classifiers did not transfer
  reliably to unseen phrasings; and
- exact-string/logit sidecars can force benchmark metrics but are behavioral
  wrappers, not internal unlearning.

V2 keeps the successful full-subject input intervention, adds a sparse
LM-head endpoint, and makes shared-row preservation a hard geometric property
of the edit directions rather than a learned runtime classifier.

## Architecture

```text
selected complete-subject embedding rows E[S]
       +
selected target_true/target_new LM-head rows W[A]
       |
       | one coherent delta per physical vocabulary row
       v
frozen Transformer Phi
       |
       v
ordinary materialized model weights (no runtime gate)
```

For forget fact `i`, the training objective redirects the answer from the
sensitive `y_i` to the non-sensitive CounterFact reference `r_i`:

```text
L_forget = relu(m - [log p(r_i | x_i) - log p(y_i | x_i)])
```

The direct prompt and preregistered relation-specific synthetic variants are
training-visible. Official paraphrases, neighborhoods, official retain prompts,
and official PPL text are absent.

## Row selection

Input rows are the union of both sentence-initial and whitespace-prefixed BPE
tokenizations of every complete forget subject. Special tokens are forbidden.
No rarest-token fallback is used: the ten-seed subject-embedding study showed
that partial coverage was the main Gen failure mechanism.

Output rows are the union of every evaluated token in `target_true` and
`target_new`. Input and output weights are untied before any edit. Relation
words are not editable; they are common vocabulary and the frozen Transformer
already supplies the relation-dependent interaction.

There is deliberately no learned ON/OFF threshold. Every selected weight is
part of the ordinary forward pass, and prompt dependence comes only from the
frozen Transformer's interaction with the edited subject and answer rows. The
registered `0.1` forget margin and preservation limits are acceptance
thresholds measured after training, not inference-time routing thresholds.

## Overlap is a constraint, not a label conflict

Each physical token row has exactly one delta even if it belongs to several
forget subjects or appears in protected text. The split builder reserves the
official 1,000-record retain split, then draws disjoint training, development,
and one-shot certification protection pools from the remaining first-half MCF
records. A Wikipedia slice beginning at document 20 supplies disjoint broad and
partial-subword protection; documents 0:20 remain reserved for official PPL.

### Input-embedding rows

For every selected input row `t`, V2 estimates a protected Jacobian sketch
`G_t` from training-safe prompts, including same-subject/different-relation
prompts. The editable delta is projected after every optimizer step:

```text
Delta E_t <- Delta E_t - Proj_rowspace(G_t)(Delta E_t)
```

Thus the first-order change of registered protected logit probes is zero. A
row absent from protection has an empty basis and remains unrestricted. A
shared common row generally receives the largest protection rank.

### LM-head rows

For every selected output row, the protected basis is the row space of frozen
Base hidden states from the training-safe protection bank. Its delta is kept in
the orthogonal complement:

```text
Delta W_a <- Delta W_a - Proj_rowspace(H_protect)(Delta W_a)
```

At the frozen Base states this makes the selected-row logit correction exactly
zero, including legitimate uses of shared answers such as `French`. This is why
V2 does not repeat the old globally suppressive LM-head failure.

These are local geometric guarantees. The Transformer is nonlinear, so V2 also
trains against top-k Base-distribution KL and top-1 log-probability drift, then
requires a disjoint final certificate. No nullspace claim is extrapolated past
the measured protection distribution.

## Feasibility and training order

1. Build a direct-only forget/protection split and hash every partition.
2. Untie the LM head and freeze the complete Transformer.
3. Select and audit input/output rows and every overlap role.
4. Cache Base protection logits and final hidden states.
5. Construct protected input-Jacobian and output-state bases.
6. For each forget record, project its Base margin gradient through both safe
   subspaces. Stop if any record has no nonzero preservation-safe direction.
7. Train sparse embedding and LM-head deltas jointly from exact zero.
8. Project both endpoints and apply frequency-aware relative row caps after
   every optimizer step.
9. Select a checkpoint only from the development bank.
10. Freeze it, open certification once, and perform no further updates.
11. Materialize native-dtype rows and repeat the complete certificate.
12. Save a candidate only if all training-only conditions pass. Official
    evaluation remains a separate hash-bound process.

## Locked training-only gates

- zero direct and synthetic forget failures at margin `>= 0.1`;
- every forget record has a nonzero projected safe gradient;
- one coherent delta per shared input/output row;
- no selected row exceeds its frequency-aware relative cap;
- protected top-k KL mean `<= 1e-4` and maximum `<= 1e-2`;
- protected Base top-1 log-probability absolute drift maximum `<= 5e-2`;
- identical acceptance before and after native-dtype materialization;
- all Transformer parameters bit-identical;
- no official prompt or metric used for row selection, training, thresholding,
  checkpoint selection, or retry.

Passing these gates demonstrates a sparse internal counterfactual rewiring
candidate. It does not by itself prove latent erasure. A strong unlearning
claim still requires blind aliases, indirect questions, adversarial extraction,
latent recovery, and relearning tests on a fresh registered evaluation.
