# MCF internal fact-conditional embedding rewiring V1

## Status

This document registers the next MCF research architecture. Implementation and
evaluation are pending. It does **not** report a new result.

The V5 exact-string/logit-bias and V6 normalization-preserving sidecars are
retired as research methods. Their code, registries, and outputs remain in the
repository as historical controls, but no additional confirmation seeds should
be spent on them. They provide behavioral suppression only while an external
router is attached; they do not erase an internal representation.

The replacement stays inside the model's causal path:

```text
overlap-aware rank-8 subject embedding code
        |
        v
frozen early Transformer blocks
        |
        v
internal multi-label subject x relation classifier
  (8 detector neurons per fact at one certified mid-layer)
        |
        v
calibrated fail-closed semantic gate
        |
        v
16 detector-disjoint actuator neurons per fact
        |
        v
sparse residual rewiring
        |
        v
frozen later blocks + untied, bit-identical frozen LM head
```

There is no prompt-string router, retrieval table, inference-time sidecar,
constant logit bias, tokenizer expansion, or answer-row LM-head edit.

## Why this is the restart point

The strongest internal result in the MCF lineage is the subject-embedding run
recorded at `791bbc4`: across ten seeds, Stage 1 reached `Eff = 0.0 +/- 0.0`,
`Gen = 1.7 +/- 2.0`, and Base-like PPL. Its specificity failures were strongly
associated with common subject subwords. That is evidence for retaining input
embeddings while replacing independent row edits with a joint overlap-aware
solver.

The other lineage results constrain the design:

- Global target-answer LM-head edits are structurally non-local because the
  same answer row is used by unrelated facts. The output reader experiment
  caused catastrophic PPL while the input-only arm retained Base PPL.
- The low-variance hidden-direction marker tested at `d577a1d` was not
  reachable from the allowed embedding edits. This rejects that particular
  marker geometry, not subject-embedding editing in general.
- V3.5.4 showed that canonical multi-label semantics can certify all 50
  contextual detectors. V3.5.5 showed that four detector features were enough
  for routing, while a separate width-16 actuator was the first registered
  actuator to reach the positive certificate at the fixed 1.50 column cap.
- V3.6.2 passed training-only preservation but failed official paraphrase and
  locality checks. Its raw detector thresholds therefore must not be carried
  into a new representation or layer.
- V4's independent 3,072-dimensional per-record classifier memorized its fit
  cells and failed held-out positives and negatives. The replacement must pool
  statistical strength through shared low-rank projections and substantially
  larger hard-negative banks.

## 1. Overlap-aware input-embedding code

### Rows that may change

For each forget subject, collect every tokenizer row in both sentence-initial
and whitespace-prefixed tokenizations of the complete subject. The primary
method may change these input-embedding rows only. It must not edit:

- answer-token embeddings merely because they encode `target_true` or
  `target_new`;
- any LM-head row;
- tokenizer vocabulary or tokenizer rules; or
- an embedding row independently once per record.

The tied input/output matrix is first untied by copying the original tensor.
The copied LM head is frozen and must remain bit-identical throughout training.

### One coherent delta for every shared row

Let `A` be the subject-by-subword incidence matrix. `A[r,t]` records the
normalized occurrence of token row `t` in the complete subject for record `r`.
Let `K` be a fixed, seeded rank-8 subject-code matrix, and let `C[t]` be the
rank-8 coefficient assigned to editable token row `t`.

Solve the frequency-weighted ridge problem once for all subjects:

```text
min_C ||A C - K||_F^2 + lambda * sum_t (1 + f_t)^alpha ||C[t]||_2^2
```

where `f_t` is measured only on the registered training-visible corpus slice.
Map coefficients to embedding space with a learned or fixed orthonormal basis
`B`:

```text
Delta E[t] = B C[t]
```

and project every row to the registered frequency-aware cap:

```text
||Delta E[t]||_2 / ||E_base[t]||_2 <= c / (1 + f_t)^alpha
```

This gives a common token such as `the`, `Robert`, or `Jean` one coherent
compromise delta rather than several contradictory per-record deltas. Complete
subject codes are distinguished by the *combination* of their subwords. No
record may be silently dropped because all of its rows are shared.

The overlap manifest must report token frequencies, all owning records,
condition number, reconstruction error per record, row caps, and the exact
Base/edit hashes. A same-subject/different-relation collateral audit is a
mandatory pre-actuator gate.

## 2. Internal sensitivity classifier

The classifier answers a multi-label question at runtime:

> Does this contextual state express forget fact `r`, rather than merely
> contain one of its subwords?

At a candidate mid-layer, shared rank-8 subject and relation projections read
the complete-subject span and the query/prediction state. A small per-fact
factorized head combines subject evidence, relation evidence, and their
interaction. Its eight detector neurons implement an AND-like decision; a
subject match alone or relation match alone is insufficient.

The target is a vector, not a record-relative scalar. If the same canonical
prompt belongs to multiple roles, it receives one canonical multi-label vector.
An exact forget-positive label takes precedence over a contradictory
preservation-negative label for that same fact and forward state; the conflict
is reported rather than presented to the optimizer.

### Required hard-negative families

Every fact must be trained and certified against:

- the same subject with a different relation;
- the same relation with a different subject;
- the same answer with a different subject;
- prompts containing only shared subject subwords;
- complete-subject aliases that refer to a different entity;
- unrelated corpus prefixes and broad corpus prompts; and
- writer-off/Base states for every positive context.

The classifier may not use an independent full-hidden-size vector per record.
Shared projections and the rank-8 bottleneck are binding capacity constraints.

## 3. Layer and threshold selection

Only layers `{8, 12, 16, 20}` are candidates. Layer selection uses classifier
fit/development/certification evidence only; actuator or behavioral metrics may
not choose the layer.

The old raw thresholds `0.20/0.25` are not portable to a new classifier. For
each frozen classifier candidate:

1. Fit its parameters on the training split.
2. Standardize scores using fit-split statistics only.
3. On a disjoint calibration bank, set
   `tau = nextafter(max_negative_score, +infinity)`.
4. Freeze `tau` and open a third, disjoint certification bank.
5. Require zero negative/cross/writer-off gate activations on at least 300,000
   certification cells, and require every registered positive/alias cell to
   exceed `tau`.

With zero errors in 300,000 independent negative cells, the rule-of-three 95%
upper bound is approximately `1e-5`. This is a measured bound on the registered
negative distribution, not a universal safety guarantee. Ambiguous scores and
unknown subjects fail closed. The raw numerical value of `tau` is an
artifact-specific output, never a hand-tuned global constant.

## 4. Sparse actuator

Once the classifier, layer, threshold, and embedding code are frozen, select
16 detector-disjoint actuator neurons per fact at the certified layer. The
primary residual is:

```text
Delta h_r = gate_r(h) * sum_j activation_rj(h) * Delta W_down[r,j]
```

Actuator columns start at exact zero and use the inherited native per-column
relative cap of 1.50. Base MLP columns remain unchanged. Only the additive
actuator residual is trained. No actuator is constructed before classifier
certification passes.

The primary objective suppresses `target_true` while preserving:

- coherent negatives;
- writer-off/Base positives;
- same-subject/different-relation prompts;
- protected corpus prompts, including a refreshed hard tail; and
- the original LM head and all non-registered model parameters.

The primary method does not optimize `target_new` and does not promote a
replacement answer. A constrained target-true reader edit may be studied only
as a separately labeled ablation after the internal method is frozen.

## 5. Locked execution order

1. Build and hash the complete-subject/subword overlap manifest.
2. Untie and freeze the bit-identical LM head.
3. Solve the rank-8 joint embedding code from exact Base.
4. Cache fit/development/certification states at all four candidate layers.
5. Fit shared factorized classifiers without constructing actuators.
6. Select one layer using classifier criteria only.
7. Calibrate and freeze its threshold; run the one-shot certification bank.
8. Freeze embedding and detector tensors.
9. Select 16 disjoint actuators and run positive reachability from exact zero.
10. If reachable, train the full preservation objective and freeze one
    checkpoint.
11. Evaluate in a separate process bound to checkpoint, protocol, split, and
    Base hashes.

Any failed gate terminates that registered run. Evaluation cannot select a
checkpoint, threshold, layer, neuron bank, cap, or subsequent seed setting.

## 6. Acceptance and claim boundary

Training-only acceptance requires:

- complete embedding-code reconstruction and row-cap certificates;
- zero classifier false positives on at least 300,000 certification cells;
- complete positive and registered-alias classifier coverage;
- zero direct and positive actuator failures;
- exact writer-off and coherent-negative identity where the gate is closed;
- passed protected hard-tail KL limits;
- unchanged detector hashes after detector freeze;
- bit-identical frozen LM head and untouched Base parameters; and
- passed causal ablations showing both the embedding code and actuator are
  necessary.

Behavioral `Eff`, `Gen`, `Spe`, and PPL are necessary but not sufficient for a
strong unlearning claim. The architecture may be described as internal
fact-conditional suppression only until it also passes held-out aliases,
indirect questions, adversarial extraction, latent-recovery probes, sidecar-free
deployment, and relearning tests. The V5/V6 sidecar numbers remain an upper
bound/control and must not be mixed with this internal lineage.
