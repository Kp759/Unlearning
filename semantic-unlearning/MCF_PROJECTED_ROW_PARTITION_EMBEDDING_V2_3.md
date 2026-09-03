# MCF frozen-head projected row-partition embedding rewiring V2.3

## Purpose

V2.3 is the frozen-LM-head sibling of V2.2. The Transformer is frozen, the LM
head is untied and must remain bit-identical, and only selected input-embedding
rows may change. There is no detector, classifier, router, sidecar, adapter, or
conditional inference branch.

The head is still *used* — as the closed-form answer-contrast reference and as
the reader whose retain-context outputs define the protected subspace — but it
is never an edit site.

## Why the head is frozen

`logits = W h` has no context input of its own, so a single `Delta W[a]` moves
that answer's logit in every context at once. On CounterFact the neighborhood
prompts that define specificity have different subjects but the same correct
answer, so efficacy and specificity are coupled through one shared parameter.

Three registered attempts tried to defeat that coupling geometrically:

- V2 found a preservation-safe first-order direction for all 50 facts but never
  produced a development-passing checkpoint;
- V2.1's folded head reached 5 direct failures but its protected correction was
  `1.741288` against a `0.02` ceiling, and floors 8 through 24 were identical,
  showing output-cap saturation;
- V2.2 keeps the trainable head under a joint retain trust region and remains
  unexecuted.

V2.3 removes the coupling by construction and spends the whole budget on the
input side, where the frozen Transformer supplies context dependence for free
and a prompt containing none of the edited rows is bit-identical to Base. Run
against V2.2 it forms a clean pair: if V2.3 reaches the same envelope, output
row capacity was never required.

## 1. Answer-contrast direction

The earlier subject-embedding stage confined the hidden displacement to
`u_s = normalize(W[target_true])`. Under a frozen head that is the wrong axis.
The trained margin is exactly

```text
log p(target_new_i) - log p(target_true_i) = h . (W[target_new_i] - W[target_true_i])
```

because the softmax normalizer cancels in the difference. V2.3 therefore uses

```text
q_i = normalize(W[target_new_i] - W[target_true_i])
```

with a separate `q_i` at each answer-token position. CounterFact answers often
share a leading token and are frequently of unequal length, so positions where
the difference is degenerate are masked out rather than normalized into noise.
Position `i` is read at the teacher-forced `target_true` prefix, matching the
margin rule the run already reports.

The optional surgical loss is

```text
L_surgical = || dh - (dh . q_i) q_i ||^2 + relu(gamma - dh . q_i)
```

The first term removes movement unrelated to the trained margin; the second
enforces its sign. It overlaps with the primary hinge, so the primary arm sets
`surgical_weight = 0` and it is enabled only as a separately registered
ablation.

## 2. Row partition by efficacy and potential

Angle alone is not a sufficient row criterion: a row can be almost orthogonal
to the retain subspace and still carry a forget gradient too small to move the
margin inside its own cap. Every candidate row is therefore scored twice:

```text
efficacy_t  = || P_perp,Gr(t) g_f(t) || / || g_f(t) ||
potential_t = cap_t * || P_perp,Gr(t) g_f(t) ||
```

and partitioned:

| Role | Condition | Treatment |
|---|---|---|
| `free` | no retain observation after targeted per-row coverage | edited without constraint |
| `projected` | retain-observed, usable efficacy and potential | kept orthogonal to its own retain basis, capped |
| `excluded` | near-parallel, negligible potential, or above the frequency bound | held at exact zero |

Forget-exclusive rows are the safe, high-value lever — no gradient path from a
pure retain prompt reaches them. Shared rows are the risky subset, which is why
they are constrained rather than targeted.

**Direct-prompt liveness is still mandatory.** Efficacy is scored on the direct
prompt, so a record whose every direct-prompt row was excluded would be
unfixable by construction. Such a record reinstates its highest-potential direct
row as `projected` and is reported as liveness-forced, higher-collateral. This
preserves the guarantee the subject-embedding stage established at `6dcb11f`,
where a permanently unedited record pinned the minimum margin at `-10.671875`
across three runs.

## 3. An empty basis is not proof of a forget-exclusive row

V2 recorded 141 of 236 input rows with an empty protected basis. That meant
"unobserved by 64 Jacobian sketches", not "never used by retained behavior".
V2.3 does not free a row on that evidence alone. Before any row is classified
forget-exclusive it must survive targeted coverage: the fit bank already
allocates dedicated corpus occurrences of that exact token row, ordinary retain
facts, and same-subject/different-relation contexts. Rows that remain
unobserved *and* were absent from the corpus entirely are counted and reported
separately rather than silently freed.

The retain subspace is also built from the three quantities the acceptance
gates actually measure — Base top-1 log-probability, the *centered* top-k logit
(the shape top-k KL responds to), and each labelled retain answer's
log-probability — rather than V2's single top-k column logit.

## 4. Projection is a warm start, not a certificate

The projection is a first-order guarantee inside the represented subspace. The
Transformer is nonlinear, so every proposed step is

```text
Delta E_t' = P_perp,Gr(t) ( Delta E_t - alpha * grad_t L_F )
```

followed by cap projection and then an **actual forward evaluation** of the
active retain bank. A step is accepted only under the registered trust region:
when the bank is feasible a step must stay feasible and improve forgetting;
when it is infeasible a repair step must strictly reduce normalized constraint
violation while increasing forget loss by at most one percent. Individual rows
can be first-order safe while their combined nonlinear effect is not, which is
precisely the failure V2.1 recorded.

Every fifty steps the complete fit bank is re-evaluated and the worst 32
prompts merge into a persistent 256-prompt tail, with 16 active places reserved
for the tail, 16 for explicit overlaps, and 16 for ordinary replay.

The projected-gradient principle is well founded but local to the represented
protected subspace, as in Orthogonal Gradient Descent (arXiv:1910.07104).
LEACE (arXiv:2306.03819) likewise states its exact erasure guarantee in the
linear setting. Neither is extrapolated through the frozen nonlinear stack.

## 5. Joint objective

Every embedding update contains both forget and retain examples. Preservation
is an optimization constraint, never a post-hoc audit. The retain side carries
ordinary retained facts, same-subject/different-relation prompts, Wikipedia
occurrences of every selected input row, generic Wikipedia prompts, and the
persistent worst-tail bank.

Adam is prohibited; steps are row-normalized, frequency-cap scaled, and
backtracked.

## 6. Firewall and claim

Development selects a candidate only when direct, synthetic, fit protection,
development protection, row caps, and role compliance pass simultaneously.
Certification opens once afterwards. Official evaluation remains unavailable to
this process.

The central claim is deliberately conditional:

> Static input-embedding edits achieve context-selective behavioral forgetting
> when the forget gradient retains a usable component outside the retained
> functional subspace, and the `(efficacy, potential)` partition diagnostic
> predicts when common subwords permit or prevent locality.

The diagnostic's prediction is recorded in `method/row_partition.json` before
training, so it can be falsified rather than fitted after the fact.

### The alias limitation is structural

A prompt containing none of the edited token ids produces **exactly** Base
behavior. This is the same fact that makes locality combinatorial, and it cuts
both ways:

- indirect questions containing the original subject tokens may generalize;
- an alias sharing edited subwords may generalize;
- a completely token-disjoint alias cannot be affected at all.

Covering aliases while remaining embedding-only requires a training-safe alias
lexicon registered before the architecture is frozen. Blind token-disjoint
aliases are a hard limitation of the parameterization, not an optimization
problem to be tuned away.

Because the Transformer stays frozen, a passing run supports behavioral,
context-selective forgetting. It does not establish latent erasure: a probe on
the frozen stack may still recover the fact. Held-out aliases, indirect
questions, adversarial extraction, logit-lens latent recovery, and relearning
remain mandatory post-freeze endpoints.

## Running

```bash
MODEL_PATH=... WIKIDATA_DIR=... MCF_PATH=... \
  bash scripts/run_mcf_projected_row_partition_embedding_v2_3_manual.sh \
  outputs/mcf_projected_row_partition_v2_3_seed1
```

The split builder is the only process that sees `MCF_PATH`; the training
process runs with official/recovery environment variables unset and refuses to
start if any are populated.
