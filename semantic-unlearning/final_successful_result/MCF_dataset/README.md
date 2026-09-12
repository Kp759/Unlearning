# Final Successful Result — MCF Dataset

## Frozen result

This file freezes the successful MCF result for the fact-association embedding bank before moving to zsRE.

- Dataset: MultiCounterFact (MCF)
- Model: Llama-3.2-3B-Instruct
- Forget sample size: 50 facts
- Retain sample size used by official evaluation: 1000 facts
- Sampling seed: 1
- Intervention layer: 19
- Trainable objects: 50 independent fact-specific residual vectors
- Base transformer: frozen
- Input embeddings: frozen
- LM head: frozen
- Tokenizer: unchanged
- External fact-ID/private-token injection: none
- Primary metric version: `zerounlearn_answer_probability_v2`
- Primary forgetting status: **PASS — display-zero Eff/Gen**
- Original frozen V1 run directory:
  `outputs/static_overlap_fact_association_embeddings_v1_hierarchical_seed1`

This result should be treated as the frozen MCF suppression result. Subsequent work should not tune this checkpoint further merely to reduce the legacy target_true-vs-target_new preference diagnostic.

---

## Architecture

Each forget fact owns one independent learned residual vector.

For fact (i):

$
F_i = (s_i, r_i, o_i)
$

where:

- (s_i) is the subject,
- (r_i) is the relation/context of the fact,
- (o_i) is the sensitive object/answer.

The object is used as the sensitive answer during training. It is **not required as a runtime trigger**.

With 50 forget facts:

$
\Delta e_1, \Delta e_2, \ldots, \Delta e_{50}
$

are the only learned fact-specific intervention vectors.

### Runtime path

```text
Natural prompt
    |
    v
Frozen Llama layers 0 ... 18
    |
    v
Layer-19 hidden state
    |
    v
V1 association router
    |
    +-- no forgotten subject match
    |       |
    |       +--> exact frozen-base path
    |
    +-- forgotten subject match
            |
            +-- exactly one bank fact owns subject
            |       |
            |       +--> directly select its fact vector
            |
            +-- multiple forgotten facts own subject
                    |
                    +--> frozen semantic relation-key selection
                            |
                            +--> select fact vector
    |
    v
Add selected vector only at original prompt-boundary position

    h'_{19,T-1} = h_{19,T-1} + \Delta e_i

All other positions remain unchanged.

    |
    v
Frozen remaining transformer layers
    |
    v
Frozen LM head
    |
    v
Output
```

If no route activates:

$
h' = h
$

so the model follows the exact frozen-base path.

### Important V1 locality limitation

For a subject appearing in only one forget-bank record, V1 activates that vector whenever the complete subject token sequence occurs, without requiring relation confirmation.

Therefore:

```text
unique forgotten subject
        ->
vector activation
```

even for some same-subject / different-relation prompts.

This is the main architectural issue left after the MCF forgetting objective was solved and is the motivation for the relation-sensitive V2 router.

---

## Why layer 19

Layer 19 is a fixed middle-to-late residual-stream intervention point.

The design motivation is:

- very early layers are more lexical/local and leave many downstream transformations that can wash out a fact-specific intervention;
- very late layers are close to the output decision and can become brittle or overly logit-specific;
- layer 19 provides a representation that is sufficiently semantic while still leaving downstream frozen layers to propagate the residual edit.

The current result does **not** establish that layer 19 is globally optimal. It is a configured working intervention layer. A layer ablation would be required to claim optimality.

---

## Training objective

Training suppresses the original sensitive answer itself.

For a prompt (x) and answer tokens

$
y=(y_1,\ldots,y_m),
$

the training scorer uses mean teacher-forced answer-token NLL:

$
\bar L(x,y)
=
\frac{1}{m}
\sum_{t=1}^{m}
-\log p(y_t\mid x,y_{<t}).
$

The historical training quantity

$
\exp(-\bar L)
$

is the geometric mean of the answer-token probabilities.

The V1 absolute suppression target is:

$
\exp(-\bar L) < 10^{-6},
$

equivalently:

$
\bar L > -\log(10^{-6}) \approx 13.8155.
$

After sufficient sensitive-answer suppression, the optimizer may optimize the abstention completion `" I don't know."` while preserving the suppression constraint.

The primary MCF result below does **not** require optimizing toward `target_new`.

---

# Primary MCF metric contract

The final MCF reporting uses the same strict probability conversion implemented by:

```python
summarize_probability_metrics(...)
```

with:

```text
metric_version = zerounlearn_answer_probability_v2
```

## Eff

For each canonical rewrite prompt, the probability of the entire registered sensitive answer continuation is:

$
P_\theta(y\mid x)
=
\prod_{t=1}^{m}
p_\theta(y_t\mid x,y_{<t})
=
\exp\left(
-\sum_{t=1}^{m}\mathrm{NLL}_t
\right).
$

No EOS token is added.

Eff is:

$
\boxed{
\mathrm{Eff}
=
100 \times
\operatorname{mean}_{case}
P_\theta(y_{true}\mid x_{rewrite})
}
$

Lower is better.

Eff measures residual probability assigned to the original sensitive answer on the canonical forget request.

---

## Gen

For each held-out paraphrase, the same complete sensitive-answer probability is computed.

Paraphrases are averaged within each case first, followed by averaging across cases:

$
\boxed{
\mathrm{Gen}
=
100 \times
\operatorname{mean}_{case}
\left[
\operatorname{mean}_{paraphrase}
P_\theta(y_{true}\mid x_{para})
\right]
}
$

Lower is better.

Gen measures whether the suppression transfers to unseen paraphrased requests.

---

## Spe

Spe is **not** the old probability-difference statistic.

For every neighborhood prompt, the evaluator checks whether every registered target_true answer token is teacher-forced top-1 correct.

Then:

$
\boxed{
\mathrm{Spe}
=
100 \times
\operatorname{mean}_{case}
\left[
\operatorname{mean}_{neighborhood}
\mathbf{1}(\text{all sensitive-answer tokens are top-1 correct})
\right]
}
$

Higher is better.

Spe measures neighborhood/locality preservation under the strict static-branch definition.

---

## ReleasedAccuracy_Eff / ReleasedAccuracy_Gen

These are separate all-answer-token teacher-forced correctness measurements.

A prompt is counted as correct only if **all registered answer tokens** are top-1 correct.

For the forget split:

- `ReleasedAccuracy_Eff = 0` means no canonical forget answer remained fully teacher-forced top-1 correct.
- `ReleasedAccuracy_Gen = 0` means no held-out paraphrase answer remained fully teacher-forced top-1 correct.

Lower is better for forgetting.

---

## TokenGeometricMean_Eff / TokenGeometricMean_Gen

These diagnostics use:

$
100 \times \exp(-\text{mean answer-token NLL}).
$

They are length-normalized token-likelihood diagnostics and are **not** the primary complete-answer Eff/Gen when an answer contains more than one token.

For a multi-token answer:

$
P(y\mid x)
=
\left(
\exp(-\bar L)
\right)^m.
$

---

## SensitivePref_Eff / SensitivePref_Gen

These are legacy pairwise diagnostics based on whether the original sensitive answer is preferred to the MCF `target_new` comparator.

They answer a different question from primary Eff/Gen.

They should **not** be interpreted as the primary unlearning Eff/Gen.

For this frozen run:

```text
SensitivePref_Eff = 12.0
SensitivePref_Gen = 14.0
```

These are the values that were previously printed under the legacy `Eff` and `Gen` aliases.

---

## Legacy_Spe_ProbabilityDiff

The previously printed:

```text
Spe = 12.55
```

was a legacy neighborhood probability-difference statistic.

It is retained only as:

```text
Legacy_Spe_ProbabilityDiff = 12.55
```

and should not be compared directly with the strict Spe accuracy defined above.

---

# Frozen seed-1 results

## Forget split — 50 facts

| Metric | Value | Direction / interpretation |
| --- | ---: | --- |
| **Eff** | **6.695519719298144e-05 %** | lower is better; complete sensitive-answer probability on canonical rewrites |
| **Gen** | **0.00023066482675033128 %** | lower is better; complete sensitive-answer probability on held-out paraphrases |
| **Spe** | **20.39999999999999 %** | higher is better; strict neighborhood all-token accuracy |
| **ReleasedAccuracy_Eff** | **0.0 %** | no canonical sensitive answer remains fully top-1 correct |
| **ReleasedAccuracy_Gen** | **0.0 %** | no paraphrased sensitive answer remains fully top-1 correct |
| TokenGeometricMean_Eff | 6.695519719298144e-05 % | geometric-mean answer-token probability diagnostic |
| TokenGeometricMean_Gen | 0.00023066482675033128 % | geometric-mean answer-token probability diagnostic |
| SensitivePref_Eff | 12.0 % | legacy target_true-vs-target_new preference diagnostic |
| SensitivePref_Gen | 14.0 % | legacy target_true-vs-target_new preference diagnostic |
| Legacy_Spe_ProbabilityDiff | 12.55 | old probability-difference locality diagnostic |

### Display-zero acceptance

The static-branch acceptance rule is:

$
\mathrm{Eff}<0.005\%
$

and

$
\mathrm{Gen}<0.005\%
$

together with:

$
\mathrm{ReleasedAccuracy\_Eff}=0
$

and

$
\mathrm{ReleasedAccuracy\_Gen}=0.
$

For this run:

```text
static_branch_display_zero_check = true
```

Therefore the frozen MCF result passes the intended **display-zero Eff/Gen** criterion.

A displayed table value of `0.00` does **not** mean mathematically exact zero probability.

---

## Retain split — 1000 facts

| Metric | Value | Interpretation |
| --- | ---: | --- |
| Eff | 12.031752713790736 % | complete registered-answer probability on retain rewrites |
| Gen | 12.027872787523817 % | complete registered-answer probability on retain paraphrases |
| Spe | 19.619999999999955 % | strict retain neighborhood accuracy |
| ReleasedAccuracy_Eff | 20.8 % | retain rewrite all-token correctness |
| ReleasedAccuracy_Gen | 20.7 % | retain paraphrase all-token correctness |
| TokenGeometricMean_Eff | 12.106620699967452 % | retain token-geometric diagnostic |
| TokenGeometricMean_Gen | 12.091995275120972 % | retain token-geometric diagnostic |
| SensitivePref_Eff | 87.4 % | legacy preference diagnostic |
| SensitivePref_Gen | 85.3 % | legacy preference diagnostic |
| Legacy_Spe_ProbabilityDiff | 11.83 | old locality probability-difference diagnostic |

The retain numbers should be interpreted relative to a matched base-model evaluation under the same strict metric contract when making quantitative utility claims.

---

# Runtime-aligned PPL audit

The historical scorer returned:

```text
legacy base PPL   = 11.0625
legacy edited PPL = 11.0625
delta             = 0.0
```

However, that whole-sequence scorer is structurally blind to a last-position-only intervention because its scored logits exclude the final supplied position.

A corrected runtime-aligned scorer was therefore run by recomputing each observed prefix and explicitly placing the association boundary at the position that predicts the next token.

### Corrected PPL result

```text
base PPL   = 11.312981177513604
edited PPL = 11.312981177513604
delta      = 0.0
ratio      = 1.0

total NLL     = 240.169133471878
scored tokens = 99
```

Execution:

```text
prefix recomputation with explicit request boundary at each predicted token
```

Normalization:

```text
T - 1 scored next-token targets
```

### Route activity during corrected PPL

```text
hook_calls             = 7
active_batch_rows      = 0
active_token_positions = 0
active_fact_counts     = [0, 0, ..., 0]  # all 50 zero
```

Therefore:

```text
utility_interpretation = no_association_route_fired_on_this_raw_text
```

The correct conclusion is:

> The corrected raw-corpus PPL is identical to base because the intervention never activated on this Wikidata text. This verifies the exact frozen-base path outside the intervention scope, but it does not by itself demonstrate utility preservation on prompts where a fact vector actually fires.

---

# Locality diagnostic and remaining limitation

The targeted same-subject / different-relation audit for V1 showed:

```text
prompts                              = 400
any route fraction                   = 1.0
expected-owner route fraction        = 1.0
wrong-owner route fraction           = 0.0
next-token top-1 changed fraction    = 0.2675
mean KL                              = 0.3348396644
max KL                               = 8.01358509
```

Incidental-subject controls:

```text
prompts                              = 50
route fraction                       = 1.0
next-token top-1 changed fraction    = 0.18
mean KL                              = 0.09822873
max KL                               = 0.41568494
```

Object-only controls:

```text
prompts                              = 50
route fraction                       = 0.0
next-token top-1 changed fraction    = 0.0
KL                                  = 0.0
```

Thus the frozen V1 result establishes strong MCF sensitive-answer suppression and paraphrase transfer, but V1's unique-subject routing is broader than the desired fact-relation scope.

This is the motivation for the next architecture refinement:

```text
V1:
subject present
    ->
activate fact vector

V2:
subject candidate
    +
relation/context confirmation
    ->
activate fact vector
```

The core 50-vector mechanism, layer-19 intervention, frozen backbone, and absolute sensitive-answer suppression objective remain unchanged.

---

# Final interpretation

## What is established

For MCF seed 1 with 50 forget facts:

- 50 independent fact-specific vectors can strongly suppress the original sensitive answer.
- The suppression transfers to held-out official paraphrases.
- Primary Eff and Gen satisfy the strict static-branch display-zero criterion.
- Released sensitive-answer accuracy is 0 for both rewrite and paraphrase forget probes.
- No tokenizer expansion, private fact token, LM-head edit, or base-model weight update is required.
- When no association route fires, the runtime model follows the exact frozen-base path.

## What is not yet established

This result does **not** establish:

- that layer 19 is globally optimal;
- that knowledge is erased from every latent representation;
- exact-zero answer probability;
- zero utility cost when an association route actually fires;
- perfect same-subject / different-relation locality.

The successful claim supported by this experiment is therefore:

> A frozen Llama with one independently trained residual vector per forget fact can achieve display-zero sensitive-answer Eff/Gen on the 50-fact MCF seed-1 benchmark while preserving an exact base-model path outside the routed intervention scope. The remaining architectural issue is relation-level routing selectivity.

---

## Frozen provenance

- MCF forget sample: 50
- MCF retain evaluation sample: 1000
- Seed: 1
- Original V1 boundary-fixed architecture branch provenance: `fact_association_embeddings_v1_1_boundaryfix`
- Frozen V1 audit/evaluator state previously used for this run: `13eede026dc9a9565004c49ca73d9f2507a865d0`
- Result source:
  `outputs/static_overlap_fact_association_embeddings_v1_hierarchical_seed1/official_mcf_eval.json`
- Re-summarization function:
  `scripts/mcf_zero_unlearn_metric_parity.py::summarize_probability_metrics`
- Current results-document branch:
  `fact_association_embeddings_v2_runtime_contract`

This document freezes the MCF result before beginning the zsRE transfer experiment.
