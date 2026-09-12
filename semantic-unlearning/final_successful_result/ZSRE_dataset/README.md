# Final Successful Result — ZsRE Dataset

## Frozen result

This folder freezes the successful **ZsRE seed-1 / 50-forget** result for the fact-association residual bank after aligning training with the exact official ZsRE token-evaluation contexts.

- Dataset: ZsRE MEND evaluation set
- Model: Llama-3.2-3B-Instruct
- Forget sample: 50 facts
- Final retain evaluation sample: 1000 records
- Seed: 1
- Intervention layer: 19
- Trainable state: 50 independent fact-specific hidden-state residual vectors
- Transformer backbone: frozen
- Input embeddings: frozen
- LM head: frozen
- Tokenizer: unchanged
- External fact-ID/private-token injection: none
- Replacement target / `target_new="Unknown"` supervision: none
- Successful run directory: `outputs/zsre_fact_assoc_seed1_exacttoken`
- Official result: `outputs/zsre_fact_assoc_seed1_exacttoken/official_zsre_eval.json`

The final seed-1 result is:

```text
forget Eff = 0.0
forget Gen = 0.0
forget Spe = 32.323179

retain Eff = 31.996101
retain Gen = 31.017218
retain Spe = 28.271409
```

The important interpretation is:

```text
forget rewrite routing       = 100%
forget paraphrase routing    = 100%
forget neighborhood routing  = 0%

all retain routing           = 0%
raw-Wikidata PPL routing     = 0%
```

Thus the intervention activates on the forgotten direct requests and their held-out rephrases, while the evaluated locality, retain, and raw-Wikidata paths remain on the frozen base-model route.

---

# Architecture

The architecture is the same fact-specific residual-bank mechanism developed for MCF.

For 50 forget facts:

[
Delta e_1,Delta e_2,ldots,Delta e_{50}
]

are the only learned fact-specific intervention vectors.

## Runtime path

```text
Natural request
    |
    v
Frozen Llama layers 0 ... 18
    |
    v
Layer-19 hidden state
    |
    v
Fact-association router
    |
    +-- no forgotten subject match
    |       |
    |       +--> exact frozen-base path
    |
    +-- forgotten subject match
            |
            +-- unique subject
            |       |
            |       +--> activate owning fact vector
            |
            +-- duplicate subject
                    |
                    +--> frozen contextual-key selection
                            |
                            +--> activate one fact vector
    |
    v
Add selected vector only at the ORIGINAL request-boundary token

    h'_{19,T-1} = h_{19,T-1} + Delta e_i

    |
    v
Frozen remaining Llama layers
    |
    v
Frozen LM head
    |
    v
Output
```

All positions other than the configured request-boundary position remain unchanged.

When no association route activates:

[
h' = h
]

so the execution path is exactly the frozen base model.

---

# Why the original request boundary matters for ZsRE

ZsRE evaluates a multi-token answer by constructing progressively longer teacher-forced contexts.

Conceptually:

```text
request
request + answer_token_1
request + answer_token_1 + answer_token_2
...
```

The intervention must **not move forward** as answer tokens are appended.

For every sensitive answer token (y_t), the deployed contract is:

[
	ext{route} = g(x)
]

and:

[
h'_{19,T_x-1}
=
h_{19,T_x-1}+Delta e_i
]

where (T_x) is the length of the **original request**, not the growing teacher-forced context.

The evaluator verifies that the original request is an exact token prefix of every teacher-forced evaluation context.

This fixed-boundary contract prevents answer suffix tokens from changing the route or intervention location.

---

# Training-visible data

Training uses only:

```text
50 direct requested_rewrite prompts
subjects
original sensitive target_true answers
```

Training does **not** use:

```text
official rephrases                  NO
official locality/neighborhood      NO
1000 retain records                 NO
target_new="Unknown"                NO
replacement/abstention target       NO
```

This preserves held-out rephrases and locality probes for final evaluation.

---

# Exact-token training alignment

The first ZsRE implementation trained on a one-shot full-answer teacher-forced sequence.

That produced:

```text
Eff = 3.8
Gen = 7.116667
Spe = 32.323179
```

Routing was already perfect:

```text
rewrite route    = 1.0
paraphrase route = 1.0
```

so this was not a routing failure.

The mismatch was that the official ZsRE evaluator scores each sensitive answer token using a separately reconstructed teacher-forced context. BPE decode/re-encode behavior means those contexts are not guaranteed to be token-identical to a one-shot full-answer encoding.

The corrected trainer therefore optimizes the **exact same direct-token contexts used by the official evaluator** while keeping the intervention pinned to the original request boundary.

After that correction:

```text
Eff: 3.8      -> 0.0
Gen: 7.116667 -> 0.0
Spe: unchanged at 32.323179
```

This establishes that the earlier nonzero Eff/Gen were due to training/evaluation context misalignment, not a need for a stronger residual architecture.

---

# Training objective

ZsRE Eff/Gen are token-level greedy accuracies.

For each sensitive answer token:

[
p_	heta(y_tmid x,y_{<t})
]

is evaluated under the exact official teacher-forced prefix.

The training objective suppresses the highest-probability sensitive answer token for each fact until:

[
max_t
p_	heta(y_tmid x,y_{<t})
<
10^{-6}.
]

No replacement answer is optimized once the sensitive tokens satisfy the absolute suppression constraint.

This is stronger than merely making one sensitive token non-top-1 on the direct training request.

---

# Metric definitions

## Eff — Efficacy

For ZsRE, Eff is **not** complete-answer probability.

Eff is the case-macro percentage of original sensitive `target_true` answer-token decisions that remain teacher-forced top-1 correct on the canonical direct requests.

Conceptually:

[
mathrm{Eff}
=
100	imes
operatorname{mean}_{case}
operatorname{mean}_{token}
mathbf{1}
[
argmax p_	heta(cdotmid prefix_t)=y_t
].
]

**Direction:** lower is better for forgetting.

### Final value

```text
forget Eff = 0.0
```

Meaning:

> None of the evaluated original sensitive-answer token decisions remained top-1 correct on the 50 canonical forget requests.

It does **not** mean the sensitive answer has mathematically zero probability.

---

## Gen — Generalization

Gen uses the same sensitive-token greedy-accuracy definition, but on the official held-out rephrases.

[
mathrm{Gen}
=
100	imes
operatorname{mean}_{case}
operatorname{mean}_{token}
mathbf{1}
[
argmax p_	heta(cdotmid paraphrase prefix_t)=y_t
].
]

**Direction:** lower is better.

### Final value

```text
forget Gen = 0.0
```

Meaning:

> None of the original sensitive-answer token decisions remained top-1 correct on the held-out ZsRE rephrases.

Again, this is zero top-1 accuracy, **not zero answer probability**.

---

## Spe — Specificity / locality

Spe evaluates the official ZsRE locality/neighborhood targets.

[
mathrm{Spe}
=
100	imes
operatorname{mean}_{case}
operatorname{mean}_{locality token}
mathbf{1}
[
argmax p_	heta(cdotmid locality prefix)=y_{locality}
].
]

**Direction:** higher is better.

### Final value

```text
forget Spe = 32.323179
```

The important routing observation is:

```text
neighborhood token decisions = 263
active routes                = 0
route fraction               = 0.0
```

Therefore these locality probes bypassed the intervention and followed the frozen base-model path.

---

# Final routing results

## Forget rewrite requests

```text
token decisions         = 125
active token decisions  = 125
route fraction          = 1.0
```

Interpretation:

> Every evaluated direct forget token decision activated an association vector.

## Forget paraphrases

```text
token decisions         = 125
active token decisions  = 125
route fraction          = 1.0
```

Interpretation:

> Every held-out paraphrase token decision activated an association vector.

This is important because `Gen=0` cannot be attributed to a router that simply failed to expose the intervention.

## Forget neighborhood/locality

```text
token decisions         = 263
active token decisions  = 0
route fraction          = 0.0
```

Interpretation:

> All evaluated locality probes bypassed the intervention.

---

# Retain evaluation

The 1000 retain records produced:

```text
retain Eff = 31.996101
retain Gen = 31.017218
retain Spe = 28.271409
```

Route activity was exactly zero:

### Retain rewrite

```text
token decisions         = 2996
active token decisions  = 0
route fraction          = 0.0
```

### Retain paraphrase

```text
token decisions         = 2996
active token decisions  = 0
route fraction          = 0.0
```

### Retain neighborhood

```text
token decisions         = 6278
active token decisions  = 0
route fraction          = 0.0
```

Thus every evaluated retain request followed the no-route frozen-base path.

This gives a structural locality statement:

[
g(x)=0
Rightarrow
h'=h.
]

It is stronger than merely observing a small average retain-score change because the evaluator directly verifies that the residual intervention never activated on these retain probes.

---

# Perplexity

## Historical / legacy PPL

```text
legacy PPL = 11.0625
```

This whole-sequence scorer is retained only for historical reproducibility.

For a final-position-only intervention, the historical scorer can be structurally blind because the affected final supplied-position logit is excluded from the usual shifted next-token objective.

## Runtime-aligned PPL

The corrected prefix-recompute scorer gives:

```text
runtime-aligned PPL = 11.312981177513604
```

with:

```text
hook calls              = 7
active batch rows       = 0
active token positions  = 0
all 50 fact counts      = 0
```

Correct interpretation:

> On this raw Wikidata fixture, no association route fired, so the edited model exactly followed the frozen base-model path and therefore had identical runtime-aligned PPL.

This does **not** establish zero utility cost on prompts where an intervention actively fires.

---

# Final successful seed-1 result

| Metric | Value | Direction | Meaning |
| --- | ---: | --- | --- |
| Forget Eff | **0.0** | lower better | sensitive-token accuracy on direct forget requests |
| Forget Gen | **0.0** | lower better | sensitive-token accuracy on held-out rephrases |
| Forget Spe | **32.323179** | higher better | official locality/neighborhood token accuracy |
| Retain Eff | 31.996101 | reference utility | retain direct-request target accuracy |
| Retain Gen | 31.017218 | reference utility | retain rephrase target accuracy |
| Retain Spe | 28.271409 | higher better | retain locality accuracy |
| Runtime-aligned PPL | 11.312981177513604 | lower/stable better | prefix-recompute raw-text PPL |
| Legacy PPL | 11.0625 | historical only | original whole-sequence scorer |

Routing summary:

| Prompt group | Decisions | Active | Route fraction |
| --- | ---: | ---: | ---: |
| Forget rewrite | 125 | 125 | **1.0** |
| Forget paraphrase | 125 | 125 | **1.0** |
| Forget neighborhood | 263 | 0 | **0.0** |
| Retain rewrite | 2996 | 0 | **0.0** |
| Retain paraphrase | 2996 | 0 | **0.0** |
| Retain neighborhood | 6278 | 0 | **0.0** |

---

# What this establishes

For the ZsRE seed-1 development run:

1. A bank of 50 independent layer-19 residual vectors can suppress the original sensitive answers to zero official direct-token accuracy.
2. The same vectors generalize to official held-out rephrases with zero sensitive-token accuracy.
3. Forget rewrite and paraphrase routes activate for every evaluated token decision.
4. Evaluated neighborhood and all 1000 retain records trigger no route and therefore follow the exact frozen-base path.
5. No tokenizer expansion, fact-ID injection, LM-head update, embedding-table update, or base-model weight update is required.
6. The successful behavior depends on preserving the original request boundary during progressively extended teacher-forced answer contexts.

---

# What this does not establish

This result does **not** establish:

- mathematically zero sensitive-answer probability;
- irreversible knowledge deletion from all latent states;
- layer 19 as a globally optimal intervention layer;
- zero utility cost on an active-route but intentionally retained same-subject/different-relation request;
- publication-grade independence of seed 1.

The supported claim is therefore conditional factual suppression/unlearning with measured routing locality, not universal latent knowledge deletion.

---

# Seed-1 protocol status

Seed 1 is a **development / architecture-validation seed**.

The first official seed-1 ZsRE result was inspected:

```text
Eff = 3.8
Gen = 7.116667
Spe = 32.323179
```

and this exposed a direct-training/evaluator token-context mismatch. The implementation was then corrected to train on the exact official direct-token contexts, yielding the final:

```text
Eff = 0.0
Gen = 0.0
Spe = 32.323179
```

Therefore seed 1 should not be described as an untouched final test seed.

For publication-grade multi-seed claims, use a precommitted set of untouched seeds after the architecture and evaluator are frozen.

---

# Provenance

- Branch used for ZsRE implementation: `fact_association_zsre_seed1`
- Training runner: `scripts/run_zsre_fact_association_embeddings_seed1.py`
- Training launcher: `scripts/run_zsre_fact_association_embeddings_seed1.sh`
- Official evaluator: `scripts/evaluate_zsre_fact_association_embeddings_official.py`
- Evaluation launcher: `scripts/evaluate_zsre_fact_association_embeddings_official.sh`
- Successful run: `outputs/zsre_fact_assoc_seed1_exacttoken`
- Official output: `outputs/zsre_fact_assoc_seed1_exacttoken/official_zsre_eval.json`

This document freezes the successful ZsRE seed-1 architecture-validation result before moving to MQuAKE and the final multi-seed protocol.
