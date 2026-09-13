# Frozen Router V2 Architecture and Seed-1 Research Snapshot

**Status:** Router V2 frozen for confirmatory experiments  
**Branch:** `fact_association_rwku_seed1`  
**Snapshot date:** 2026-09-13  
**Base model:** Meta Llama 3.2 3B Instruct  
**Model hidden size:** 3072  
**Intervention layer:** transformer layer 19

This document records the exact architecture, routing contract, optimization
hyperparameters, seed-1 benchmark protocol, observed results, diagnostics, and
known limitations for the current fact-association residual-bank method.

The purpose of this file is to prevent silent protocol drift before
confirmatory multi-seed / multi-window experiments.

---

## 1. Method summary

The method is a **sparse, fact-addressed residual intervention** for selective
behavioral unlearning.

The pretrained language model is not modified. Instead, each unique protected
factual association owns one trainable residual vector:

[
Delta e_i in mathbb{R}^{3072}.
]

The complete residual bank is therefore

[
Delta E in mathbb{R}^{N_{mathrm{assoc}} 	imes 3072},
]

where (N_{mathrm{assoc}}) is the number of **unique protected factual
associations**, not the number of raw examples and not the number of relation
types.

At runtime, Router V2 receives only the natural prompt. If the prompt is judged
to express a protected association (i), the corresponding residual vector is
added at the original request boundary in transformer layer 19:

[
h'_{19,t} = h_{19,t} + Delta e_i.
]

If no association is selected, the residual is exactly zero and execution
follows the frozen base-model path.

### 1.1 What is frozen

All of the following remain frozen:

- tokenizer
- input embeddings
- transformer backbone
- attention / MLP parameters
- layer norms
- LM head
- all ordinary model weights

There is:

- no vocabulary extension,
- no private fact-ID token,
- no fact ID supplied at runtime,
- no target object required in the runtime input,
- no replacement-target training,
- no LM-head edit,
- no base-weight edit.

The only trainable parameters are the residual-bank rows.

### 1.2 Association identity

The conceptual protected item is

[
(	ext{subject}, 	ext{relation/context}, 	ext{sensitive object}).
]

Dataset-specific identities are:

- **MCF:** one residual per sampled protected factual association.
- **ZsRE:** one residual per direct natural request / subject / sensitive answer
  association; ZsRE does not expose the same clean symbolic relation ID as
  MQuAKE.
- **MQuAKE:** normalized
  ((	ext{subject},	ext{relation_id},	ext{target_true})).
  Exact duplicate atomic records share one residual.
- **RWKU:** normalized
  ((	ext{subject},	ext{selected natural query/context},	ext{sensitive answer})).
  Source-record identity is provenance only and is never a runtime input.

---

## 2. Residual-bank size

| Dataset | Raw protected training records | Unique residual rows | Residual-bank shape | Trainable residual parameters |
|---|---:|---:|---:|---:|
| MCF | 50 facts | 50 | (50 	imes 3072) | 153,600 |
| ZsRE | 50 facts | 50 | (50 	imes 3072) | 153,600 |
| MQuAKE | 127 atomic records | 105 | (105 	imes 3072) | 322,560 |
| RWKU | 50 selected probes | 50 | (50 	imes 3072) | 153,600 |

For MQuAKE, 127 sampled atomic records collapse to 105 unique associations;
22 duplicate records are collapsed across 14 duplicate groups.

---

## 3. Router V2

Router V2 changes only **addressing/routing**. It does not change the residual
intervention mechanism.

### 3.1 V1 vs V2

V1 used hierarchical routing:

1. find protected associations whose subject occurs in the prompt,
2. if only one protected association owns that subject, activate it directly,
3. otherwise use a frozen contextual key to disambiguate.

The unique-subject bypass could therefore activate a protected residual even
when the prompt expressed an unrelated relation.

Router V2 removes that bypass.

In V2:

1. subject occurrence is **candidate eligibility only**,
2. every candidate, including unique-subject candidates, must pass contextual
   confirmation,
3. candidates are scored using frozen positive and negative context
   prototypes,
4. the strongest qualifying candidate is selected,
5. an ambiguous top-1/top-2 decision is rejected,
6. if no candidate qualifies, routing abstains and the frozen base path is
   used.

### 3.2 Frozen context score

For association (i), let (P_i) be its positive prototype bank and (N_i)
its negative prototype bank. At layer 19, the normalized request-boundary
query is (q).

Positive similarity:

[
u_i(q)=max_{pin P_i}cos(q,p)
]

Negative similarity:

[
v_i(q)=max_{nin N_i}cos(q,n)
]

Relative context margin:

[
d_i(q)=u_i(q)-v_i(q).
]

The current Router V2 intentionally disables the absolute (u)-threshold by
setting

[
alpha_i=-1.
]

Thus, after subject eligibility, the principal confirmation rule is

[
d_i(q)ge 	au_i.
]

If multiple candidates qualify, the candidate with the highest (d_i(q)) is
selected only if the top-1/top-2 margin is at least the ambiguity margin.

### 3.3 Router V2 hyperparameters

| Parameter | Frozen value |
|---|---:|
| Routing representation layer | 19 |
| Negative controls per association | 12 |
| Absolute similarity threshold (alpha) | -1.0 (disabled) |
| Ambiguity margin | 0.02 |
| Nonseparable fixed margin slack | 0.02 |
| Separable-gap operating point | 0.10 |
| Unique-subject bypass | false |
| Subject scan scope | prompt prefix only |
| Teacher-forced suffix allowed to affect routing | false |
| Target object used by runtime router | false |

For training-separable positive/negative margins:

[
	au_i
=
d^{-}_{max}
+
0.10left(d^{+}_{min}-d^{-}_{max}ight).
]

Equivalently, the threshold stays 10% of the observed training-only separable
gap above the hardest training negative.

If the training controls are not separable:

[
	au_i = d^{+}_{min} - 0.02.
]

The 10% operating point was chosen during seed-1 development and is now
**frozen**. It must not be tuned on confirmatory seeds.

### 3.4 Positive and negative prototypes

Positive prototypes use **training-visible direct prompts only**.

Negative controls are constructed in this order:

1. direct prompts for competing protected relations/contexts with the same
   subject, when available;
2. deterministic synthetic wrong-context controls created by transplanting
   the current subject into other training-visible direct prompts.

Exact positive/negative prompt collisions are filtered.

The router is not calibrated using:

- `target_new`
- official paraphrases
- locality probes
- retain records
- neighbor records
- utility records
- MIA records
- PPL text.

### 3.5 Exact base-path contract

Residual rows are initialized to exact zero.

For an unmatched input:

[
Delta e = 0
]

and therefore

[
h'_{19,t}=h_{19,t}.
]

The implementation explicitly audits that unmatched natural prompts preserve
the base logits exactly after training. PPL evaluation also records route
activity, which is zero in the current seed-1 runs.

---

## 4. Residual optimization

### 4.1 Common parameters

| Parameter | Value |
|---|---:|
| Optimizer | Adam |
| Learning rate | 0.05 |
| Intervention layer | 19 |
| Backtracks per proposal | 12 |
| Max stalled row steps | 150 |
| Sensitive-token target probability | (10^{-6}) |
| Gradient clipping | 1.0 |
| Initialization | exact zero |
| Optimizer scope | one optimizer per residual row |
| Base parameters trainable | 0 |

The trust-radius schedule is:

| Current sensitive probability | Proposal radius |
|---:|---:|
| (p ge 10^{-3}) | 1.00 |
| (10^{-5} le p < 10^{-3}) | 0.35 |
| (10^{-6} le p < 10^{-5}) | 0.08 |
| (p < 10^{-6}) | 0.02 |

Candidate residual updates are backtracked and accepted only under the
dataset-specific safety/monotonicity conditions. Failed proposals restore the
residual row and optimizer state.

### 4.2 Dataset-specific training budgets

| Dataset | Unique rows | Row updates / row | Nominal steps | Max training time used by seed-1 launcher | Max length |
|---|---:|---:|---:|---:|---:|
| MCF | 50 | 30 nominal | 1500 | 3600 s | 512 |
| ZsRE | 50 | 30 | 1500 | 3600 s | 512 |
| MQuAKE | 105 | 30 | 3150 | 7200 s | 512 |
| RWKU | 50 | 30 | 1500 | 7200 s | 4096 |

MCF additionally uses the phase-lexicographic objective with:

- unknown completion: `" I don't know."`
- unknown weight: 1.0
- post-feasible gates: 2
- development routing preflight floor: 0.90
- MCF legacy gate slack field: 0.04.

The actual Router V2 context gate itself uses the frozen 0.02 margin slack
described above.

---

## 5. Seed-1 direct-training diagnostics

These are optimization diagnostics, not the official benchmark scores.

### 5.1 MCF

The MCF run reached the one-hour training budget and restored its best
checkpoint.

- stop reason: `wall_time_budget`
- best step: 650
- restored best checkpoint: true
- train max geometric-mean answer probability:
  (1.2651428	imes10^{-5})
- development max geometric-mean answer probability:
  (2.4350059	imes10^{-5})
- strict (10^{-6}) target was not globally feasible
- unmatched neutral logits remained exactly base after training.

Routing audit:

- train correct-row active fraction: **1.00**
- train wrong-row active fraction: **0.00**
- development correct-row active fraction: **0.99**
- development wrong-row active fraction: **0.00**
- the two development misses were abstentions, not wrong-row selections.

The strict optimization target was therefore incomplete, but official MCF
forgetting remained extremely strong.

### 5.2 ZsRE

- 50 protected associations
- 29/50 facts satisfy the strict (p<10^{-6}) constraint
- maximum sensitive-token probability:
  (5.5366644	imes10^{-4})
- sensitive top-1 tokens: **0/125**
- direct sensitive-token micro accuracy: **0%**
- every protected fact has zero direct sensitive-token accuracy.

### 5.3 MQuAKE

- 127 raw atomic records
- 105 unique protected associations
- 61/105 associations satisfy the strict (p<10^{-6}) constraint
- maximum sensitive-token probability:
  (2.2377379	imes10^{-4})
- sensitive top-1 tokens: **0/284**
- direct sensitive-token micro accuracy: **0%**
- every protected association has zero direct sensitive-token accuracy.

### 5.4 RWKU direct stage

- 50 selected training probes
- 50 unique protected associations
- 36/50 satisfy the strict (p<10^{-6}) constraint
- maximum sensitive-token probability:
  (1.0914011	imes10^{-4})
- sensitive top-1 tokens: **0/125**
- direct sensitive-token micro accuracy: **0%**
- every protected association has zero direct sensitive-token accuracy.

The V2 rollout-hardening stage has **not yet been applied** to this Router V2
checkpoint.

---

## 6. Official seed-1 benchmark results

Arrows indicate the desired direction for the metric.

### 6.1 Consolidated Base vs V1 vs Router V2

| Dataset / metric | Frozen base | Router V1 | **Router V2** |
|---|---:|---:|---:|
| **MCF Forget Eff ↓** | 12.266065 | 0.000066955 | **0.000178522** |
| MCF Forget Gen ↓ | 7.766598 | 0.000230665 | **0.000595133** |
| MCF Forget Spe | 20.400000 | 20.400000 | **20.400000** |
| MCF ReleasedAccuracy Eff ↓ | 20.0 | 0.0 | **0.0** |
| MCF ReleasedAccuracy Gen ↓ | 16.0 | 0.0 | **0.0** |
| MCF Retain Eff ↑ | 12.031753 | 12.031753 | **12.031753** |
| MCF Retain Gen ↑ | 12.028646 | 12.027873 | **12.027775** |
| MCF Retain Spe | 19.650000 | 19.620000 | **19.620000** |
| **ZsRE Forget Eff ↓** | 29.966667 | 0.0 | **0.0** |
| ZsRE Forget Gen ↓ | 27.900000 | 0.0 | **0.0** |
| ZsRE Forget Spe | 32.323179 | 32.323179 | **32.323179** |
| ZsRE Retain Eff ↑ | 31.996101 | 31.996101 | **31.996101** |
| ZsRE Retain Gen ↑ | 31.017218 | 31.017218 | **31.017218** |
| ZsRE Retain Spe | 28.271409 | 28.271409 | **28.271409** |
| **MQuAKE Forget Eff ↓** | 73.063367 | 0.0 | **0.0** |
| MQuAKE Forget AtomicGen ↓ | 50.189351 | **1.771654** | **2.952756** |
| MQuAKE official Retain Eff ↑ | 66.409350 | 57.158140 | **57.440449** |
| MQuAKE official Retain AtomicGen ↑ | 43.464235 | 36.675384 | **36.910642** |
| **RWKU same50 recovery ↓** | 60.000000 | 10.000000 | **10.000000** |
| RWKU same50 sensitive top-1 ↓ | 79.200000 | 0.0 | **0.0** |
| RWKU heldout L1 recovery ↓ | 67.441860 | **23.255814** | **25.581395** |
| RWKU heldout L2 recovery ↓ | 56.000000 | 32.000000 | **32.000000** |
| RWKU heldout paraphrase recovery ↓ | 62.000000 | 44.000000 | **44.000000** |
| RWKU Level3 recovery ↓ | 70.520230 | **56.069360** | **58.381503** |
| RWKU neighbor recovery ↑ | 69.175630 | 65.591398 | **65.591398** |
| Runtime-aligned PPL | 11.312981 | 11.312981 | **11.312981** |

The legacy PPL reported by the existing evaluator is 11.0625. The preferred
runtime-aligned PPL is 11.312981177513604.

---

## 7. Dataset-specific interpretation

### 7.1 MCF

Router V2 preserves extremely strong forgetting:

- Forget Eff: **0.000178522%**
- Forget Gen: **0.000595133%**
- ReleasedAccuracy Eff: **0**
- ReleasedAccuracy Gen: **0**

Retain Eff remains exactly equal to the frozen base:

[
12.0317527.
]

PPL is unchanged:

[
11.3129811775.
]

The strict (10^{-6}) fitting target was not reached before the MCF wall-time
budget, so the official benchmark result should be reported separately from the
training feasibility diagnostic.

### 7.2 ZsRE

Router V2 gives the cleanest locality result:

- Forget Eff: **0**
- Forget Gen: **0**
- Retain Eff: **31.996101**, exactly frozen base
- Retain Gen: **31.017218**, exactly frozen base
- PPL: **11.3129811775**, exactly frozen base.

Route activity:

| Split / probe | Route-active fraction |
|---|---:|
| Forget rewrite | 1.000 |
| Forget paraphrase | 0.968 |
| Forget neighborhood | 0.000 |
| Retain rewrite | 0.000 |
| Retain paraphrase | 0.000 |
| Retain neighborhood | 0.000 |
| PPL text | 0.000 |

This is strong evidence that the Router V2 intervention can remain narrowly
scoped while suppressing the protected answers.

### 7.3 MQuAKE official result

Official seed-1 V2:

- Forget Eff: **0**
- Forget AtomicGen: **2.952756**
- Retain Eff: **57.440449**
- Retain AtomicGen: **36.910642**
- PPL: **11.3129811775**.

Raw route activity:

| Split / probe | V1 | V2 |
|---|---:|---:|
| Forget rewrite | 1.000 | **1.000** |
| Forget AtomicGen | ~0.9894 | **0.9577** |
| Retain rewrite | 0.1789 | **0.1649** |
| Retain AtomicGen | 0.1773 | **0.1573** |

At first glance, the official retain score appears far below base. The
association-overlap diagnostic shows that this raw retain score is substantially
confounded by exact atomic facts that occur in both the sampled forget and
nominal retain instance sets.

---

## 8. MQuAKE atomic-overlap diagnosis

The MQuAKE protocol samples disjoint **multi-hop source instances**, then
flattens them into atomic factual records. Disjoint source instances do not
guarantee disjoint atomic factual associations.

Seed-1 flattening produces:

- forget atomic records: **127**
- unique protected associations: **105**
- retain atomic records: **1594**

Nominal retain atomic records decompose into:

| Retain category | Atomic records |
|---|---:|
| Exact protected association overlap | **240** |
| Same subject, different relation | **81** |
| Subject-disjoint | **1273** |
| **Association-disjoint total** | **1354** |

Thus, 240 nominal retain atomic records are exactly the same normalized
((	ext{subject},	ext{relation_id},	ext{target_true})) associations that
the method is explicitly protecting/suppressing.

### 8.1 V2 routed-retain decomposition

Across all V2 active retain token decisions:

- total active token decisions: **1221**
- exact protected association: **1038 (85.0%)**
- same-subject/different-relation: **159 (13.0%)**
- lexical subject overlap: **24 (2.0%)**

Therefore most of the nominal retain routing is **expected activation on an
exact protected association**, not a router selectivity error.

### 8.2 Association-disjoint locality

For the 1354 retain atomic facts that do **not** exactly overlap the protected
association set:

| Association-disjoint MQuAKE | Frozen base | V1 | **V2** |
|---|---:|---:|---:|
| Retain Eff ↑ | 67.732700 | 67.289569 | **67.621917** |
| Retain AtomicGen ↑ | 44.062454 | 42.585349 | **42.696132** |
| Rewrite route activity ↓ | 0.0000 | 0.046861 | **0.030628** |
| AtomicGen route activity ↓ | 0.0000 | 0.045023 | **0.025421** |

The V2 association-disjoint Retain Eff is only

[
67.732700 - 67.621917 = 0.110783
]

percentage points below frozen base.

### 8.3 Same-subject / different-relation locality

This is the hardest genuine relation-selectivity subset:

| Metric | Frozen base | V1 | **V2** |
|---|---:|---:|---:|
| Eff ↑ | 81.172840 | 73.765432 | **79.320988** |
| AtomicGen ↑ | 57.098765 | 34.259259 | **36.111111** |
| Rewrite route activity ↓ | 0.0000 | 1.0000 | **0.6940** |
| AtomicGen route activity ↓ | 0.0000 | 0.9552 | **0.4925** |

Router V2 therefore recovers most of the V1 direct-query locality loss on this
difficult subset, although held-out AtomicGen relation selectivity remains an
open limitation.

### 8.4 Subject-disjoint locality

For 1273 completely subject-disjoint retain atomic facts:

- frozen-base Eff: **66.877514**
- V1 Eff: **66.877514**
- V2 Eff: **66.877514**

V2 rewrite route activity is only **0.2236%** and AtomicGen route activity is
**0.54296%** on this stratum.

### 8.5 Reporting rule

For benchmark comparability, always report the **official overall MQuAKE retain
metrics**.

For method-locality analysis, additionally report the **association-disjoint
retain metrics** and explicitly state why the diagnostic is needed.

Do not replace the official metric with the overlap-stratified metric.

---

## 9. RWKU protocol and status

The current RWKU experiment is explicitly a

`probe_assisted_cross_benchmark_method_extension`.

RWKU natively provides target entities rather than the same type of forget
corpus used by MCF/ZsRE/MQuAKE. The current Batch-50 protocol uses:

- 5 people per deterministic batch/window,
- 10 selected direct probes per person,
- 50 selected direct training/efficacy probes,
- remaining Level-1 / Level-2 probes held out,
- deterministic paraphrases of held-out Level-2 probes,
- Level-3, neighbor, utility, MIA, and PPL data as evaluation-only.

Current Router V2 direct-stage result:

- same50 recovery: **10%**
- same50 sensitive-token top-1 accuracy: **0%**
- same50 route-correct fraction: **1.0**
- heldout L1 recovery: **25.581395%**
- heldout L2 recovery: **32%**
- heldout paraphrase recovery: **44%**
- Level3 recovery: **58.381503%**
- neighbor recovery: **65.591398%**
- neighbor route-active fraction: **0.050179**
- PPL: **11.3129811775**
- PPL route activity: zero.

The previous V1-router checkpoint plus rollout hardening reached:

- same50 recovery: **0%**
- same50 sensitive-token top-1 accuracy: **0%**
- heldout L1: **23.255814%**
- heldout L2: **32%**
- heldout paraphrase: **42%**
- Level3: **53.757225%**
- neighbor: **65.591398%**
- PPL unchanged.

This historical V1+rollout result is **not** the final Router V2 result.
Router V2 rollout hardening remains pending.

---

## 10. Router V2 preflight status

Before residual optimization, all four Router V2 seed-1 preflights passed.

### MCF

- train prompts: 646
- train correct route: **1.00**
- train wrong route: **0.00**
- development prompts: 200
- development correct route: **0.99**
- development wrong route: **0.00**
- two development failures were abstentions.

### ZsRE

- 50 direct protected prompts
- correct route: **1.00**
- wrong route: **0.00**.

### MQuAKE

- 127 atomic records
- 105 unique associations
- direct correct route: **1.00**
- wrong route: **0.00**
- 22 duplicate records collapsed
- 14 duplicate groups.

### RWKU

- 50 selected probes
- 50 unique associations
- correct route: **1.00**
- wrong route: **0.00**
- every target has 10 same-subject association candidates in the current
  Batch-50 construction.

---

## 11. Frozen implementation decision

**Router V2 is frozen. Do not introduce Router V3 before confirmatory
experiments.**

The current evidence supports keeping the architecture:

[
oxed{
	ext{Frozen LLM}
+
	ext{Context-Gated Router V2}
+
	ext{one 3072-D residual per unique association}
}
]

with a layer-19 request-boundary intervention.

The earlier concern that MQuAKE had roughly 16--18% false retain routing was
misleading because 85% of V2 active retain token decisions are exact protected
association overlaps. On the association-disjoint retain set, V2 direct Retain
Eff is within 0.111 percentage points of frozen base.

The remaining meaningful Router V2 weakness is relation generalization on
same-subject/different-relation AtomicGen prompts. This should currently be
treated as an ablation/limitation, not as justification for another
seed-specific architecture redesign.

---

## 12. Experimental status and next steps

### Completed

- MCF Router V2 seed-1 preflight
- ZsRE Router V2 seed-1 preflight
- MQuAKE Router V2 seed-1 preflight
- RWKU Router V2 seed-1 preflight
- all four Router V2 seed-1 direct trainings
- all four Router V2 seed-1 official direct evaluations
- MQuAKE retain-overlap diagnosis
- MQuAKE Base vs V1 vs V2 overlap-stratified comparison.

### Pending

1. Apply **rollout hardening** to the frozen RWKU Router V2 direct checkpoint.
2. Evaluate the hardened RWKU Router V2 checkpoint.
3. Produce the final seed-1 Base / V1 / V2(+RWKU rollout) table.
4. Freeze all remaining hyperparameters.
5. Run confirmatory seeds / deterministic windows without seed-specific
   retuning.
6. Report MCF/ZsRE/MQuAKE stochastic-seed aggregates as appropriate.
7. Report RWKU deterministic Batch-50 window aggregates separately from
   stochastic-seed claims.

Seed 1 is a **development seed** because the Router V2 threshold operating point
was selected during seed-1 development. Confirmatory runs must not use held-out
results to retune (	au), ambiguity margin, negative count, residual optimizer,
or intervention layer.

---

## 13. Reproduction entry points

### Direct training

```bash
bash scripts/train_fact_association_router_v2_seed1.sh
```

Output directories:

```text
outputs/mcf_fact_assoc_router_v2_seed1
outputs/zsre_fact_assoc_router_v2_seed1
outputs/mquake_fact_assoc_router_v2_seed1
outputs/rwku_fact_assoc_router_v2_seed1_direct
```

### Official direct evaluation

```bash
bash scripts/evaluate_fact_association_router_v2_seed1.sh
```

### MQuAKE overlap diagnosis

```bash
python scripts/diagnose_mquake_router_v2_overlap.py \
  --run-dir outputs/mquake_fact_assoc_router_v2_seed1 \
  --mquake-path data/MQuAKE-CF-3k-v2.json \
  --local-files-only
```

### MQuAKE overlap-stratified Base / V1 / V2 comparison

```bash
python scripts/compare_mquake_retain_overlap_strata_seed1.py \
  --mquake-path data/MQuAKE-CF-3k-v2.json \
  --local-files-only
```

---

## 14. Result artifact locations

### Router V2 official evaluations

```text
outputs/mcf_fact_assoc_router_v2_seed1/official_mcf_eval.json
outputs/zsre_fact_assoc_router_v2_seed1/official_zsre_eval.json
outputs/mquake_fact_assoc_router_v2_seed1/official_mquake_eval.json
outputs/rwku_fact_assoc_router_v2_seed1_direct/official_rwku_batch50_eval.json
```

### MQuAKE diagnostics

```text
outputs/mquake_fact_assoc_router_v2_seed1/router_v2_overlap_diagnostic.json
outputs/mquake_fact_assoc_router_v2_seed1/mquake_seed1_overlap_stratified_comparison.json
```

---

## 15. Claim boundary

The current evidence supports claims about **behavioral suppression / selective
retrieval-expression control** through sparse fact-addressed residual
interventions.

It does **not** establish that the protected knowledge has been deleted from
the pretrained model weights.

Preferred wording:

> We prevent retrieval/expression of selected factual associations through
> sparse, fact-addressed residual interventions while leaving the pretrained
> model unchanged outside the routed scope.

Avoid claims of universal knowledge deletion or erasure.

---

## 16. One-sentence architecture description

> Router V2 keeps a frozen Llama backbone and allocates one zero-initialized
> 3072-dimensional residual vector to each unique protected factual
> association; a training-only subject-plus-context router selects at most one
> association from natural input, and its residual is injected at the original
> request boundary of transformer layer 19, while unmatched inputs follow the
> exact frozen base path.
