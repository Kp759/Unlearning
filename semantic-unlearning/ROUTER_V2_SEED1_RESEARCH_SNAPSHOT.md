# Frozen Router V2 Architecture and Seed-1 Research Snapshot

**Status:** Router V2 frozen for confirmatory experiments  
**Branch:** \`fact_association_rwku_seed1\`  
**Snapshot date:** 2026-09-13  
**Base model:** Meta Llama 3.2 3B Instruct  
**Hidden size:** 3072  
**Intervention layer:** transformer layer 19

This document freezes the current architecture, routing contract, optimization
hyperparameters, seed-1 protocols, observed results, diagnostics, and claim
boundaries for the fact-association residual-bank method. Its purpose is to
prevent protocol drift before confirmatory seeds / deterministic RWKU windows.

---

## 1. Architecture

The method is a **sparse, fact-addressed residual intervention** for selective
behavioral unlearning.

The pretrained language model stays frozen. Each unique protected factual
association owns one trainable residual vector

\[
\Delta e_i \in \mathbb{R}^{3072}.
\]

The residual bank is

\[
\Delta E \in \mathbb{R}^{N_{\mathrm{assoc}}\times 3072},
\]

where \(N_{\mathrm{assoc}}\) is the number of unique protected factual
associations, not the number of raw examples and not the number of relation
types.

At runtime, Router V2 receives only the natural prompt. If it selects protected
association \(i\), the corresponding residual is added at the original request
boundary \(t\) of transformer layer 19:

\[
h'_{19,t}=h_{19,t}+\Delta e_i.
\]

If the router abstains, no residual is added and the frozen base-model path is
used exactly.

### 1.1 Frozen components

All ordinary model components remain frozen:

- tokenizer,
- input embeddings,
- transformer attention / MLP blocks,
- layer norms,
- LM head,
- all pretrained weights.

The method uses:

- no vocabulary extension,
- no special/private fact-ID token,
- no external fact ID at runtime,
- no target object in the runtime routing input,
- no replacement target,
- no LM-head edit,
- no base-weight edit.

The only trainable parameters are the residual-bank rows.

### 1.2 Association identity

The conceptual protected unit is

\[
(\text{subject},\ \text{relation/context},\ \text{sensitive object}).
\]

Dataset-specific identities:

- **MCF:** one residual per sampled protected factual association.
- **ZsRE:** one residual per direct natural request / subject / sensitive-answer
  association. ZsRE does not expose the same clean symbolic relation IDs as
  MQuAKE, so the direct request supplies the relation/context.
- **MQuAKE:** normalized
  \((\text{subject},\text{relation\_id},\text{target\_true})\).
  Exact duplicate atomic records share one residual.
- **RWKU:** normalized
  \((\text{subject},\text{selected natural query/context},\text{sensitive answer})\).
  Source-record identity is provenance only and never a runtime input.

### 1.3 Residual-bank size

| Dataset | Raw protected records | Unique residual rows | Bank shape | Trainable residual parameters |
|---|---:|---:|---:|---:|
| MCF | 50 | 50 | \(50\times3072\) | 153,600 |
| ZsRE | 50 | 50 | \(50\times3072\) | 153,600 |
| MQuAKE | 127 atomic records | 105 | \(105\times3072\) | 322,560 |
| RWKU | 50 selected probes | 50 | \(50\times3072\) | 153,600 |

For MQuAKE, 127 atomic records collapse to 105 unique protected associations:
22 duplicate records are collapsed across 14 duplicate groups.

---

## 2. Router V2

Router V2 changes only **natural-input addressing/routing**. The residual-bank
architecture and layer-19 intervention are unchanged from V1.

### 2.1 V1 versus V2

V1 used hierarchical routing:

1. match a protected subject in the prompt,
2. if exactly one protected association owns that subject, activate it directly,
3. otherwise use a frozen contextual key for disambiguation.

The unique-subject bypass could therefore activate a protected residual for an
unrelated relation.

Router V2 removes that bypass.

V2 routing:

1. subject occurrence defines candidate eligibility only,
2. every candidate, including unique-subject candidates, must pass frozen
   contextual confirmation,
3. candidates are scored by positive and negative prototype banks,
4. at most one candidate is selected,
5. ambiguous top-1/top-2 decisions are rejected,
6. if no candidate qualifies, routing abstains.

### 2.2 Frozen context score

For association \(i\), positive prototypes are \(P_i\), negative prototypes are
\(N_i\), and the normalized layer-19 request-boundary query is \(q\).

\[
u_i(q)=\max_{p\in P_i}\cos(q,p)
\]

\[
v_i(q)=\max_{n\in N_i}\cos(q,n)
\]

\[
d_i(q)=u_i(q)-v_i(q)
\]

The current implementation intentionally disables the absolute-similarity
threshold with

\[
\alpha_i=-1.
\]

Thus, after subject eligibility, the principal confirmation condition is

\[
d_i(q)\ge \tau_i.
\]

Among qualifying candidates, the highest-\(d_i\) association is selected only
if the top-1/top-2 margin is at least the ambiguity margin.

### 2.3 Frozen Router V2 hyperparameters

| Parameter | Frozen value |
|---|---:|
| Routing representation layer | 19 |
| Negative controls per association | 12 |
| Absolute similarity threshold \(\alpha\) | -1.0 |
| Absolute threshold enabled | false |
| Ambiguity margin | 0.02 |
| Nonseparable fixed margin slack | 0.02 |
| Separable-gap operating point | 0.10 |
| Unique-subject bypass | false |
| Subject scan scope | prompt prefix only |
| Teacher-forced suffix affects routing | false |
| Target object used by router | false |

For training-separable positive / negative margins,

\[
\tau_i
=
d^{-}_{\max}
+
0.10\left(d^{+}_{\min}-d^{-}_{\max}\right).
\]

So \(\tau_i\) is placed 10% of the observed training-only separable gap above
the hardest negative.

If training controls are not separable,

\[
\tau_i=d^{+}_{\min}-0.02.
\]

The 10% operating point was selected during seed-1 development and is now
**frozen**. It must not be retuned on confirmatory seeds.

### 2.4 Prototype construction

Positive prototypes use **training-visible direct prompts only**.

Negative controls are constructed in this order:

1. same-subject competing protected direct contexts, when available;
2. deterministic subject-transplanted wrong-context prompts made only from
   other training-visible direct prompts.

Exact positive / negative prompt collisions are filtered.

Router fitting does **not** use:

- \`target_new\`,
- official paraphrases,
- official locality probes,
- retain records,
- neighbor records,
- utility records,
- MIA records,
- PPL text.

### 2.5 Exact base-path contract

Residual rows start at exact zero. When no association routes,

\[
\Delta e=0
\]

and therefore

\[
h'_{19,t}=h_{19,t}.
\]

The implementation audits unmatched prompts for exact base-logit preservation.
The seed-1 runtime-aligned PPL evaluations also show zero route activity.

---

## 3. Residual optimization

### 3.1 Common optimization parameters

| Parameter | Value |
|---|---:|
| Optimizer | Adam |
| Learning rate | 0.05 |
| Intervention layer | 19 |
| Backtracks | 12 |
| Max stalled row steps | 150 |
| Sensitive-token probability target | \(10^{-6}\) |
| Gradient clipping | 1.0 |
| Initialization | exact zero |
| Optimizer scope | separate optimizer per row |
| Base parameters trainable | 0 |

Trust-radius schedule:

| Current sensitive probability | Max proposal radius |
|---:|---:|
| \(p\ge10^{-3}\) | 1.00 |
| \(10^{-5}\le p<10^{-3}\) | 0.35 |
| \(10^{-6}\le p<10^{-5}\) | 0.08 |
| \(p<10^{-6}\) | 0.02 |

Failed proposals restore both the residual row and its optimizer state.

### 3.2 Dataset-specific budgets

| Dataset | Unique rows | Updates / row | Nominal steps | Seed-1 launcher wall-time cap | Max length |
|---|---:|---:|---:|---:|---:|
| MCF | 50 | 30 nominal | 1500 | 3600 s | 512 |
| ZsRE | 50 | 30 | 1500 | 3600 s | 512 |
| MQuAKE | 105 | 30 | 3150 | 7200 s | 512 |
| RWKU | 50 | 30 | 1500 | 7200 s | 4096 |

MCF additionally uses:

- phase-lexicographic objective,
- unknown completion: \`" I don't know."\`,
- unknown weight: 1.0,
- post-feasible gates: 2,
- development route-recall preflight floor: 0.90,
- legacy MCF gate-slack field: 0.04.

The Router V2 contextual gate itself uses the 0.02 margin slack recorded above.

---

## 4. Seed-1 direct-training diagnostics

These are optimization diagnostics, not official benchmark scores.

### 4.1 MCF

The run reached the one-hour wall-time budget and restored its best checkpoint.

- stop reason: \`wall_time_budget\`
- best step: 650
- restored best checkpoint: true
- train max geometric-mean answer probability:
  \(1.2651428\times10^{-5}\)
- development max geometric-mean answer probability:
  \(2.4350059\times10^{-5}\)
- strict \(10^{-6}\) criterion not globally feasible
- unmatched neutral logits remained exactly base after training.

Routing audit:

- train correct-row active fraction: **1.00**
- train wrong-row active fraction: **0.00**
- development correct-row active fraction: **0.99**
- development wrong-row active fraction: **0.00**
- both development failures were abstentions, not wrong-row routes.

### 4.2 ZsRE

- protected associations: 50
- strict \(p<10^{-6}\): **29/50**
- max sensitive-token probability:
  \(5.5366644\times10^{-4}\)
- sensitive top-1 tokens: **0/125**
- direct sensitive-token micro accuracy: **0%**
- all facts have zero direct sensitive-token accuracy.

### 4.3 MQuAKE

- raw forget atomic records: 127
- unique protected associations: 105
- strict \(p<10^{-6}\): **61/105**
- max sensitive-token probability:
  \(2.2377379\times10^{-4}\)
- sensitive top-1 tokens: **0/284**
- direct sensitive-token micro accuracy: **0%**
- all protected associations have zero direct sensitive-token accuracy.

### 4.4 RWKU direct stage

- selected training probes: 50
- unique protected associations: 50
- strict \(p<10^{-6}\): **36/50**
- max sensitive-token probability:
  \(1.0914011\times10^{-4}\)
- sensitive top-1 tokens: **0/125**
- direct sensitive-token micro accuracy: **0%**
- all protected associations have zero direct sensitive-token accuracy.

Router V2 rollout hardening has **not yet** been applied to this checkpoint.

---

## 5. Official seed-1 results

Arrows indicate desired direction.

### 5.1 Base vs Router V1 vs Router V2

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

Preferred runtime-aligned PPL:

\[
11.312981177513604.
\]

Legacy evaluator PPL:

\[
11.0625.
\]

---

## 6. Dataset-specific Router V2 results

### 6.1 MCF

Official Router V2:

- Forget Eff: **0.000178522%**
- Forget Gen: **0.000595133%**
- Forget Spe: **20.4**
- ReleasedAccuracy Eff: **0**
- ReleasedAccuracy Gen: **0**
- Retain Eff: **12.0317527**
- Retain Gen: **12.0277748**
- Retain Spe: **19.62**
- runtime-aligned PPL: **11.3129811775**.

Retain Eff is exactly frozen-base performance. The strict fitting target was not
reached before wall-time, so training feasibility and official forgetting must
be reported separately.

### 6.2 ZsRE

Official Router V2:

- Forget Eff: **0**
- Forget Gen: **0**
- Forget Spe: **32.323179**
- Retain Eff: **31.996101**
- Retain Gen: **31.017218**
- Retain Spe: **28.271409**
- runtime-aligned PPL: **11.3129811775**.

Routing:

| Probe | Route-active fraction |
|---|---:|
| Forget rewrite | **1.000** |
| Forget paraphrase | **0.968** |
| Forget neighborhood | **0.000** |
| Retain rewrite | **0.000** |
| Retain paraphrase | **0.000** |
| Retain neighborhood | **0.000** |
| PPL corpus | **0.000** |

ZsRE is currently the cleanest selective-routing result.

### 6.3 MQuAKE official

Official Router V2:

- Forget Eff: **0**
- Forget AtomicGen: **2.952756**
- Retain Eff: **57.440449**
- Retain AtomicGen: **36.910642**
- runtime-aligned PPL: **11.3129811775**.

Raw routing:

| Probe | V1 | **V2** |
|---|---:|---:|
| Forget rewrite | 1.000 | **1.000** |
| Forget AtomicGen | ~0.9894 | **0.9577** |
| Retain rewrite | 0.1789 | **0.1649** |
| Retain AtomicGen | 0.1773 | **0.1573** |

The raw official retain score is confounded by atomic association overlap,
described below.

---

## 7. MQuAKE atomic-overlap diagnosis

MQuAKE samples disjoint **multi-hop source instances** and then flattens them
into atomic records. Disjoint source instances do not imply disjoint atomic
associations.

Seed-1 flattening:

- forget atomic records: **127**
- unique protected associations: **105**
- retain atomic records: **1594**.

Nominal retain strata:

| Retain category | Atomic records |
|---|---:|
| Exact protected association overlap | **240** |
| Same subject, different relation | **81** |
| Subject-disjoint | **1273** |
| **Association-disjoint total** | **1354** |

Thus 240 nominal retain atomic records are exactly the same normalized

\[
(\text{subject},\text{relation\_id},\text{target\_true})
\]

associations that the method intentionally suppresses.

### 7.1 Routed-retain decomposition

Across Router V2 active retain token decisions:

- total: **1221**
- exact protected association: **1038 (85.0%)**
- same-subject/different-relation: **159 (13.0%)**
- lexical subject overlap: **24 (2.0%)**.

Therefore most raw retain routing is expected activation on exact protected
associations, not a locality error.

### 7.2 Association-disjoint locality

For the 1354 retain atomic records that are not exact protected associations:

| Association-disjoint MQuAKE | Frozen base | V1 | **V2** |
|---|---:|---:|---:|
| Retain Eff ↑ | 67.732700 | 67.289569 | **67.621917** |
| Retain AtomicGen ↑ | 44.062454 | 42.585349 | **42.696132** |
| Rewrite route activity ↓ | 0.000000 | 0.046861 | **0.030628** |
| AtomicGen route activity ↓ | 0.000000 | 0.045023 | **0.025421** |

The V2 direct locality gap from frozen base is only

\[
67.732700-67.621917=0.110783
\]

percentage points.

### 7.3 Same-subject / different-relation locality

This is the hardest genuine relation-selectivity subset:

| Metric | Frozen base | V1 | **V2** |
|---|---:|---:|---:|
| Eff ↑ | 81.172840 | 73.765432 | **79.320988** |
| AtomicGen ↑ | 57.098765 | 34.259259 | **36.111111** |
| Rewrite route activity ↓ | 0.0000 | 1.0000 | **0.6940** |
| AtomicGen route activity ↓ | 0.0000 | 0.9552 | **0.4925** |

V2 recovers most of the V1 direct-query locality loss here, although held-out
AtomicGen relation generalization remains a limitation.

### 7.4 Subject-disjoint locality

For 1273 subject-disjoint atomic retain records:

- frozen-base Eff: **66.877514**
- V1 Eff: **66.877514**
- V2 Eff: **66.877514**
- V2 rewrite route activity: **0.2236%**
- V2 AtomicGen route activity: **0.54296%**.

### 7.5 Reporting contract

For benchmark comparability:

- always report the **official overall MQuAKE retain metrics**.

For method-locality analysis:

- additionally report **association-disjoint retain metrics**,
- explain the exact atomic overlap,
- do not replace the official benchmark number with the stratified diagnostic.

---

## 8. RWKU protocol and status

This experiment is explicitly a

\`probe_assisted_cross_benchmark_method_extension\`.

Current Batch-50 protocol:

- 5 people per deterministic window,
- 10 selected direct probes per person,
- 50 selected direct training/efficacy probes,
- remaining Level-1 / Level-2 probes held out,
- deterministic paraphrases of held-out Level-2,
- Level-3, neighbor, utility, MIA, and PPL evaluation-only.

Current Router V2 **direct-stage** result:

- same50 recovery: **10%**
- same50 sensitive-token top-1 accuracy: **0%**
- same50 route-correct fraction: **1.0**
- heldout L1 recovery: **25.581395%**
- heldout L2 recovery: **32%**
- heldout paraphrase recovery: **44%**
- Level3 recovery: **58.381503%**
- neighbor recovery: **65.591398%**
- neighbor route-active fraction: **0.050179**
- runtime-aligned PPL: **11.3129811775**
- PPL route activity: zero.

Historical V1-router + rollout-hardening reference:

- same50 recovery: **0%**
- sensitive-token top-1 accuracy: **0%**
- heldout L1: **23.255814%**
- heldout L2: **32%**
- heldout paraphrase: **42%**
- Level3: **53.757225%**
- neighbor: **65.591398%**
- PPL unchanged.

The V1+rollout result is not the final Router V2 result. Router V2 rollout
hardening remains pending.

---

## 9. Router V2 preflight status

All four seed-1 preflights passed before expensive residual optimization.

### MCF

- train prompts: 646
- train correct route: **1.00**
- train wrong route: **0.00**
- development prompts: 200
- development correct route: **0.99**
- development wrong route: **0.00**
- two development failures were abstentions.

### ZsRE

- direct protected prompts: 50
- correct route: **1.00**
- wrong route: **0.00**.

### MQuAKE

- atomic records: 127
- unique associations: 105
- direct correct route: **1.00**
- wrong route: **0.00**
- duplicate records collapsed: 22
- duplicate groups: 14.

### RWKU

- selected probes: 50
- unique associations: 50
- correct route: **1.00**
- wrong route: **0.00**
- subject candidate-count range: 10 to 10.

---

## 10. Frozen decision

**Router V2 is frozen. Do not introduce Router V3 before confirmatory
experiments.**

Current architecture:

\[
\boxed{
\text{Frozen LLM}
+
\text{Context-Gated Router V2}
+
\text{one 3072-D residual per unique association}
}
\]

with a layer-19 request-boundary intervention.

The earlier interpretation that MQuAKE had roughly 16--18% false retain
routing was misleading: 85% of active V2 retain token decisions are exact
protected-association overlaps. On association-disjoint retain data, V2 Retain
Eff is within 0.111 percentage points of the frozen base.

The remaining meaningful limitation is relation generalization on
same-subject/different-relation AtomicGen prompts. This should currently be
treated as an ablation/limitation, not as justification for another
seed-specific router redesign.

---

## 11. Completed and pending experiments

### Completed

- MCF Router V2 seed-1 preflight
- ZsRE Router V2 seed-1 preflight
- MQuAKE Router V2 seed-1 preflight
- RWKU Router V2 seed-1 preflight
- all four Router V2 seed-1 direct trainings
- all four Router V2 seed-1 official direct evaluations
- MQuAKE atomic-overlap diagnosis
- MQuAKE Base / V1 / V2 overlap-stratified retain comparison.

### Pending

1. Apply rollout hardening to the frozen RWKU Router V2 direct checkpoint.
2. Evaluate hardened RWKU Router V2.
3. Produce the final seed-1 Base / V1 / V2(+RWKU rollout) table.
4. Freeze remaining parameters.
5. Run confirmatory seeds / deterministic RWKU windows without retuning.
6. Aggregate MCF/ZsRE/MQuAKE results appropriately across confirmatory seeds.
7. Report RWKU deterministic Batch-50 windows separately from stochastic-seed
   claims.

Seed 1 is a **development seed** because the Router V2 10% separable-gap
operating point was selected during seed-1 development. Confirmatory runs must
not retune \(\tau\), ambiguity margin, negative count, residual optimizer,
intervention layer, or trust-radius schedule from held-out results.

---

## 12. Reproduction entry points

### Direct Router V2 training

\`\`\`bash
bash scripts/train_fact_association_router_v2_seed1.sh
\`\`\`

Outputs:

\`\`\`text
outputs/mcf_fact_assoc_router_v2_seed1
outputs/zsre_fact_assoc_router_v2_seed1
outputs/mquake_fact_assoc_router_v2_seed1
outputs/rwku_fact_assoc_router_v2_seed1_direct
\`\`\`

### Official direct evaluation

\`\`\`bash
bash scripts/evaluate_fact_association_router_v2_seed1.sh
\`\`\`

### MQuAKE overlap diagnosis

\`\`\`bash
python scripts/diagnose_mquake_router_v2_overlap.py \\
  --run-dir outputs/mquake_fact_assoc_router_v2_seed1 \\
  --mquake-path data/MQuAKE-CF-3k-v2.json \\
  --local-files-only
\`\`\`

### MQuAKE overlap-stratified comparison

\`\`\`bash
python scripts/compare_mquake_retain_overlap_strata_seed1.py \\
  --mquake-path data/MQuAKE-CF-3k-v2.json \\
  --local-files-only
\`\`\`

---

## 13. Result artifacts

Official Router V2 evaluations:

\`\`\`text
outputs/mcf_fact_assoc_router_v2_seed1/official_mcf_eval.json
outputs/zsre_fact_assoc_router_v2_seed1/official_zsre_eval.json
outputs/mquake_fact_assoc_router_v2_seed1/official_mquake_eval.json
outputs/rwku_fact_assoc_router_v2_seed1_direct/official_rwku_batch50_eval.json
\`\`\`

MQuAKE diagnostics:

\`\`\`text
outputs/mquake_fact_assoc_router_v2_seed1/router_v2_overlap_diagnostic.json
outputs/mquake_fact_assoc_router_v2_seed1/mquake_seed1_overlap_stratified_comparison.json
\`\`\`

---

## 14. Claim boundary

Current evidence supports **behavioral suppression / selective
retrieval-expression control** through sparse fact-addressed residual
interventions.

It does **not** establish that the protected knowledge has been deleted from
pretrained model weights.

Preferred claim:

> We prevent retrieval/expression of selected factual associations through
> sparse, fact-addressed residual interventions while leaving the pretrained
> model unchanged outside the routed scope.

Avoid claims of universal knowledge deletion or erasure.

---

## 15. One-sentence architecture description

> Router V2 keeps a frozen Llama backbone and allocates one zero-initialized
> 3072-dimensional residual vector to each unique protected factual
> association; a training-only subject-plus-context router selects at most one
> association from natural input, and its residual is injected at the original
> request boundary of transformer layer 19, while unmatched inputs follow the
> exact frozen base path.
