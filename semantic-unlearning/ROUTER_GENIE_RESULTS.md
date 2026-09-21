# Router Genie Results: MCF and RWKU

This note records the Router V2 genie/oracle decomposition experiments used to separate routing failures from residual-actuation/coverage limits.

## Terminology

- **Recovery**: the forgotten answer is still generated after intervention. Lower recovery is better.
- **Residual**: the learned fact-association intervention vector injected into the frozen model when the router activates.
- **Residual bank**: the collection of trained residual rows.
- **Exact genie**: supplies the known gold residual row for an association that has a trained row.
- **Subject genie**: for an unseen RWKU fact, tries residual rows trained for the same person and selects a row according to the specified diagnostic criterion.

The genie is a diagnostic, not a deployable router. In particular, the generation-based subject genie uses held-out answer/recovery information to choose among existing same-person residual rows.

---

## 1. MCF Genie Decomposition

### Oracle validation

The MCF oracle behaved as intended:

- Oracle table collisions: **0**
- Oracle lookup hits: **150**
  - 50 rewrites
  - 100 paraphrases
- Oracle lookup misses: **1,500**
  - 500 neighborhood
  - 1,000 retain
- Each of the 50 protected facts activated exactly **3** times under the oracle.

### Routing result

Router V2 is essentially at the exact-genie ceiling on MCF.

- Correctly routed forget prompts: **147 / 150 = 98%**
- Rewrite routing gap: **0**
- Neighborhood false activations: **0 / 500**
- Approximate 95% Wilson upper bound for neighborhood activation: **0.76%**
- Retain gap: **0**

The entire paraphrase mean-probability gap is explained by the three missed paraphrases.

### Base vs V2 vs oracle answer probability

| Group | Base mean answer probability | Router V2 | Oracle genie |
|---|---:|---:|---:|
| Rewrite | 0.12284918 | 1.7913e-6 | 1.7913e-6 |
| Paraphrase | 0.07783218 | 6.0379e-6 | 5.0102e-6 |
| Neighborhood | 0.12922890 | 0.12922890 | 0.12922890 |
| Retain | 0.12023926 | 0.12023926 | 0.12023926 |

The paraphrase V2-to-oracle mean-probability gap is **1.0277e-6**.

### Strict 1e-6 threshold view

| Group | Base below 1e-6 | V2 below 1e-6 | Oracle below 1e-6 |
|---|---:|---:|---:|
| Rewrite | 1/50 | 24/50 | 24/50 |
| Paraphrase | 0/100 | 43/100 | 46/100 |
| Neighborhood | 9/500 | 9/500 | 9/500 |
| Retain | 5/1000 | 5/1000 | 5/1000 |

This strict threshold should not be interpreted as "no suppression" when a prompt remains above 1e-6. V2 reduced answer probability on **50/50 rewrites** and **97/100 paraphrases**, often into the 1e-6 to 1e-5 range.

### MCF interpretation

MCF does **not** show meaningful routing headroom for Router V2. The router is already nearly identical to the exact genie, and neighborhood/retain prompts are untouched because the router abstains.

The remaining MCF gap is therefore not primarily a classifier-selection problem.

---

## 2. RWKU Genie Decomposition

RWKU differs from MCF because held-out RWKU probes can ask **new facts about the same protected person** for which no dedicated residual row was trained.

Therefore two genie concepts are required:

1. **genie_exact** for the same 50 trained associations.
2. **genie_subject** for held-out facts, where existing residual rows for the same person are searched.

---

## 3. RWKU Same-50 Exact Genie

Generated-answer recovery:

| Arm | Recovery |
|---|---:|
| Base | 62% |
| Router V2 | 12% |
| Exact genie | 12% |

V2 attribution:

- Routed and suppressed: **44**
- Routed but recovered: **6**
- Not-routed recovered: **0**

### Interpretation

For trained RWKU associations, Router V2 is already at the exact-genie ceiling. All 50 trained requests were routed to the intended association row; the remaining 6 recovered answers are not explained by routing failure.

---

## 4. RWKU Held-Out: Teacher-Forced Subject Genie

The first subject-genie experiment selected the same-person residual that minimized teacher-forced sensitive-answer log-probability.

Lower recovery is better.

| Group | Base | V2 | Teacher-forced genie | Random same-person row | V2-to-genie headroom |
|---|---:|---:|---:|---:|---:|
| Held-out Level 1 | 67.44% | 20.93% | 13.95% | 39.53% | 6.98 pp |
| Held-out Level 2 | 54% | 34% | 22% | 50% | 12 pp |
| Held-out paraphrase | 62% | 42% | 36% | 56% | 6 pp |

The teacher-forced genie already showed that some existing same-person residuals transfer to unseen facts and that V2 does not always select the most suppressive available row.

However, teacher-forced probability and generated-answer recovery are not the same objective.

---

## 5. RWKU Held-Out: Generation-Based Subject Genie

The second subject-genie experiment aligned row selection with the headline RWKU recovery metric: for each held-out probe, same-person residual rows were tried under generation and a row that prevented recovery was preferred when available.

| Group | Base | V2 | Generation genie | Random same-person row | V2-to-genie headroom |
|---|---:|---:|---:|---:|---:|
| Held-out Level 2 | 54% | 34% | **10%** | 44% | **24 pp** |
| Held-out paraphrase | 62% | 42% | **22%** | 54% | **20 pp** |

### Interpretation

This materially changes the RWKU diagnosis.

For held-out Level 2:

- Base recovery: **54%**
- V2 recovery: **34%**
- Generation genie recovery: **10%**

For held-out paraphrases:

- Base recovery: **62%**
- V2 recovery: **42%**
- Generation genie recovery: **22%**

The same frozen residual bank therefore contains substantially more transferable suppression capability than Router V2 currently realizes on unseen RWKU facts.

This is **not** evidence that a deployable router can automatically achieve 10% or 22%. The generation genie uses held-out answer/recovery information and is intentionally evaluation-informed. Its role is to measure available headroom in the current bank.

---

## 6. RWKU Neighbor Locality

From the full teacher-forced decomposition run:

- Neighbor prompts: **279**
- V2 route activation rate: **13 / 279 = 4.66%**
- 95% Wilson CI: approximately **2.74% to 7.81%**
- Base neighbor recovery: **70.61%**
- V2 neighbor recovery: **68.46%**
- Locality cost in recovery: approximately **2.15 percentage points**

This contrasts with MCF, where V2 activated on **0 / 500** neighborhood prompts.

---

## 7. Main Cross-Benchmark Finding

The genie experiments reveal two different regimes.

### MCF

**V2 approximately equals the genie.**

- 147/150 forget prompts routed correctly.
- Zero neighborhood activations in 500 probes.
- Very little additional suppression is available from better association selection.

### RWKU trained associations

**V2 equals the exact genie.**

- Same-50 recovery: V2 **12%**, exact genie **12%**.
- The router correctly identifies the trained association.

### RWKU unseen facts

**V2 is substantially above the generation genie.**

- Level 2: **34% -> 10%**
- Paraphrase: **42% -> 22%**

This indicates a generalization problem in how existing residuals are selected/activated for previously unseen facts about a protected person.

The remaining nonzero genie recovery also indicates a separate residual-bank coverage/generalization limit: even the best tested single same-person residual does not suppress every unseen answer.

---

## 8. Practical Research Conclusion

The current evidence does **not** motivate redesigning the MCF router.

For RWKU, the next architectural target should be a training-safe mechanism that better predicts which existing residual, if any, will transfer to an unseen protected-subject query, while reducing the observed neighbor false activations.

The oracle-selected held-out rows must not be used as training labels and then evaluated on the same held-out examples, because that would leak evaluation information.

