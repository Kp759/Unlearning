# Router Genie Results: MCF and RWKU (with MQuAKE selectivity audit)

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

The teacher-forced genie suggested that some existing same-person residuals can suppress unseen facts and that V2 does not always select the most suppressive available row. Subject-specific transfer cannot be concluded from this arm alone; see the matched cross-person control below.

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

The same frozen residual bank therefore contains more *answer-informed best-of-K suppression headroom* than Router V2 currently realizes on unseen RWKU facts. Whether that headroom is subject-conditioned rather than generic disruption requires the matched cross-person control below.

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

This demonstrates an answer-informed residual-selection headroom on previously unseen facts about a protected person. The later matched cross-person control distinguishes subject-conditioned advantage from generic disruption.

The remaining nonzero genie recovery also indicates a separate residual-bank coverage/generalization limit: even the best tested single same-person residual does not suppress every unseen answer.

---

## 8. Practical Research Conclusion

The current evidence does **not** motivate redesigning the MCF router.

For RWKU, the matched cross-person test below supports investigating a training-safe mechanism that better predicts which existing residual, if any, suppresses an unseen protected-subject query, while reducing observed neighbor false activations. It does not show that the deployable router can achieve the answer-aware genie's recovery.

The oracle-selected held-out rows must not be used as training labels and then evaluated on the same held-out examples, because that would leak evaluation information.


---

## 9. September 21, 2026 update: matched RWKU cross-person genie

This section **supersedes the interpretation of the earlier generation-genie number for held-out Level 2** where the two procedures disagree. Sections 4–5 remain as a historical record of their respective runs.

**Protocol:** 5 protected people with 10 learned residual rows per person; 50 held-out Level-2 and 50 held-out paraphrase probes. For every probe, the control ran the same frozen model, prompt formatting, fixed-boundary greedy generation, 30-token limit, and generated-answer recovery check under all **50 residual rows plus base**. Total: **5,100 generations**. The same-person arm gets all K=10 rows of the probe's person. The cross-person arm gets an equal K=10 attempts drawn without replacement from the 40 rows belonging to other people.

**What "exact" means:** given the fixed cached suppression matrix, expected best-of-K recovery for a random K-subset of a pool with M rows, R of which recover the sensitive answer, is `C(R,K)/C(M,K)`. The cross-person result averages this value over held-out probes; this is exact subset enumeration *analytically*, **not** a claim that model generation itself is numerically deterministic. Same-person best-of-10 is the observed any-suppresses result across all 10 same-person rows. The genie uses held-out answer outcomes and is **not deployable**.

| RWKU group | Base generated recovery | Router V2 | Same-person best-of-10 (matrix) | Matched cross-person best-of-10 | Cross minus same |
|---|---:|---:|---:|---:|---:|
| Held-out Level 2 (n=50) | 54% | 34% | **8.00%** (4/50) | 25.61% | +17.61 percentage points |
| Held-out paraphrase (n=50) | 62% | 42% | **22.00%** (11/50) | 39.37% | +17.37 percentage points |

**Paired probe-level bootstrap intervals for cross-person minus same-person recovery:**
- Level 2: **+7.96 to +26.96 percentage points** (point estimate +17.61).
- Paraphrase: **+5.80 to +28.26 percentage points** (point estimate +17.37).

The comparison uses the same matrix, generation settings, recovery criterion, and *selection-attempt budget* for both arms. It supports a **same-person advantage beyond equal-budget generic disruption on these five protected people and held-out probes**. Because the 50 questions in a group are clustered within just five people, the probe-level intervals do **not** independently demonstrate generalization across new people; evaluate subject-level uncertainty or additional subjects before making broad claims.

### Generic disruption is also present

| Group | Fraction of same-person residual rows that prevent recovery | Fraction of other-person residual rows that prevent recovery |
|---|---:|---:|
| Level 2 | 50.00% | 46.55% |
| Paraphrase | 44.00% | 40.65% |

These per-row rates include probes on which the **base model already failed** to recover the answer (base recovery 54% and 62%). Thus, they are not conditional causal suppression rates. A useful next analysis is suppression **restricted to base-recovered probes**, with paired per-person accounting and neighbor/retain utility checks.

### Why the original Level-2 genie says 10%, not 8%

The earlier `decomposition_generation/rwku_router_decomposition_rows.json` records Level-2 genie recovery **5/50 = 10%**; the new matrix's answer-informed best-of-10 is **4/50 = 8%**. Paraphrase agrees at **11/50 = 22%**.

The original decomposition **regenerates** once under the chosen row *after* trying candidate rows. The matrix analyzes the cached candidate generations directly. An offline comparison of `rwku_genie_subject_candidates.json`, original final rows, and `suppression_matrix.jsonl` identified:

- **One final-recovery disagreement**, Level-2 probe index **48**: “In which year did Cindy Crawford quit full-time modeling?” (protected answer **2000**).
- Original chosen row **45** did **not** recover the answer in its original candidate-selection pass, but **did** recover it in the original final pass: “Cindy Crawford quit full-time modeling in 2000.” The new matrix also shows recovery under row 45, yet has another same-person row that suppresses the answer.
- **Five Level-2 probes had at least one changed candidate-row recovery outcome** across the original selection pass versus the newer matrix (indices **13, 31, 45, 46, 48**). The paraphrase group had zero candidate or final disagreements.

Do **not** silently interchange 8% and 10% in a paper table. The matched comparison is **8.00% versus 25.61% within the new matrix**; the old 10% is a separately observed final-generation outcome. Same-model greedy generation can amplify numerical differences; the precise source of the changed candidate outcomes (precision, hardware, statefulness, etc.) **has not been established**.

Outputs:
- `outputs/rwku_fact_assoc_router_v2_seed1_direct/cross_person_control/cross_person_control.json`
- `outputs/rwku_fact_assoc_router_v2_seed1_direct/cross_person_control/suppression_matrix.jsonl`
- `outputs/rwku_fact_assoc_router_v2_seed1_direct/decomposition_generation/rwku_router_decomposition_rows.json`
- `outputs/rwku_fact_assoc_router_v2_seed1_direct/decomposition_generation/rwku_genie_subject_candidates.json`

Implementation: `scripts/evaluate_rwku_cross_person_control.py`.

---

## 10. September 21, 2026 update: MCF full training/evaluation parity audit

This complements the **exact** MCF genie, which routes all 50 canonical rewrites to their intended rows. The previous strict result remains **24/50 rewrites below epsilon = 1e-6, 26/50 at or above epsilon**.

| Check | Measured result |
|---|---|
| Selected training checkpoint | Best step **650** |
| Recorded stop reason | `wall_time_budget` |
| Globally feasible at selected step | **False** |
| Facts satisfying trainer's **full per-fact criterion** | **0/50** |
| Canonical training/evaluation full-token sequence identical | **50/50** |
| Intervention boundary identical | **50/50** |
| Answer-token IDs identical | **50/50** |
| Full-audit classifications | **24 `passes_eval`; 26 `training_reported_failing`** |

Training minimizes the geometric-mean answer-token probability across its required views; the evaluation's full-answer probability is its product over tokens. With the *same* tokens, residual state, dtype, and context, the full-answer probability cannot exceed that geometric mean. Metric aggregation alone therefore cannot turn a genuinely passing training view into a failing complete-answer evaluation. The canonical tokenization and intervention-boundary mismatches were ruled out on this run.

**Supported interpretation:** the saved checkpoint was **not globally feasible when training stopped at the wall-time budget**. The 26 canonical failures are already within the trainer's reported failing facts; choosing a better router cannot rescue the 50 rewrites because the exact genie uses the same rows. The report's **0/50** full-training pass count does **not** mean 0/50 canonical complete-answer probabilities pass: the canonical-only result is 24/50.

**Classification-order caveat:** `training_reported_failing` is assigned before other failure labels for any row the full trainer reported as failing. Consequently, the 26 labels do **not independently eliminate** additional saved-state or FP32/BF16 effects; inspect per-row `scores.float32` and `scores.bfloat16` to quantify any precision contribution. Do not claim that precision was excluded merely because no row is labeled `precision`.

Output: `outputs/mcf_fact_assoc_router_v2_seed1/parity_audit/mcf_train_eval_parity.json`.
Implementation: `scripts/audit_mcf_train_eval_parity.py`.

---

## 11. MQuAKE same-subject selectivity audit (not a genie)

**MQuAKE has no exact/subject-genie comparison in this note.** Its completed Router V2 retain audit provides a separate, necessary constraint on interpreting any future genie-assisted routing improvements.

The locked seed-1 retain set has **1,594 atomic records**: **240 exact protected-association overlaps**, **81 same-subject/different-relation records**, and **1,273 subject-disjoint records**.

| Retain stratum | V2 rewrite activation | V2 AtomicGen activation |
|---|---:|---:|
| Exact protected association | 240/240 (100%) | 237/240 (98.75%) |
| **Same subject, different relation** | **63/81 (77.78%)** | **47/81 (58.02%)** |
| Subject-disjoint metadata | 4/1,273 (0.31%) | 10/1,273 (0.79%) |

Deduplicated activated case/prompt-type/selected-row relationship counts: **477 exact protected association**, **110 same-subject/different-relation**, **14 lexical subject overlap**. The 14 lexical-overlap activations account for the 4 rewrite plus 10 AtomicGen subject-disjoint-record activations. Note that “subject-disjoint” is by record metadata; the request itself can contain overlapping subject token surfaces.

**Scope of inference:** activation means an intervention was made, **not** that the retained answer became incorrect. To measure actual collateral damage, compare base versus V2 predictions/accuracy on the 81 same-subject/different-relation records. The 240 exact-overlap retain records are not suitable evidence of unprotected same-subject locality because they are also protected forgetting targets. Do not call this audit a MQuAKE genie result.

Output: `outputs/mquake_fact_assoc_router_v2_seed1/router_v2_overlap_diagnostic.json`.
Implementation: `scripts/diagnose_mquake_router_v2_overlap.py`.

---

## 12. Current interpretation and immediate checks

1. **Trained MCF/RWKU associations:** exact genie ≈ V2; routing is not the explanation for the remaining trained-association recovery/strict-threshold failures.
2. **Unseen RWKU associations:** an answer-informed same-person best-of-10 outperforms an equally answer-informed cross-person best-of-10 on the current five-person sample. This supports subject-conditioned suppression *in addition to* generic perturbation, but does not establish that a deployable selector can achieve genie performance.
3. **MCF actuation:** the selected checkpoint stopped before satisfying its full training criterion; address convergence separately from router changes.
4. **Association-specific locality:** MQuAKE's same-subject retain activations (63/81 rewrites; 47/81 AtomicGen) argue for harder training-safe same-subject/different-relation negatives and a direct base-to-V2 retain-damage analysis.
5. **Before a paper headline:** resolve the Level-2 candidate-generation reproducibility issue; calculate base-recovered-conditional suppression and subject-level uncertainty; inspect MCF FP32/BF16 per-row scores. Keep genie-selected rows out of held-out router training labels.
