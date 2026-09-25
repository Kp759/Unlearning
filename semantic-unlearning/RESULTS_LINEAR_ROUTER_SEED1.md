# Learned Linear Router — Seed-1 Results

This file freezes the current **Seed-1 exploratory/development evidence** for the
learned fact-association router on MCF, ZsRE, MQuAKE, and RWKU.

The scorer comparison is:

- **cosine**: Router V2 prototype-margin score;
- **linear**: one independent binary linear classifier per protected association,
  vectorized as a single `Linear(d, N)` scorer;
- **global**: one held-out-calibrated threshold shared by all association heads;
- **per-association**: one held-out-calibrated threshold per association/head.

For MCF, ZsRE, and MQuAKE, lower audit false activation is better while correct
routing of protected calibration/audit positives should remain high. For RWKU,
the benchmark is entity-level: the default **subject gate** intentionally admits
same-subject prompts, so its conventional false-activation number is not directly
comparable to a threshold gate.

All results below use seed 1. The v2.4 threshold-cap patch is commit
`ed4286faf3c1e2d1e9d9b2a161e062ddfec51a7b`.

## 1. Audit 2x2 summary

### MCF

Source: `outputs/mcf_linear_2x2_seed1_v24`

| Scorer | Threshold policy | Audit correct route | Audit false activation |
| --- | --- | ---: | ---: |
| Linear | Global | 100% | 51.65% |
| Linear | Per-association | 100% | **35.16%** |
| Cosine | Global | 100% | 84.07% |
| Cosine | Per-association | 96% | 56.59% |
| Shipped V2 | In-sample per-association tau | 100% | 90.11% |

Selected linear fit: L2=`1e-5`, PCA=0, global threshold
`-3.780111789703369`.

### ZsRE

Source: `outputs/zsre_linear_2x2_seed1`

| Scorer | Threshold policy | Audit correct route | Audit false activation |
| --- | --- | ---: | ---: |
| Linear | Global | 100% | **2.57%** |
| Linear | Per-association | 100% | 10.00% |
| Cosine | Global | 100% | 24.86% |
| Cosine | Per-association | 100% | 9.43% |
| Shipped V2 | In-sample per-association tau | 100% | 32.00% |

Selected linear fit: L2=`1e-6`, PCA=0.

**Note:** these ZsRE per-association official results were obtained before the
v2.4 training-positive threshold cap. The corrected per-association arms should
be rerun before using them as final confirmatory numbers.

### MQuAKE

Source: `outputs/mquake_linear_2x2_seed1_v24`

| Scorer | Threshold policy | Audit correct route | Audit false activation |
| --- | --- | ---: | ---: |
| Linear | Global | 100% | 5.80% |
| Linear | Per-association | 100% | 6.35% |
| Cosine | Global | 100% | **5.11%** |
| Cosine | Per-association | 100% | 8.70% |
| Shipped V2 | In-sample per-association tau | 100% | 20.30% |

Selected linear fit: L2=`1e-5`, PCA=0, global threshold
`-2.8640079498291016`.

### RWKU

Threshold-gate fit source: `outputs/rwku_linear_global_seed1_v24`

| Scorer | Threshold policy | Audit correct route | Audit false activation |
| --- | --- | ---: | ---: |
| Linear | Global | 100% | **2.86%** |
| Linear | Per-association | 100% | 34.29% |
| Cosine | Global | 100% | 45.71% |
| Cosine | Per-association | 100% | 45.71% |
| Shipped V2 | In-sample per-association tau | 100% | 77.14% |

The separate RWKU subject-gate fit
(`outputs/rwku_linear_subject_seed1_v24`) has audit correct route 100%, but
reports conventional audit false activation as 100% because every prompt naming
a protected subject is intentionally admitted. That value should **not** be read
as ordinary threshold-gate FPR.

## 2. Official MCF results

| Arm | Forget Eff ↓ | Forget Gen ↓ | Retain Eff ↑ | Retain Gen ↑ | Display-zero check |
| --- | ---: | ---: | ---: | ---: | --- |
| Cosine + global | 0.0001785 | **0.0005786** | 12.0318 | 12.0278 | pass |
| Cosine + per-association | 0.0001785 | 0.0263776 | 12.0318 | 12.0278 | fail |
| **Linear + global** | 0.0001785 | **0.0029328** | 12.0318 | 12.0278 | **pass** |
| Linear + per-association | 0.0001785 | 0.0728995 | 12.0318 | 12.0286 | fail |

All four arms have runtime-aligned PPL `11.312981` (legacy `11.0625`).

Interpretation: per-association thresholds reduce audit false activation further,
but over-abstain on protected paraphrases. Linear + global gives the best
selectivity/forgetting balance among the linear policies. Cosine + global gives
the lowest raw Forget Gen but has much worse audit false activation.

## 3. Official ZsRE results

| Arm | Forget Eff ↓ | Forget Gen ↓ | Forget paraphrase routed | Audit false activation |
| --- | ---: | ---: | ---: | ---: |
| **Linear + global** | 0 | 7.83 | 83/125 = 66.4% | **2.57%** |
| Linear + per-association | 0 | 3.83 | 105/125 = 84.0% | 10.00% |
| Cosine + global | 0 | **1.67** | 115/125 = 92.0% | 24.86% |
| Cosine + per-association | 0 | 2.33 | 104/125 = 83.2% | 9.43% |
| Shipped V2 | 0 | **0** | 121/125 = 96.8% | 32.00% |

All learned/cosine 2x2 arms route 125/125 canonical rewrites and have zero
neighborhood routing in the recorded Seed-1 evaluation. The retain routing set
also stays inactive.

Interpretation: ZsRE exposes a specificity-vs-paraphrase-recall tradeoff. Linear
+ global is by far the most selective, while per-association calibration recovers
protected paraphrase routing at a higher false-activation rate. The cosine scorer
still has stronger official paraphrase coverage on this benchmark.

**Status:** corrected v2.4 per-association arms still need to be rerun before
these per-association numbers are considered final.

## 4. Official MQuAKE results

| Arm | Forget AtomicGen ↓ | AtomicGen routed ↑ | Retain AtomicGen ↑ | Same-subject / different-relation fires (rewrite / AtomicGen, of 81) |
| --- | ---: | ---: | ---: | ---: |
| Shipped V2 | 2.95 | 95.8% | 36.91 | 63 / 47 |
| Cosine + global | 3.5827 | 88.73% | 36.8636 | 31 / 36 |
| Cosine + per-association (pre-v2.4 official) | 6.2992 | 88.38% | 37.2086 | 23 / 29 |
| **Linear + global** | **2.1654** | **96.48%** | 37.9301 | 14 / 14 |
| Linear + per-association (v2.4) | 3.7402 | 92.61% | **38.0242** | **13 / 12** |

The v2.4 cap fixed the prior linear per-association evaluator failure: direct
rewrite routing is now 284/284. However, protected AtomicGen routing falls from
274/284 under linear + global to 263/284 under linear + per-association.

Interpretation: **linear + global is the strongest valid MQuAKE operating point**.
Per-association thresholds make a small additional gain in same-subject relation
selectivity but lose too much protected AtomicGen coverage.

**Status:** the cosine + per-association official number above is the pre-v2.4
run. Its corrected v2.4 official rerun is still pending.

## 5. Official RWKU results

RWKU differs semantically from MCF/ZsRE/MQuAKE: it evaluates generalization to
unseen questions about the same protected entity. Lower recovery is better on
protected RWKU probes.

| Metric | Linear + subject gate | Linear + global threshold | Preferred direction |
| --- | ---: | ---: | --- |
| Same-50 recovery | 10.0 | 10.0 | lower |
| Same-50 correct-row fraction | 100% | 100% | higher |
| Held-out Level-1 recovery | **23.26** | 51.16 | lower |
| Held-out Level-2 recovery | **34.0** | 50.0 | lower |
| Held-out paraphrase recovery | **32.0** | 48.0 | lower |
| Level-3 recovery | **54.34** | 68.79 | lower |
| Neighbor recovery | 66.67 | **68.82** | higher |
| Neighbor route-active fraction | 5.38% | **0%** | lower |

Interpretation: the global threshold is conventionally selective, but it
over-abstains on unseen questions about protected people. The subject gate gives
substantially stronger held-out forgetting with a small locality cost.

**Status:** RWKU per-association thresholding has an audit 2x2 result above, but
has **not** been run through the full official RWKU evaluator. The benchmark's
principled main policy remains the subject gate because its protection scope is
entity-level rather than one isolated fact.

## 6. Seed-1 picture to freeze before confirmatory seeds

| Benchmark | Scorer | Main gate / threshold interpretation |
| --- | --- | --- |
| MCF | Linear classifier | Global threshold |
| MQuAKE | Linear classifier | Global threshold |
| ZsRE | Linear classifier | Global threshold for the clean association-level main method; per-association remains an ablation because it trades specificity for paraphrase recall |
| RWKU | Linear classifier | Subject gate |

The intended paper framing is therefore **one common learned linear scorer with
a gate determined by the protection scope**, not a post-hoc winner chosen from
official test metrics:

- fact/association-level protection (MCF, ZsRE, MQuAKE): thresholded routing;
- entity-level protection (RWKU): subject-conditioned routing.

Seed 1 should be treated as exploratory/development evidence. Gate policy,
threshold calibration rule, ambiguity margin, and hyperparameter grids should be
frozen before confirmatory multi-seed evaluation.
