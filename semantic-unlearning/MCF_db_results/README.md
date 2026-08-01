# MCF Five-Fold Results

This directory stores the current five-fold MCF results for the controlled-unlearning experiments.

## Result status

- **Official Eff/Gen/SPE/PPL metrics:** complete for folds 0–4.
- **Legacy post-hoc Judge-B leakage analysis:** complete but should remain labeled as legacy auxiliary analysis.
- **Judge-B-v2 leakage analysis:** raw combined result available; human audit is incomplete and it must not be reported as final.

## 1. Official five-fold MCF results

| Fold | Eff ↓ | Gen ↓ | SPE ↑ | PPL ↓ |
|---:|---:|---:|---:|---:|
| 0 | 0.00 | 0.00 | 9.74 | 11.379462 |
| 1 | 0.00 | 0.00 | 16.11 | **11.378746** |
| 2 | 0.00 | 0.00 | **20.93** | 11.381207 |
| 3 | 0.00 | 0.00 | 18.27 | 11.379174 |
| 4 | 0.00 | 0.00 | 17.83 | 11.379182 |
| **Mean ± SD** | **0.00 ± 0.00** | **0.00 ± 0.00** | **16.58 ± 3.75** | **11.379554 ± 0.000959** |

## Interpretation

- Eff is zero for every fold, indicating zero direct forgotten-answer accuracy under the official metric.
- Gen is zero for every fold, indicating zero generalized/paraphrased forgotten-answer accuracy under the official metric.
- SPE averages 16.58 with noticeable fold variation; fold 2 is highest at 20.93 and fold 0 is lowest at 9.74.
- PPL is extremely stable across all five folds, ranging from 11.378746 to 11.381207.

## 2. Legacy post-hoc Judge-B semantic leakage analysis

Each fold contains 70 forget prompts.

| Fold | Leak → Leak | Leak → Safe | Safe → Leak | Safe → Safe | Base leak rate | Unlearned leak rate |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2 | 0 | 2 | 66 | 2.86% | 5.71% |
| 1 | 2 | 0 | 1 | 67 | 2.86% | 4.29% |
| 2 | 7 | 1 | 1 | 61 | 11.43% | 11.43% |
| 3 | 3 | 0 | 0 | 67 | 4.29% | 4.29% |
| 4 | 0 | 0 | 3 | 67 | 0.00% | 4.29% |
| **Total** | **14** | **1** | **7** | **328** | **15/350 = 4.29%** | **21/350 = 6.00%** |

Legacy aggregate interpretation:

- Successfully suppressed leaks: 1.
- Newly introduced leaks: 7.
- Net additional leaks: 6.
- Leakage increased by 1.71 percentage points in this legacy Judge-B analysis.

## 3. Judge-B-v2 raw result — pending audit

Combined raw result across 350 forget prompts:

| Metric | Base | Unlearned |
|---|---:|---:|
| Leaks | 6 / 350 | 5 / 350 |
| Leak rate | 1.71% | 1.43% |

Paired transitions:

| Transition | Count |
|---|---:|
| Leak → Leak | 4 |
| Leak → Safe | 2 |
| Safe → Leak | 1 |
| Safe → Safe | 343 |

Audit status:

- 16 paired cases were flagged for human audit.
- 11 of those cases had validation failures.
- Human-adjudicated fold 2 showed no leakage-rate change: Base 1/70 and Unlearned 1/70.
- Exact audited Judge-B-v2 per-fold results for folds 0, 1, 3, and 4 are not yet recorded here.

## Reporting rule

Use the official Eff/Gen/SPE/PPL table as the primary MCF result. Keep both Judge-B sections clearly labeled as auxiliary semantic diagnostics. Do not present Judge-B-v2 as final until all flagged cases and all five folds are fully audited.

## Files

- `official_metrics.csv`: machine-readable official five-fold MCF metrics.
- `judge_b_semantic_diagnostics.csv`: machine-readable legacy and raw-v2 transition summaries.
