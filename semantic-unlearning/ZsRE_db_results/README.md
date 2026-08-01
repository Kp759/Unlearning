# ZsRE Five-Fold Results

This directory stores the current five-fold ZsRE results for the controlled-unlearning experiments.

## Result status

- **Official final-application metrics:** complete for folds 0–4.
- **Locked-test LLM1 diagnostics:** complete for folds 0–4.
- **Original free-generation collapse diagnostics:** complete for folds 0–4.
- **Stop-on-Unknown decoding analysis:** post-hoc and incomplete across all five folds; do not mix it with the primary results.

## 1. Official final-application metrics

| Fold | Forget Eff ↓ | Forget Gen ↓ | Forget SPE ↑ | Retain Eff ↑ | Retain Gen ↑ | Retain SPE ↑ |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.000000 | 0.000000 | 20.726190 | 38.157143 | 39.546429 | 30.371760 |
| 1 | 0.000000 | 0.000000 | 30.357143 | 32.698214 | 35.107024 | 30.289440 |
| 2 | 0.000000 | 0.000000 | 33.839286 | 32.395833 | 33.815833 | 32.428522 |
| 3 | 0.000000 | 0.000000 | 24.571429 | 32.750549 | 35.991136 | 24.560380 |
| 4 | 0.000000 | 0.000000 | 31.500000 | 32.107143 | 36.172619 | 31.053399 |
| **Mean** | **0.000000** | **0.000000** | **28.198810** | **33.621776** | **36.126608** | **29.740700** |

## 2. Locked-test LLM1 diagnostics

| Fold | Forget Eff ↓ | Forget Gen ↓ | Lexical leak ↓ | Strict record no-leak ↑ | Sensitive probability ratio ↓ | Retain probability ratio ↑ | Locality probability ratio ↑ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0000 | 0.0000 | 0.1429 | 0.1000 | 0.4058 | 1.0214 | 0.9938 |
| 1 | 0.0000 | 0.0000 | 0.0000 | 0.4000 | 0.0742 | 0.9814 | 0.9909 |
| 2 | 0.0000 | 0.0000 | 0.0714 | 0.3000 | 0.0828 | 0.9928 | 0.9603 |
| 3 | 0.0000 | 0.0000 | 0.0429 | 0.3000 | 0.0217 | 0.9197 | 0.8757 |
| 4 | 0.0000 | 0.0000 | 0.1000 | 0.3000 | 0.1803 | 1.0020 | 1.0003 |
| **Mean** | **0.0000** | **0.0000** | **0.071429** | **0.280000** | **0.152960** | **0.983455** | **0.964198** |
| **SD** | **0.0000** | **0.0000** | **0.054398** | **0.109545** | **0.152506** | **0.038557** | **0.051788** |

Additional aggregate diagnostics:

| Metric | Mean ± SD |
|---|---:|
| Sensitive probability reduction ↑ | 0.847040 ± 0.152506 |
| Sensitive preference rate ↓ | 0.420000 ± 0.108138 |

## 3. Original free-generation Unknown-loop diagnostics

| Fold | Unknown-loop prompts | Total prompts | Loop rate ↓ |
|---:|---:|---:|---:|
| 0 | 285 | 2751 | 10.36% |
| 1 | 1240 | 2751 | 45.07% |
| 2 | 1138 | 2751 | 41.37% |
| 3 | 152 | 2751 | 5.53% |
| 4 | 144 | 2751 | 5.23% |
| **Total** | **2959** | **13755** | **21.51%** |

### By expected behavior

| Behavior | Count | Base loop rate | Unlearned loop rate | Newly introduced loop rate |
|---|---:|---:|---:|---:|
| avoid_sensitive | 350 | 2.86% | 39.43% | 37.14% |
| answer_correctly | 7000 | 2.36% | 23.61% | 21.86% |
| preserve_locality | 6405 | 0.98% | 18.24% | 17.53% |

## Interpretation

- Exact-token efficacy and generalization are zero across all five folds.
- Semantic forgetting is incomplete: strict record no-leak averages 28% and sensitive preference averages 42%.
- Retain and locality answer-probability ratios remain relatively high.
- Free generation reveals a major `Unknown` repetition failure that is not captured by Eff/Gen or teacher-forced answer-probability metrics.
- The stop-on-Unknown patch is an inference-time safeguard only. It improves termination/readability but does not change the checkpoint-level metrics or remove unwanted abstention on retain/locality prompts.

## Files

- `official_and_llm1_metrics.csv`: machine-readable official and LLM1 five-fold metrics.
- `generation_collapse_metrics.csv`: machine-readable Unknown-loop diagnostics.
