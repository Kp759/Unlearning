# SURE ICLR paper results table templates

This file is the population workbook for the confirmatory SURE paper results. Replace `TBD` only from frozen evaluator artifacts. Use `—` only when a method is unsupported or a metric is not applicable; never use `0` for missing data.

## Reporting contract

- Report `mean ± sample SD` over independent runs in every main table.
- Store 95% confidence intervals and raw numerator/denominator counts in the result registry and appendix.
- Bold the best and underline the second-best result only among methods that pass the table's preregistered utility gate.
- Do not include Base or Retrain/retain-only references when ranking unlearning methods.
- Never combine results from different model, tokenizer, dataset, evaluator, decoding, or split revisions.
- Rerun published baselines locally when the published protocol is not byte-identical to the SURE protocol.
- Mark a released/native implementation with `N`, a paper-created adapter with `A`, and an inappropriate unchanged mapping with `X`.
- Mark externally augmented methods such as SURE-U100K with `‡` and report the cache revision and document count.
- For every populated row, record the result artifact, execution commit, model hash, tokenizer hash, dataset hash, evaluator hash, seed list, hyperparameter-search budget, and checkpoint hash.

## Method support and table-placement matrix

| Dataset/protocol | ZeroUnlearn | ROME | MEMIT | Paper placement |
| --- | :---: | :---: | :---: | --- |
| MCF-ZU | N | N | N | Main factual table |
| ZsRE | N | N | N | Main factual table |
| MQuAKE released ZeroUnlearn track | N | N | N | Main factual table |
| MQuAKE-Remastered deletion adapter | A | A | A | Main only after deletion semantics are frozen |
| TOFU | A | A | A | Run all; promote to main when the shared adapter passes the firewall |
| RWKU | A | A | A | Adapted entity-level comparison using one shared atomic bundle |
| MUSE | A-Seq | X | X | ZeroUnlearn sequence extension if implemented; ROME/MEMIT negative controls only |
| WMDP | A-Domain | X | X | ZeroUnlearn domain extension if implemented; never train ROME/MEMIT on evaluation MCQs |

`N` does not mean that a published number may be copied. It means a released implementation exists for the task family. Every main-table number still requires the locked local model and evaluator.

## Table 1 — Factual unlearning on MCF-ZU and ZsRE

Primary table. All task metrics are percentage points; PPL is raw perplexity. Use the exact ZeroUnlearn target semantics and evaluator. `Eff` and `Gen` are leakage metrics, so lower is better.

| Method | Support | MCF Eff. ↓ | MCF Gen. ↓ | MCF Spe. ↑ | MCF PPL ↓ | ZsRE Eff. ↓ | ZsRE Gen. ↓ | ZsRE Spe. ↑ | ZsRE PPL ↓ |
| --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | Ref. | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| GA | N/R | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| FT/GradDiff | N/R | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| ROME | N | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| MEMIT | N | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| AlphaEdit | N/R | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| NPO | R/A | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| ZeroUnlearn | N | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| SURE-F | Ours | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| SURE-U100K‡ | Ours | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Population notes:

- Use seeds 1–10 for every non-reference method.
- Report the same raw PPL fixture for MCF and ZsRE and first reproduce the Base reference.
- Do not insert MCF-Original `FS/GFS` or `Spe-success` values here.
- ZeroUnlearn `Spe` is not the same metric or scale as MCF-Original `Spe-success`.

## Table 2 — MCF-Original deletion stress test

Separate protocol in which `target_true` is sensitive. This table must never be described as the ZeroUnlearn MCF task.

| Method | FS ↑ | GFS ↑ | Generated sensitive leakage ↓ | Spe-success ↑ | Spe-margin ↑ | Retain KL mean ↓ | Retain KL p95 ↓ | PPL/Base ≈ 1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | TBD | TBD | TBD | TBD | TBD | 0 | 0 | 1.000 |
| GA+GD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| NPO adapter | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| SURE-F | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| SURE-U1K‡ | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| SURE-U10K‡ | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| SURE-U100K‡ | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Population notes:

- `FS/GFS` are pairwise NLL-success rates under target-true-sensitive semantics.
- Generated sensitive leakage must use a frozen alias/normalization policy.
- Report KL median, p99, maximum, and counts above thresholds in the appendix.
- A PPL ratio closer to 1 is preferred; do not reward artificial PPL reduction as unlearning.

## Table 3A — MQuAKE released ZeroUnlearn track

Use this table only when the dataset revision matches the released ZeroUnlearn MQuAKE adapter.

| Method | Support | Eff. ↓ | Atomic Gen. ↓ | Multi-hop old-answer leakage ↓ | Unedited atomic accuracy ↑ | PPL/Base ≈ 1 |
| --- | :---: | ---: | ---: | ---: | ---: | ---: |
| Base | Ref. | TBD | TBD | TBD | TBD | 1.000 |
| ROME | N | TBD | TBD | TBD | TBD | TBD |
| MEMIT | N | TBD | TBD | TBD | TBD | TBD |
| AlphaEdit | N/R | TBD | TBD | TBD | TBD | TBD |
| NPO | R/A | TBD | TBD | TBD | TBD | TBD |
| ZeroUnlearn | N | TBD | TBD | TBD | TBD | TBD |
| SURE-F | Ours | TBD | TBD | TBD | TBD | TBD |
| SURE-U100K‡ | Ours | TBD | TBD | TBD | TBD | TBD |

## Table 3B — MQuAKE-Remastered deletion adapter

Populate this table only after naming the task `MQuAKE-R-Delete` and freezing the sensitive target, neutral target, train/test edit split, and old-answer tie handling. Use separate panels for 100, 1,000, 3,000, and full deletion-request scales.

| Method | Support | Train atomic old-answer leak ↓ | Held-out atomic old-answer leak ↓ | Multi-hop old-answer leak ↓ | Unedited accuracy ↑ | PPL/Base ≈ 1 |
| --- | :---: | ---: | ---: | ---: | ---: | ---: |
| Base | Ref. | TBD | TBD | TBD | TBD | 1.000 |
| ROME-R† | A | TBD | TBD | TBD | TBD | TBD |
| MEMIT-R† | A | TBD | TBD | TBD | TBD | TBD |
| AlphaEdit-R† | A | TBD | TBD | TBD | TBD | TBD |
| ZeroUnlearn-R† | A | TBD | TBD | TBD | TBD | TBD |
| SURE-F | Ours | TBD | TBD | TBD | TBD | TBD |
| SURE-U100K‡ | Ours | TBD | TBD | TBD | TBD | TBD |

Do not mix the official MQuAKE-Remastered knowledge-insertion accuracy with deletion leakage. If the official editing task is also reported, place it in a separately captioned knowledge-editing table.

## Table 4A — TOFU native results

Use one panel each for `forget01`, `forget05`, and `forget10`. `Retain composite`, `Real Authors composite`, and `World Facts composite` are within-set summaries of Probability, ROUGE-L, and Truth Ratio; preserve all nine native components in the appendix.

| Method | Support | Forget Quality KS-p ↑ | KS-D ↓ | Model Utility ↑ | Retain composite ↑ | Real Authors composite ↑ | World Facts composite ↑ |
| --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full/Target | Ref. | TBD | TBD | TBD | TBD | TBD | TBD |
| Retrain oracle | Ref. | TBD | TBD | TBD | TBD | TBD | TBD |
| GA | N/R | TBD | TBD | TBD | TBD | TBD | TBD |
| GradDiff | N/R | TBD | TBD | TBD | TBD | TBD | TBD |
| NPO | N/R | TBD | TBD | TBD | TBD | TBD | TBD |
| SimNPO | N/R | TBD | TBD | TBD | TBD | TBD | TBD |
| PDU | R | TBD | TBD | TBD | TBD | TBD | TBD |
| ZeroUnlearn-TOFU† | A | TBD | TBD | TBD | TBD | TBD | TBD |
| ROME-TOFU† | A | TBD | TBD | TBD | TBD | TBD | TBD |
| MEMIT-TOFU† | A | TBD | TBD | TBD | TBD | TBD | TBD |
| SURE-F | Ours | TBD | TBD | TBD | TBD | TBD | TBD |
| SURE-U100K‡ | Ours | TBD | TBD | TBD | TBD | TBD | TBD |

Population notes:

- Give ZeroUnlearn-TOFU, ROME-TOFU, MEMIT-TOFU, and SURE the same forget QA records and locked neutral target.
- The retain-only model is a reference, not a trainable baseline.
- Never rank on KS p-value alone; report KS-D and the underlying truth-ratio distributions.
- Promote adapted factual editors to the main table only after their adapter and data access are identical and documented.

## Table 4B — TOFU leakage robustness

| Method | Forget ROUGE-L ↓ | Exact memorization ↓ | Extraction strength ↓ | \|MIA AUC−0.5\| ↓ | Refusal rate | Degenerate generation rate ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Full/Target | TBD | TBD | TBD | TBD | TBD | TBD |
| Retrain oracle | TBD | TBD | TBD | TBD | TBD | TBD |
| GA | TBD | TBD | TBD | TBD | TBD | TBD |
| NPO | TBD | TBD | TBD | TBD | TBD | TBD |
| ZeroUnlearn-TOFU† | TBD | TBD | TBD | TBD | TBD | TBD |
| ROME-TOFU† | TBD | TBD | TBD | TBD | TBD | TBD |
| MEMIT-TOFU† | TBD | TBD | TBD | TBD | TBD | TBD |
| SURE-F | TBD | TBD | TBD | TBD | TBD | TBD |
| SURE-U100K‡ | TBD | TBD | TBD | TBD | TBD | TBD |

## Table 5A — RWKU forgetting, locality, and privacy

Run adapted factual methods on one shared target-only atomic bundle. Official forget, adversarial, neighbor, MIA, and utility probes remain evaluation-only.

| Method | Support | Forget FB ↓ | Forget QA ↓ | Forget AA ↓ | Forget All ↓ | Neighbor FB ↑ | Neighbor QA ↑ | Neighbor All ↑ | FM ↑ | RM ↓ |
| --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Before | Ref. | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| ICU | N/R | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| RepE | N/R | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| GA | N/R | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| DPO/RT | N/R | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| NPO | N/R | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| ZeroUnlearn-RWKU† | A | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| ROME-RWKU† | A | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| MEMIT-RWKU† | A | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| SURE-RWKU-F | Ours | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| SURE-RWKU-U100K‡ | Ours | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## Table 5B — RWKU utility

| Method | General ↑ | Reasoning ↑ | Truthfulness ↑ | Factuality ↑ | Fluency ↑ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Before | TBD | TBD | TBD | TBD | TBD |
| ICU | TBD | TBD | TBD | TBD | TBD |
| RepE | TBD | TBD | TBD | TBD | TBD |
| GA | TBD | TBD | TBD | TBD | TBD |
| DPO/RT | TBD | TBD | TBD | TBD | TBD |
| NPO | TBD | TBD | TBD | TBD | TBD |
| ZeroUnlearn-RWKU† | TBD | TBD | TBD | TBD | TBD |
| ROME-RWKU† | TBD | TBD | TBD | TBD | TBD |
| MEMIT-RWKU† | TBD | TBD | TBD | TBD | TBD |
| SURE-RWKU-F | TBD | TBD | TBD | TBD | TBD |
| SURE-RWKU-U100K‡ | TBD | TBD | TBD | TBD | TBD |

Population notes:

- Report individual entities and aggregate across the frozen entity set; do not claim RWKU from Stephen King alone.
- Evaluate batch sizes 1, 10, 20, and 50 in separate panels or a scalability figure.
- Publish the atomic-bundle hash and ensure all adapted methods receive identical prompts, sensitive continuations, and neutral targets.
- `RWKU-H-W1K` is a feasibility result until all official post-freeze probes are evaluated.

## Table 6A — MUSE native results

Use separate panels for News and Books. Signed PrivLeak has an ideal value of zero; it is not a simple lower-is-better metric.

| Method | Support | VerbMem-F ↓ | KnowMem-F ↓ | PrivLeak signed, 0 ideal | \|PrivLeak\| ↓ | KnowMem-R ↑ | Retain gap to retrain ↓ |
| --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Target | Ref. | TBD | TBD | TBD | TBD | TBD | TBD |
| Retrain | Ref. | TBD | TBD | 0.0 | 0.0 | TBD | 0.0 |
| GA | N/R | TBD | TBD | TBD | TBD | TBD | TBD |
| GradDiff | N/R | TBD | TBD | TBD | TBD | TBD | TBD |
| NPO | N/R | TBD | TBD | TBD | TBD | TBD | TBD |
| SimNPO | N/R | TBD | TBD | TBD | TBD | TBD | TBD |
| PDU | R | TBD | TBD | TBD | TBD | TBD | TBD |
| RMU | R | TBD | TBD | TBD | TBD | TBD | TBD |
| ZeroUnlearn-Seq† | A | TBD | TBD | TBD | TBD | TBD | TBD |
| SURE-F | Ours | TBD | TBD | TBD | TBD | TBD | TBD |
| SURE-U100K‡ | Ours | TBD | TBD | TBD | TBD | TBD | TBD |

Do not put unchanged ROME or MEMIT in the primary MUSE table. If a QA-only factual adaptation is run as a negative control, place it in the following appendix table and state that it does not consume the original passage-level request.

## Table 6B — MUSE factual-editor negative controls

| Method | Adapter input | VerbMem-F ↓ | KnowMem-F ↓ | \|PrivLeak\| ↓ | KnowMem-R ↑ | Interpretation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| ROME-QA† | Generated forget QA | TBD | TBD | TBD | TBD | Negative control; not passage-level unlearning |
| MEMIT-QA† | Generated forget QA | TBD | TBD | TBD | TBD | Negative control; not passage-level unlearning |

Use curves rather than one scalar for MUSE scalability and sustainability:

| Method | Forget-set size/request number | VerbMem-F ↓ | KnowMem-F ↓ | \|PrivLeak\| ↓ | KnowMem-R ↑ | Cumulative GPU-hours ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## Table 7A — WMDP native results

Use the official zero-shot multiple-choice evaluator. The target chance gap prevents below-chance answer suppression from being rewarded as stronger unlearning.

`Target chance gap = (|Bio accuracy − 25| + |Cyber accuracy − 25|) / 2`.

| Method | Support | WMDP-Bio ↓ | WMDP-Cyber ↓ | WMDP-Chem ↓ | Target chance gap ↓ | MMLU All ↑ | MT-Bench ↑ |
| --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | Ref. | TBD | TBD | TBD | TBD | TBD | TBD |
| LLMU | N/R | TBD | TBD | TBD | TBD | TBD | TBD |
| SCRUB | N/R | TBD | TBD | TBD | TBD | TBD | TBD |
| SSD | N/R | TBD | TBD | TBD | TBD | TBD | TBD |
| RMU | N/R | TBD | TBD | TBD | TBD | TBD | TBD |
| TAR | R | TBD | TBD | TBD | TBD | TBD | TBD |
| GA/GradDiff | R | TBD | TBD | TBD | TBD | TBD | TBD |
| NPO | R | TBD | TBD | TBD | TBD | TBD | TBD |
| SimNPO | R | TBD | TBD | TBD | TBD | TBD | TBD |
| ZeroUnlearn-Domain† | A | TBD | TBD | TBD | TBD | TBD | TBD |
| SURE-F | Ours | TBD | TBD | TBD | TBD | TBD | TBD |
| SURE-U100K‡ | Ours | TBD | TBD | TBD | TBD | TBD | TBD |

Population notes:

- Never use official WMDP evaluation MCQs to construct ROME, MEMIT, ZeroUnlearn, or SURE training requests.
- A ZeroUnlearn-Domain row is valid only if it consumes the same permitted hazardous corpus as the other corpus-level methods.
- Keep ROME/MEMIT corpus-to-triple experiments, if run, in a negative-control appendix.
- Report attacked/relearned WMDP accuracy in a robustness table; low clean accuracy alone is insufficient.

## Table 7B — WMDP locality detail and robustness

| Method | College Biology ↑ | Virology ↑ | College CS ↑ | Cybersecurity ↑ | Worst attacked WMDP ↓ | Relearned WMDP ↓ | Probe recovery ↓ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| RMU | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| TAR | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| NPO | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| SimNPO | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| ZeroUnlearn-Domain† | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| SURE-F | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| SURE-U100K‡ | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## Table 8 — Cross-benchmark constrained summary

Do not average native raw metrics. Each dataset cell should contain the native forgetting value followed by its principal utility/locality value. The utility gate must be frozen before confirmatory evaluation.

| Dataset | Primary forgetting / utility pair | Best non-SURE baseline | SURE-F | SURE-U100K‡ | Utility gate | Pareto winner |
| --- | --- | --- | --- | --- | --- | --- |
| MCF-ZU | Eff ↓ / Spe ↑ | TBD | TBD | TBD | TBD | TBD |
| ZsRE | Eff ↓ / Spe ↑ | TBD | TBD | TBD | TBD | TBD |
| MQuAKE | Multi-hop leakage ↓ / Unedited accuracy ↑ | TBD | TBD | TBD | TBD | TBD |
| TOFU | Forget Quality ↑ / Model Utility ↑ | TBD | TBD | TBD | TBD | TBD |
| RWKU | Forget All ↓ / Neighbor All ↑ | TBD | TBD | TBD | TBD | TBD |
| MUSE | KnowMem-F ↓ / KnowMem-R ↑ | TBD | TBD | TBD | TBD | TBD |
| WMDP | Target chance gap ↓ / MMLU All ↑ | TBD | TBD | TBD | TBD | TBD |

Recommended summary claims:

- Count datasets on which a method is on the forgetting–utility Pareto frontier.
- Report win/tie/loss against the strongest eligible non-SURE baseline.
- Report the number of preregistered utility gates passed.
- If average rank is shown, calculate it only over methods with complete, comparable coverage and publish the missing-data rule.
- Never form a mean of Eff, KS p-values, ROUGE, PrivLeak, and WMDP accuracy.

## Table 9 — Efficiency and reproducibility

| Method | Dataset | Trainable parameters | Edited parameters/rows | Rank | External documents | GPU-hours | Peak memory | Checkpoint size | Inference overhead | Tuning trials |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ZeroUnlearn | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| ROME | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| MEMIT | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| NPO | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| RMU | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| SURE-F | TBD | TBD | TBD | TBD | 0 | TBD | TBD | TBD | TBD | TBD |
| SURE-U100K‡ | TBD | TBD | TBD | TBD | 100,000 | TBD | TBD | TBD | TBD | TBD |

## Row-completion checklist

Before replacing a row's `TBD` values, verify:

- [ ] The method support class and adapter name are correct.
- [ ] The exact model and tokenizer revisions are recorded.
- [ ] The dataset revision, split manifest, and evaluator revision are hashed.
- [ ] Training-visible, calibration-only, and evaluation-only data roles pass the firewall.
- [ ] The checkpoint was selected without official held-out metrics.
- [ ] All scheduled seeds, entities, or independent runs completed without post hoc rejection.
- [ ] Base and any retrain/retain reference were evaluated by the same local evaluator.
- [ ] Mean, sample SD, confidence interval, and raw counts were generated from per-run artifacts.
- [ ] Utility, fluency, privacy, and collapse gates were applied exactly as preregistered.
- [ ] Compute and hyperparameter-search budgets are complete.
- [ ] The exact command, execution commit, checkpoint hash, and output paths reproduce the row.

## Metric sources

- [ZeroUnlearn: MCF, ZsRE, MQuAKE, ROME, and MEMIT adapters](https://github.com/XMUDeepLIT/ZeroUnlearn)
- [ROME factual-association interface](https://rome.baulab.info/)
- [MEMIT mass factual-editing interface](https://memit.baulab.info/)
- [TOFU metrics and task](https://locuslab.github.io/tofu/)
- [RWKU benchmark paper](https://rwku-bench.github.io/static/RWKU.pdf)
- [MUSE benchmark paper](https://arxiv.org/pdf/2407.06460)
- [MQuAKE official repository](https://github.com/princeton-nlp/MQuAKE)
- [WMDP benchmark](https://www.wmdp.ai/)
- [OpenUnlearning unified TOFU, MUSE, and WMDP framework](https://github.com/locuslab/open-unlearning)
