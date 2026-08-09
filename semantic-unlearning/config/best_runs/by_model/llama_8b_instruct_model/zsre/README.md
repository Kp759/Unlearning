# Llama-3.1-8B-Instruct — ZsRE Best Run

## Final protocol

Setting 5e followed by BF16-safe protected active LM-head repair of the single `Unknown` output row.

- Model: `meta-llama/Llama-3.1-8B-Instruct`
- Snapshot: `0e9e39f249a16976918f6564b8830bc894c89659`
- Seeds: 1-10
- Forget records per seed: 50
- Retain records per seed: 1000
- Neutral target: `Unknown`
- Neutral token ID: 14109

### Setting 5e

- Steps: 600
- Batch size: 1
- Retain batch size: 4
- Embedding/LM-head LR: `1e-4`
- Forget weight: 2.0
- Retain weight: 1.0
- Forget margin: 1.0
- Sampling: epoch
- Post-training alphas: 0.75 / 0.50 / 0.25

### Active repair

- Maximum steps: 800
- Repair LR: 0.005
- Optimizer: AdamW
- Active logit margin: 0.25
- Selection logit margin: 0.05
- Repair rank: 0 (unrestricted protected-orthogonal repair space)
- Repair L2: `1e-6`
- Retain calibration records: 384
- Calibration seed: 1729
- Project away protected hidden directions: yes
- Stop when all active constraints are satisfied: yes
- Selected candidate scale: 1.0 for all ten seeds

### Acceptance gates

- Forget Eff <= 0.0
- Forget Gen <= 0.0
- Utility-drop tolerance: 1.20 percentage points
- Maximum PPL ratio relative to Setting 5e: 1.16
- Strict utility gates enabled

## Official 10-seed result

- Configured-gate acceptance: **10/10**
- Forget Eff: **0.0000 ± 0.0000**
- Forget Gen: **0.0000 ± 0.0000**
- Forget Spe: **28.897980 ± 2.477641**
- Retain Eff: **38.674115 ± 1.106311**
- Retain Gen: **36.701937 ± 1.045737**
- Retain Spe: **29.213883 ± 0.985178**
- PPL, all 10 seeds: **22.362500 ± 34.132984**
- Repair steps: mean **82.2**, median **57**, range **24-236**
- Active tokens: **511 -> 0**
- Protected incremental regressions: **0**
- All optimizer constraints satisfied: **10/10**

Sample standard deviations use `ddof=1`.

## Per-seed selected results

| Seed | Accepted | Eff | Gen | Spe | Retain Eff | Retain Gen | PPL | Active | Steps | Scale |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | yes | 0.0 | 0.0 | 29.844 | 38.117 | 35.244 | 11.4375 | 44->0 | 24 | 1.0 |
| 2 | yes | 0.0 | 0.0 | 33.425 | 37.548 | 36.178 | 10.7500 | 49->0 | 69 | 1.0 |
| 3 | yes | 0.0 | 0.0 | 28.328 | 39.626 | 37.674 | 11.4375 | 49->0 | 44 | 1.0 |
| 4 | yes | 0.0 | 0.0 | 31.255 | 38.815 | 37.121 | 11.6250 | 36->0 | 55 | 1.0 |
| 5 | yes | 0.0 | 0.0 | 27.979 | 37.067 | 35.605 | 11.4375 | 63->0 | 73 | 1.0 |
| 6 | yes | 0.0 | 0.0 | 25.669 | 39.570 | 37.617 | 11.6250 | 49->0 | 39 | 1.0 |
| 7 | yes | 0.0 | 0.0 | 29.884 | 40.025 | 37.663 | 11.6250 | 43->0 | 44 | 1.0 |
| 8 | yes | 0.0 | 0.0 | 29.682 | 39.216 | 36.709 | 12.3750 | 64->0 | 59 | 1.0 |
| 9 | yes | 0.0 | 0.0 | 27.721 | 37.188 | 35.313 | 11.8125 | 66->0 | 236 | 1.0 |
| 10 | yes | 0.0 | 0.0 | 25.192 | 39.569 | 37.896 | **119.5000** | 48->0 | 179 | 1.0 |

## Seed-10 PPL caveat

Seed 10 is an important absolute-utility anomaly and must be disclosed when reporting the aggregate.

| Stage | PPL |
|---|---:|
| Base Llama-3.1-8B | 10.75 |
| Setting 5e | 108.5 |
| Active-repair candidate | 119.5 |
| Selected checkpoint | 119.5 |

The repair-to-Setting-5e PPL ratio is approximately `1.1014`, so it passes the configured `1.16x` relative PPL gate. However, the absolute PPL is clearly abnormal. The extreme degradation originates primarily in the seed-10 Setting 5e checkpoint (`10.75 -> 108.5`) and is then increased further by repair (`108.5 -> 119.5`).

For descriptive context only, seeds 1-9 have PPL `11.5694 ± 0.4244`. This nine-seed statistic must not replace the official ten-seed aggregate.

## Strict seed-1 gate ablation

A stricter pilot used utility-drop tolerance `0.10 pp` and maximum PPL ratio `1.02`.

The raw seed-1 repair still achieved:

- Eff: 0.0
- Gen: 0.0
- Forget Spe: 29.843978 (unchanged)
- PPL: 11.4375 (unchanged)
- Active tokens: 44 -> 0
- Protected regressions: 0
- Optimizer convergence: 24 steps

It was rejected only because retain Eff changed from `38.277137` to `38.117137`, a `0.160 pp` drop, exceeding the stricter `0.10 pp` threshold. Under the frozen 3B-matched `1.20 pp / 1.16x` protocol, the same zero-forgetting behavior is accepted.

## Scientific interpretation

The Llama-3.1-8B ZsRE repair reliably eliminates measured forget-set rewrite/paraphrase correctness under the configured protocol: all 511 active tokens across ten seeds are reduced to zero with no protected incremental token regressions, yielding Eff=0 and Gen=0 for all ten seeds.

The result should **not** be summarized as uniformly preserving absolute language-model utility because seed 10 has severe PPL degradation already after Setting 5e. Report the zero-forgetting result and the configured-gate 10/10 acceptance together with the seed-10 PPL caveat.

## Provenance

- Raw output root: `outputs/crossmodel/llama31_8b/zsre_official_10seed_u1p20_ppl1p16_cal384`
- Slurm job: `1168768`
- Model snapshot: `0e9e39f249a16976918f6564b8830bc894c89659`
- Frozen configuration: `zsre_llama31_8b_u1p20_ppl1p16_config.json`
- Machine-readable results: `zsre_llama31_8b_u1p20_ppl1p16_10seed_summary.json`
