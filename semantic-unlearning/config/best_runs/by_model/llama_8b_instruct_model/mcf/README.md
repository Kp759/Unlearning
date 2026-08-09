# Llama-3.1-8B-Instruct — MultiCounterFact Best Run

## Final configuration

Setting 5e followed by protected sparse LM-head active repair.

- Seeds: 0-9
- Repair rank: 8
- Maximum repair steps: 800
- Repair LR: 0.003
- Active margin: 0.25
- Retain KL mu: 0.1
- Retain calibration examples: 200
- Project away retain hidden directions: yes
- Strict post-reload requirement: Eff = 0, Gen = 0
- Minimum post-reload margin: 0.10
- Early stopping: all repair constraints satisfied

## Official 10-seed result

- Strict success: **10/10**
- Mean Eff: **0.000**
- Mean Gen: **0.000**
- Mean Spe: **16.319**
- Spe sample SD: **3.610**
- Mean PPL: **10.800**
- PPL sample SD: **0.109**
- Mean post-reload margin: **0.300**
- Minimum post-reload margin: **0.125**
- Mean repair steps: **25.2**
- Median repair steps: **19.5**
- Repair-step range: **4-49**
- Initially active prompt instances: **107**
- Active after repair: **0**
- Newly activated protected prompts: **0**

## Per-seed results

| Seed | Eff | Gen | Margin | Spe | PPL | Steps | Active | Delta norm |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0 | 0.0 | 0.2500 | 14.33 | 10.7500 | 47 | 6->0 | 0.2258 |
| 1 | 0.0 | 0.0 | 0.2500 | 16.15 | 11.0625 | 28 | 18->0 | 0.3990 |
| 2 | 0.0 | 0.0 | 0.3125 | 12.62 | 10.7500 | 16 | 11->0 | 0.2441 |
| 3 | 0.0 | 0.0 | 0.3125 | 13.96 | 10.7500 | 4 | 8->0 | 0.0700 |
| 4 | 0.0 | 0.0 | 0.1250 | 19.89 | 10.9375 | 46 | 9->0 | 0.3818 |
| 5 | 0.0 | 0.0 | 0.3750 | 22.45 | 10.7500 | 11 | 10->0 | 0.1895 |
| 6 | 0.0 | 0.0 | 0.4375 | 17.58 | 10.7500 | 15 | 10->0 | 0.2412 |
| 7 | 0.0 | 0.0 | 0.3125 | 10.94 | 10.7500 | 49 | 15->0 | 0.4373 |
| 8 | 0.0 | 0.0 | 0.1875 | 19.84 | 10.7500 | 23 | 8->0 | 0.3061 |
| 9 | 0.0 | 0.0 | 0.4375 | 15.43 | 10.7500 | 13 | 12->0 | 0.2151 |

## Provenance

- Raw output root: `outputs/crossmodel/llama31_8b/mcf_official_10seed_margin025_rank8_s800_final`
- Model snapshot: `0e9e39f249a16976918f6564b8830bc894c89659`
- Repository commit at archival time: `e7fd336c2463bc40dd1a7e288f1021b09e8e34e6`

The values above are taken from the serialized/reloaded checkpoint evaluation,
not only from in-memory optimization metrics.
