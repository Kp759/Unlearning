# ZsRE Setting 5e + active LM-head repair — seeds 1–10

## Raw candidate result

The full repaired candidate checkpoint for each seed was evaluated before strict-gate rollback. Eff and Gen reached zero on all ten seeds.

| Seed | Strict gate | Scale | Eff ↓ | Gen ↓ | Spe ↑ | PPL ↓ | Failed gates |
|---:|:---:|---:|---:|---:|---:|---:|:---|
| 1 | FAIL | 1.000000 | 0.000000 | 0.000000 | 30.614846 | 11.6250 | retain_Eff,retain_Gen |
| 2 | FAIL | 1.000000 | 0.000000 | 0.000000 | 31.251610 | 11.0625 | retain_Eff,retain_Gen,retain_Spe |
| 3 | FAIL | 1.000000 | 0.000000 | 0.000000 | 27.016594 | 11.8125 | retain_Eff,retain_Gen,retain_Spe |
| 4 | FAIL | 1.000000 | 0.000000 | 0.000000 | 27.924298 | 11.4375 | retain_Eff,retain_Gen,retain_Spe |
| 5 | FAIL | 1.000000 | 0.000000 | 0.000000 | 30.203361 | 13.3750 | retain_Eff,retain_Gen,retain_Spe,PPL |
| 6 | FAIL | 1.000000 | 0.000000 | 0.000000 | 25.296392 | 12.0000 | retain_Eff,retain_Gen,PPL |
| 7 | FAIL | 1.000000 | 0.000000 | 0.000000 | 30.512363 | 12.5625 | retain_Eff,retain_Gen,retain_Spe |
| 8 | FAIL | 1.000000 | 0.000000 | 0.000000 | 27.109579 | 12.9375 | retain_Eff,retain_Gen,retain_Spe |
| 9 | FAIL | 1.000000 | 0.000000 | 0.000000 | 23.365873 | 12.1875 | retain_Eff,retain_Gen,retain_Spe |
| 10 | FAIL | 1.000000 | 0.000000 | 0.000000 | 23.935885 | 14.2500 | retain_Eff,retain_Gen,retain_Spe,PPL |

## Aggregate: mean ± sample SD

| Metric | Mean ± SD |
|---|---:|
| Eff ↓ | **0.000000 ± 0.000000** |
| Gen ↓ | **0.000000 ± 0.000000** |
| Spe ↑ | **27.723080 ± 2.884395** |
| PPL ↓ | **12.325000 ± 0.971736** |
| Retain Eff | **32.208858 ± 1.083448** |
| Retain Gen | **31.261879 ± 0.739930** |
| Retain Spe | **28.213617 ± 0.819901** |

## Interpretation

- Raw behavioral forgetting success: **10/10 seeds** (`Eff=0`, `Gen=0`).
- Strict relative-utility acceptance: **0/10 seeds**.
- Every final `selected_checkpoint` is the Setting 5e fallback.
- Every `active_candidate_checkpoint` is the preserved full repaired candidate.

The result supports reproducible behavioral forgetting on the official ZsRE Eff/Gen metrics. Strict utility-gate acceptance is reported separately and was not achieved.

The machine-readable authoritative record is `config/best_runs/zsre/official_setting5e_active_repair_seeds1_10.json`.
