# MCF protected-v2 — official seeds 0–9

## Result

The matching seed-specific checkpoints under
`outputs/gagd_5e_ultra_protected_v2_10seeds/seed{seed}/checkpoint`
were evaluated on the official MCF split for the same seed.

| Seed | Gate | Eff ↓ | Gen ↓ | Spe ↑ | Minimum margin ↑ | PPL ↓ | Retain Eff | Retain Gen | Retain Spe |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | PASS | 0.000 | 0.000 | 13.620 | 0.37500 | 11.0625 | 18.900 | 19.000 | 10.480 |
| 1 | PASS | 0.000 | 0.000 | 14.220 | 0.31250 | 12.3750 | 18.000 | 19.450 | 10.650 |
| 2 | PASS | 0.000 | 0.000 | 9.980 | 0.50000 | 11.0625 | 21.800 | 22.100 | 9.660 |
| 3 | PASS | 0.000 | 0.000 | 11.530 | 0.18750 | 12.0000 | 19.300 | 21.550 | 10.260 |
| 4 | PASS | 0.000 | 0.000 | 13.510 | 0.50000 | 12.0000 | 18.100 | 19.300 | 9.860 |
| 5 | PASS | 0.000 | 0.000 | 14.980 | 0.87500 | 11.0625 | 19.900 | 20.500 | 10.490 |
| 6 | PASS | 0.000 | 0.000 | 13.870 | 0.50000 | 11.0625 | 17.500 | 20.450 | 9.600 |
| 7 | FAIL | 0.000 | 0.000 | 10.310 | 0.09375 | 11.0625 | 20.100 | 22.000 | 10.930 |
| 8 | PASS | 0.000 | 0.000 | 15.950 | 0.50000 | 11.0625 | 18.400 | 18.400 | 10.370 |
| 9 | PASS | 0.000 | 0.000 | 13.170 | 0.18750 | 11.0625 | 17.800 | 20.500 | 10.380 |

## Aggregate: mean ± sample SD

| Metric | Mean ± SD |
|---|---:|
| Eff ↓ | **0.000000 ± 0.000000** |
| Gen ↓ | **0.000000 ± 0.000000** |
| Spe ↑ | **13.114000 ± 1.941621** |
| Minimum margin ↑ | **0.403128 ± 0.225559** |
| PPL ↓ | **11.381250 ± 0.523286** |
| Retain Eff | **18.980000 ± 1.325644** |
| Retain Gen | **20.325000 ± 1.282846** |
| Retain Spe | **10.268000 ± 0.432533** |

## Acceptance contract

A reloaded checkpoint passes only when all of the following hold:

- `Eff ≤ 0.0`
- `Gen ≤ 0.0`
- minimum rewrite/paraphrase margin `≥ 0.10`

The method reached `Eff=0` and `Gen=0` on **10/10 seeds** and passed the
full strict gate on **9/10 seeds**. Seed 7 remained a failure because its
minimum margin was `0.09375`, which is `0.00625` below the frozen threshold.

## Evaluation configuration

- Evaluator: `scripts/mcf_zero_unlearn_official_eval.py`
- Dataset: `data/multi_counterfact.json`
- PPL corpus: `data/wikidata`
- Sample mode: `official`
- Forget records: `50`
- Retain records: `1000`
- Dtype: `bfloat16`
- Device map: `single`
- Checkpoint seed N evaluated on official split seed N

The machine-readable authoritative record is
`config/best_runs/mcf/official_protected_v2_seeds0_9.json`.

## Checkpoint storage

The `model.safetensors` files are approximately 7.21 GB each, so they cannot
be stored as single ordinary Git objects, single Git LFS objects on GitHub, or
single GitHub Release assets. Preserve them as sub-2-GiB split Release assets,
with SHA-256 checksums and reconstruction instructions. Exact training config
files and hashes must be copied from persistent Wulver storage rather than
reconstructed from directory names.
