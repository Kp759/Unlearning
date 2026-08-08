# ZsRE Setting 5e + Active LM-Head Repair — Llama-3.2-3B-Instruct

## Result

All configured metric and utility gates passed for all ten seeds.

- Base model: `meta-llama/Llama-3.2-3B-Instruct`
- Seeds: 1–10
- Accepted: 10/10
- Rejected: 0/10
- Selected repair scale: 1.0 for every seed
- Forget Eff: **0.0000 ± 0.0000**
- Forget Gen: **0.0000 ± 0.0000**
- Retain calibration records: 384
- Utility-drop tolerance: 1.20 percentage points
- Maximum PPL ratio: 1.16
- Target Eff: 0.0
- Target Gen: 0.0

Uncertainty is sample standard deviation across ten seeds (`ddof=1`).

| Method | Forget Eff ↓ | Forget Gen ↓ | Forget Spe ↑ | Retain Eff ↑ | Retain Gen ↑ | Retain Spe ↑ | PPL ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Setting 5e | 20.8133 ± 3.1632 | 20.8110 ± 3.7952 | 27.7231 ± 2.8844 | 32.8411 ± 1.0764 | 31.8887 ± 0.7420 | 28.4971 ± 0.9018 | 11.9250 ± 0.5942 |
| Selected active repair | **0.0000 ± 0.0000** | **0.0000 ± 0.0000** | 27.7231 ± 2.8844 | 32.2089 ± 1.0834 | 31.2619 ± 0.7399 | 28.2136 ± 0.8199 | 12.3250 ± 0.9717 |

## Per-seed selected results

| Seed | Accepted | Scale | Forget Eff | Forget Gen | PPL |
|---:|:---:|---:|---:|---:|---:|
| 1 | yes | 1.0 | 0.0 | 0.0 | 11.6250 |
| 2 | yes | 1.0 | 0.0 | 0.0 | 11.0625 |
| 3 | yes | 1.0 | 0.0 | 0.0 | 11.8125 |
| 4 | yes | 1.0 | 0.0 | 0.0 | 11.4375 |
| 5 | yes | 1.0 | 0.0 | 0.0 | 13.3750 |
| 6 | yes | 1.0 | 0.0 | 0.0 | 12.0000 |
| 7 | yes | 1.0 | 0.0 | 0.0 | 12.5625 |
| 8 | yes | 1.0 | 0.0 | 0.0 | 12.9375 |
| 9 | yes | 1.0 | 0.0 | 0.0 | 12.1875 |
| 10 | yes | 1.0 | 0.0 | 0.0 | 14.2500 |

## Run provenance

- Repository commit used for the run: `a8d6b85450d662d28d31ee2f0f01e1c3b4be706e`
- Python: 3.10.20
- Hardware: NVIDIA A100-SXM4-80GB
- Slurm job: 1166025
- Runner: `scripts/zsre_bf16_safe_active_repair_v2.py`

Canonical machine-readable record: `config/best_runs/zsre/setting5e_active_repair_u1p20_ppl1p16_cal384_seeds1_10.json`.
Canonical Markdown record: `config/best_runs/zsre/setting5e_active_repair_u1p20_ppl1p16_cal384_seeds1_10.md`.

## Checkpoint provenance

The model weights remain in persistent Wulver storage. Checkpoint weight hashes are still marked pending local capture and should be archived before publication-grade reproducibility is claimed.
