# Llama-3.2-3B-Instruct / MCF — base-model baseline

This is the matched **unmodified base-model** baseline for the current 10-seed MCF SURE result.

## Protocol

- model: `meta-llama/Llama-3.2-3B-Instruct`
- seeds: `1–10`
- evaluator: `scripts/mcf_zero_unlearn_official_eval.py`
- sampling: `official`
- forget evaluation: `50` MCF records per seed
- retain evaluation: `1000` MCF records per seed
- model modification: none
- AWS output root: `outputs/aws_base_mcf_llama32_3b`

## Per-seed forget metrics

| Seed | Eff ↓ | Gen ↓ | Spe ↑ | Spe_success ↑ |
|---:|---:|---:|---:|---:|
| 1 | 16.00 | 14.00 | 12.57 | 88.60 |
| 2 | 14.00 | 17.00 | 8.09 | 81.60 |
| 3 | 16.00 | 19.00 | 9.81 | 82.40 |
| 4 | 10.00 | 15.00 | 13.60 | 86.40 |
| 5 | 4.00 | 12.00 | 14.05 | 90.40 |
| 6 | 18.00 | 14.00 | 13.23 | 85.20 |
| 7 | 18.00 | 14.00 | 9.38 | 76.00 |
| 8 | 14.00 | 13.00 | 12.93 | 90.80 |
| 9 | 14.00 | 16.00 | 10.71 | 86.00 |
| 10 | 12.00 | 14.00 | 10.16 | 88.60 |

## Aggregate base-model result

Your terminal aggregation used **sample SD** across the 10 matched MCF seeds:

| Metric | Mean ± sample SD |
|---|---:|
| Eff ↓ | **13.6000 ± 4.1952** |
| Gen ↓ | **14.8000 ± 2.0440** |
| Spe ↑ | **11.4530 ± 2.0675** |
| Spe_success ↑ | **85.6000 ± 4.5636** |
| PPL ↓ / stable | **11.0625** |

For convention-matched comparison with the existing SURE snapshot, which stores population SD, the same base results are:

| Metric | Mean ± population SD |
|---|---:|
| Eff ↓ | **13.6000 ± 3.9799** |
| Gen ↓ | **14.8000 ± 1.9391** |
| Spe ↑ | **11.4530 ± 1.9614** |
| Spe_success ↑ | **85.6000 ± 4.3294** |
| PPL ↓ / stable | **11.0625** |

## PPL provenance and correction

The matched MCF base-model PPL is **11.0625** and uses the Git-tracked dataset:

`data/wikidata`

The evaluator loads it with `datasets.load_from_disk`, concatenates `train["text"][:20]`, and truncates the tokenizer input to 100 tokens. For Llama-3.2-3B-Instruct, the exact 100-token PPL input has SHA-256:

`3337c0260ab8be7036bafa12784fc45106ca2a5e9d1d6639fb11d139852a5e56`

An earlier AWS diagnostic evaluation produced `PPL=16.625` using `data/wikidata_aws_diag`. That directory produces a different 100-token input with SHA-256:

`30277ca578fecbe96f6866a61082b0d5eef768347b1dece252db7e68c20d2e07`

Therefore `16.625` is **not** used for the MCF Base-vs-SURE comparison.

Because the base checkpoint and PPL text are fixed, base PPL is reported once rather than as a seed mean ± SD.

## Matched Base → SURE comparison

The matched SURE record is `zerounlearn_locked_forget_only_rank2_seeds1_10_20260810`. To avoid mixing SD conventions, this table uses **population SD for both Base and SURE**.

| Metric | Base | SURE |
|---|---:|---:|
| Eff ↓ | **13.6000 ± 3.9799** | **0.0000 ± 0.0000** |
| Gen ↓ | **14.8000 ± 1.9391** | **4.0000 ± 3.6332** |
| Spe ↑ | **11.4530 ± 1.9614** | **27.7110 ± 3.6742** |
| Spe_success ↑ | **85.6000 ± 4.3294** | **96.3000 ± 1.8639** |
| PPL ↓ / stable | **11.0625** | **11.5500 ± 0.6771** |

Base → SURE changes:

- Eff: `13.6 → 0.0` (**100% reduction**)
- Gen: `14.8 → 4.0` (**72.97% reduction**)
- Spe: `11.453 → 27.711` (**+16.258**)
- Spe_success: `85.6 → 96.3` (**+10.7 percentage points**)
- PPL: `11.0625 → 11.55` (**+0.4875**, about **+4.41%**)

## Provenance

- evaluation completed on AWS EC2 on `2026-08-16`
- 10/10 base-model seeds completed
- raw local outputs: `outputs/aws_base_mcf_llama32_3b/seed1.json` through `seed10.json`
- corrected matched PPL output: `outputs/aws_base_mcf_llama32_3b/base_correct_wikidata.json`
- raw per-seed output JSON files are not committed in this snapshot
