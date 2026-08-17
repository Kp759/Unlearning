# Llama-3.2-3B-Instruct / ZsRE — base-model baseline

This is the matched **unmodified base-model** baseline for the current 10-seed ZsRE SURE result.

## Protocol

- model: `meta-llama/Llama-3.2-3B-Instruct`
- seeds: `1–10`
- evaluator: `scripts/zsre_zero_unlearn_official_eval.py`
- forget evaluation: `50` ZsRE records per seed
- retain evaluation: `1000` ZsRE records per seed
- model modification: none
- evaluation batch size: `2`
- dtype: `bf16`
- device map: `single`
- AWS output root: `outputs/aws_base_zsre_llama32_3b`

## Per-seed forget metrics

| Seed | Eff ↓ | Gen ↓ | Spe ↑ | PPL |
|---:|---:|---:|---:|---:|
| 1 | 29.9667 | 27.9000 | 32.3232 | 16.6250 |
| 2 | 32.7778 | 30.8778 | 31.4334 | 16.6250 |
| 3 | 30.7762 | 30.4429 | 26.4127 | 16.6250 |
| 4 | 30.4000 | 31.7333 | 28.6868 | 16.6250 |
| 5 | 36.8667 | 35.0667 | 29.8086 | 16.6250 |
| 6 | 29.3667 | 26.2000 | 25.3200 | 16.6250 |
| 7 | 28.9143 | 31.4619 | 30.4130 | 16.6250 |
| 8 | 41.1508 | 38.3008 | 27.5296 | 16.6250 |
| 9 | 38.6627 | 40.1817 | 24.7825 | 16.6250 |
| 10 | 29.4476 | 30.1476 | 24.7976 | 16.6250 |

## Aggregate base-model result

The current ZsRE SURE record uses **sample standard deviation (`n-1`)**, so this is the primary comparison convention.

| Metric | Base model, mean ± sample SD |
|---|---:|
| Eff ↓ | **32.8329 ± 4.4302** |
| Gen ↓ | **32.2313 ± 4.3898** |
| Spe ↑ | **28.1507 ± 2.7958** |
| PPL ↓ / stable | **16.6250** |

Population-SD values are also stored in the JSON snapshot:

- Eff: `32.8329 ± 4.2029`
- Gen: `32.2313 ± 4.1646`
- Spe: `28.1507 ± 2.6523`
- PPL: `16.6250 ± 0.0000`

`Spe_success` is intentionally omitted. The ZsRE evaluator outputs used for this baseline do not expose the MCF-style `Spe_success` field, so no value is inferred or imputed.

## Matched Base → SURE comparison

Matched SURE record: `zerounlearn_locked_no_neutral_gagd_aggressive_seeds1_10_20260813`.

All SURE uncertainty values below are **sample SD**, matching the current ZsRE best-run record.

| Metric | Base | SURE final |
|---|---:|---:|
| Eff ↓ | **32.8329 ± 4.4302** | **0.025 ± 0.079** |
| Gen ↓ | **32.2313 ± 4.3898** | **0.627 ± 0.510** |
| Spe ↑ | **28.1507 ± 2.7958** | **26.158 ± 2.367** |
| PPL ↓ / stable | **16.6250** | **18.650 ± 2.204** |

Base → SURE changes:

- Eff: `32.8329 → 0.025` (**99.92% reduction**)
- Gen: `32.2313 → 0.627` (**98.05% reduction**)
- Spe: `28.1507 → 26.158` (**−1.9927 absolute**)
- PPL: `16.6250 → 18.650` (**+2.025**, about **+12.18%**)

This baseline makes the forgetting gain explicit: the unmodified base model retains roughly one-third of the original sensitive ZsRE token decisions on both direct and held-out rephrase prompts, while SURE reduces both metrics to well below 1% on average. The tradeoff is a modest decrease in locality/specificity and an increase in PPL.

## PPL provenance

The base run uses:

`data/wikidata_aws_diag`

The evaluator joins `train["text"][:20]`, tokenizes the resulting text, and truncates to a maximum input length of 100 tokens. For the current Llama-3.2-3B-Instruct AWS fixture, the exact 100-token sequence has SHA-256:

`30277ca578fecbe96f6866a61082b0d5eef768347b1dece252db7e68c20d2e07`

The current ZsRE SURE runner is also configured with `WIKIDATA_DIR=data/wikidata_aws_diag` by default. However, this AWS diagnostic dataset is **not Git-tracked**, so Git alone cannot independently reconstruct and hash the historical SURE-run bytes. The comparison is therefore path/protocol matched, with that provenance caveat recorded explicitly.

## Provenance

- evaluation completed on AWS EC2 on `2026-08-16`
- 10/10 base-model seeds completed
- raw local outputs: `outputs/aws_base_zsre_llama32_3b/seed1.json` through `seed10.json`
- raw per-seed output JSON files are not committed in this snapshot
