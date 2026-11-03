# Paper tables (Llama-3.2-3B-Instruct; appendix: Llama-3.1-8B-Instruct, Qwen3-4B; 5 seeds)

Table 1 is the main-paper table (`paper/table1_unlearning.tex`). Table 2 is supplementary and lives
here only. Numbers: every row from `outputs/l16_ours_v1` (L19 router, Δe at L16; routers with
base-model rewrites, 20 positives / 20 negatives per fact). The baselines were rerun there
(jobs 1483319/1483320) so that GA, GD, NPO and RMU train on this router's forget views; the
earlier baseline rows (`outputs/l16_paper_v1`) trained on the regular_v1 views. ROME, MEMIT,
ZeroUnlearn and every MCF cell are unchanged by the rerun. Bold / underline: see `paper/main.tex`. Regenerate with `python scripts/l16_paper_table.py --root outputs/<tag>`;
representation recoverability with `recoverability.slurm` (README_REPRESENTATION_RECOVERABILITY.md).

## Table 1. Unlearning results

Eff, Gen ↓ (forgetting); Spe ↑ (locality); PPL should stay at Base (11.31). Bold: best trained method
(Base and Genie excluded; PPL: closest to Base). Eff = the forget fact on its own prompt. Gen =
paraphrases (MCF, ZsRE), atomic questions (MQuAKE), held-out Level 1 / Level 2 / paraphrase probes
about the person (RWKU). Spe = neighbourhood accuracy (MCF, ZsRE), retain-fact accuracy (MQuAKE),
neighbour recovery (RWKU).

| Method | MCF Eff | Gen | Spe | PPL | ZsRE Eff | Gen | Spe | PPL | MQuAKE Eff | Gen | Spe | PPL | RWKU Eff | Gen | Spe | PPL |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Base | 13.46 | 10.26 | 19.88 | 11.31 | 32.37 | 31.20 | 29.79 | 11.31 | 73.23 | 46.44 | 66.13 | 11.31 | 57.60 | 54.77 | 63.70 | 11.31 |
| GA | 0.025 | 0.277 | 0.36 | 30.82 | 20.50 | 19.81 | 26.43 | 12.63 | 6.22 | 7.42 | 10.73 | 25.54 | 26.40 | 28.52 | 39.24 | 11.73 |
| GD | 0.032 | 0.321 | 0.48 | 31.00 | 20.48 | 19.91 | 26.35 | 12.94 | 7.42 | 7.87 | 11.74 | 29.11 | 25.20 | 27.94 | 36.77 | 11.96 |
| NPO | 0.026 | 0.160 | 0.12 | 29.02 | 18.87 | 18.72 | 25.31 | 12.76 | 4.01 | 4.83 | 6.89 | 23.31 | 18.80 | 18.95 | 22.26 | 12.00 |
| RMU | 0.008 | 0.039 | 1.00 | 11.50 | 2.62 | 2.98 | 12.57 | 11.45 | 0.25 | 0.00 | 0.40 | 11.76 | 0.00 | 0.00 | 0.07 | 11.43 |
| ROME | 4.749 | 5.636 | 18.36 | 11.46 | 33.46 | 31.94 | 29.90 | 11.42 | 10.23 | 15.90 | 32.35 | 11.97 | 2.40 | 4.32 | 30.74 | 11.36 |
| MEMIT | 9.563 | 8.546 | 19.76 | 11.38 | 32.94 | 31.11 | 29.80 | 11.36 | 31.36 | 26.24 | 55.05 | 11.47 | 36.80 | 39.78 | 59.95 | 11.34 |
| ZeroUnlearn | 0.270 | 2.313 | 15.84 | 11.55 | 27.81 | 27.37 | 29.23 | 11.54 | 18.93 | 22.76 | 49.79 | 11.45 | 18.80 | 22.53 | 49.49 | 11.45 |
| **Ours** | **0.001** | **0.027** | **19.84** | **11.31** | **0.17** | **1.24** | 29.79 | **11.31** | **0.10** | 1.49 | **55.07** | **11.31** | 2.00 | 33.61 | **61.89** | **11.31** |
| Genie | 0.001 | 0.001 | 19.88 | 11.31 | 0.17 | 0.17 | 29.79 | 11.31 | 0.10 | 0.10 | 56.21 | 11.31 | 0.00 | 54.77 | 63.70 | 11.31 |
| Genie (entity) | – | – | – | – | – | – | – | – | – | – | – | – | 0.00 | 21.04 | 63.70 | 11.31 |

Notes. MQuAKE Spe over retain facts that do not repeat a forget fact: Base 67.3, Ours 66.0. Genie =
the same rows with every official forget prompt routed to its own row (router not run); the gap
between Ours and Genie is the cost of routing. RWKU Gen asks held-out facts about the forgotten
person, which have no row of their own; the fact-level router (trained against same-person
competitor facts) declines to fire on them, hence 33.6 against 18.0 for the earlier, looser router.
Genie (entity) routes every such probe to the person's row and bounds what entity-level routing
reaches with these rows (21.0); the `RWKU_GATE=subject` run is the entity-level policy for that cell.

Changes against the earlier table (`RESULTS_L16_PAPER_FINAL.md`, Ours (L16) with `regular_v1`
routers): MCF Gen 0.086 → 0.027; ZsRE 0.30 / 1.44 → 0.17 / 1.24; MQuAKE Gen 7.20 → 1.49; RWKU Spe
57.4 → 61.9; RWKU Gen 18.0 → 33.6 (routing, see above).

## Table 3 (ablation). Where the classifier reads and where Δe is written

`paper/table_ablation_layers.tex`. Llama-3.2-3B-Instruct, 5 seeds; same router recipe and row
training in every row; PPL = 11.31 (Base) in every cell. Runs: `outputs/l16_ours_v1` (main),
`outputs/onepass_l19_v1`, `outputs/onepass_l16_v1`.

| Variant | read | write | MCF Eff / Gen / Spe | ZsRE Eff / Gen / Spe | MQuAKE Eff / Gen / Spe (Spe†) | RWKU Eff / Gen / Spe |
|---|---|---|---|---|---|---|
| Base | – | – | 13.46 / 10.26 / 19.88 | 32.37 / 31.20 / 29.79 | 73.23 / 46.44 / 66.13 (67.34) | 57.60 / 54.77 / 63.70 |
| **Ours** (two passes) | 19 | 16 | 0.001 / 0.027 / 19.84 | **0.17 / 1.24** / 29.79 | **0.10 / 1.49** / 55.07 (65.98) | **2.00 / 33.61** / 61.89 |
| one pass, read late | 19 | 19 | 0.001 / 0.027 / 19.84 | 0.41 / 1.48 / 29.79 | 0.53 / 1.83 / 55.53 (66.50) | 2.40 / 34.10 / 61.82 |
| one pass, read early | 16 | 16 | 0.001 / **0.013** / 19.80 | **0.04** / 2.76 / 29.79 | 0.17 / 2.62 / 55.25 (66.18) | 2.40 / 34.81 / 62.18 |
| Genie (19 / 16) | 19 | 16 | 0.001 / 0.001 / 19.88 | 0.17 / 0.17 / 29.79 | 0.10 / 0.10 / 56.21 (67.34) | 0.00 / 54.77 / 63.70 |
| Genie (19 / 19) | 19 | 19 | 0.001 / 0.001 / 19.88 | 0.41 / 0.41 / 29.79 | 0.53 / 0.53 / 56.23 (67.34) | 0.40 / 54.77 / 63.70 |
| Genie (16 / 16) | 16 | 16 | 0.001 / 0.001 / 19.88 | 0.04 / 0.04 / 29.79 | 0.17 / 0.17 / 56.21 (67.34) | 0.40 / 54.77 / 63.70 |
| Genie (entity), write 16 | – | 16 | – | – | – | 0.00 / 21.04 / 63.70 |
| Genie (entity), write 19 | – | 19 | – | – | – | 0.40 / 40.83 / 63.70 |

Reading. The read layer sets what the router catches: a block-16 reader misses paraphrases (ZsRE
Gen 2.76 vs 1.24 with a block-19 reader) while erasing the request prompt itself slightly better
(ZsRE Eff 0.04). The write layer sets how far the rows generalise: written at block 16 they remove
more of MQuAKE's atomic questions (Gen 1.49 vs 1.83) and, under entity-level routing, far more of a
person's other facts (RWKU 21.0 vs 40.8). Genie rows isolate the rows from the router: the gap
between a variant and its genie is routing, the gap between genies is the write layer. The two-pass
variant costs one prompt-only forward truncated at block 19 per request, nothing per generated
token. Representation recoverability (Table 2) is the same for the three variants (MCF 6.7–7.2,
ZsRE 7.4, MQuAKE 5.8).

## Table 4 (Appendix A). Llama-3.1-8B-Instruct

`outputs/paper_llama31_8b_r20_w18_ours_v1` (main: classifier at block 20, Δe at block 18, of 32) and
`outputs/paper_llama31_8b_onepass_v1` (one-pass ablation at block 22). 5 seeds, mean ± std; PPL is
11.033 for every row (zero variance). Read layer from the classifier sweep on MCF seed 1
(`outputs/layer_smoke_llama31_8b`); write layer at the same relative depth as the 3B model's.

| Dataset | Method | Eff ↓ | Gen ↓ | Spe ↑ | Spe† ↑ |
|---|---|---|---|---|---|
| MCF | Base | 17.089 ± 2.750 | 13.007 ± 4.123 | 22.480 ± 4.653 | – |
| MCF | **Ours** | **0.000 ± 0.000** | **0.353 ± 0.365** | **22.440 ± 4.657** | – |
| MCF | Genie | 0.000 ± 0.000 | 0.002 ± 0.001 | 22.480 ± 4.653 | – |
| zsRE | Base | 39.807 ± 1.839 | 36.436 ± 2.584 | 30.565 ± 2.481 | – |
| zsRE | **Ours** | **0.080 ± 0.179** | **3.273 ± 3.516** | **30.565 ± 2.481** | – |
| zsRE | Genie | 0.080 ± 0.179 | 0.080 ± 0.179 | 30.565 ± 2.481 | – |
| MQuAKE | Base | 72.817 ± 3.315 | 63.482 ± 2.431 | 67.757 ± 0.413 | 70.098 ± 0.507 |
| MQuAKE | **Ours** | **0.790 ± 0.597** | **4.954 ± 1.841** | **58.414 ± 1.603** | **69.692 ± 0.716** |
| MQuAKE | Genie | 0.790 ± 0.597 | 0.790 ± 0.597 | 58.749 ± 1.347 | 70.098 ± 0.507 |
| RWKU | Base | 78.400 ± 2.608 | 83.983 ± 2.541 | 84.711 ± 2.458 | – |
| RWKU | **Ours** | **3.600 ± 5.899** | **49.211 ± 6.417** | **82.619 ± 3.790** | – |
| RWKU | Genie | 1.600 ± 1.673 | 83.983 ± 2.541 | 84.711 ± 2.458 | – |
| RWKU | Genie (entity) | 1.600 ± 1.673 | 27.654 ± 10.716 | 84.711 ± 2.458 | – |

One-pass ablation (classifier and Δe both at block 22):

| Dataset | Eff ↓ | Gen ↓ | Spe ↑ | Spe† ↑ | Genie (entity) Gen |
|---|---|---|---|---|---|
| MCF | 0.000 ± 0.000 | 0.353 ± 0.363 | 22.440 ± 4.657 | – | – |
| zsRE | 0.133 ± 0.298 | 4.093 ± 2.883 | 30.565 ± 2.481 | – | – |
| MQuAKE | 1.317 ± 1.069 | 4.940 ± 1.864 | 58.490 ± 1.519 | 69.803 ± 0.633 | – |
| RWKU | 3.600 ± 5.899 | 43.938 ± 13.592 | 82.441 ± 4.256 | – | 50.288 ± 12.986 |

Reading. Request-prompt recall falls by 99.5–100% (MCF, zsRE, MQuAKE) and 95% (RWKU), held-out
recall by 91–97% on MCF, zsRE and MQuAKE, with locality and perplexity at the base model's level.
Genie removes 99.7–99.99% of held-out recall with the same vectors, so the remaining leak is routing,
as on the 3B model. Writing early (18) against one pass (22): lower request-prompt recall on zsRE
and MQuAKE, lower zsRE held-out recall (3.27 vs 4.09), and an entity-level RWKU genie of 27.7 vs
50.3 — the 3B model's write-early effect, reproduced. The write sweep on seed 1 prefers block 17 by
a small margin (held-out Gen 1.17 vs 1.37 averaged over MCF and zsRE); not rerun. Baselines (job 1483247, same run dir; GA/GD/NPO train the MLP down projections, RMU block 8):

| Dataset | Method | Eff ↓ | Gen ↓ | Spe ↑ | Spe† ↑ | PPL |
|---|---|---|---|---|---|---|
| MCF | Base | 17.09 ± 2.75 | 13.01 ± 4.12 | 22.48 ± 4.65 | – | 11.03 ± 0.00 |
| MCF | GA | 0.02 ± 0.03 | 0.02 ± 0.03 | 0.00 ± 0.00 | – | 88.85 ± 74.21 |
| MCF | GD | 0.01 ± 0.01 | 0.01 ± 0.01 | 0.00 ± 0.00 | – | 74.78 ± 64.55 |
| MCF | NPO | 0.01 ± 0.02 | 0.01 ± 0.02 | 0.00 ± 0.00 | – | 95.48 ± 129.35 |
| MCF | RMU | 0.00 ± 0.00 | 0.01 ± 0.03 | 0.60 ± 0.58 | – | 10.73 ± 0.07 |
| MCF | ROME | 3.59 ± 1.99 | 4.36 ± 2.13 | 20.00 ± 4.60 | – | 11.12 ± 0.03 |
| MCF | MEMIT | 6.32 ± 1.63 | 7.73 ± 2.95 | 21.48 ± 4.64 | – | 11.09 ± 0.04 |
| MCF | ZeroUnlearn | 1.93 ± 1.23 | 4.48 ± 1.57 | 18.32 ± 3.96 | – | 11.17 ± 0.12 |
| MCF | **Ours** | 0.00 ± 0.00 | 0.35 ± 0.37 | 22.44 ± 4.66 | – | 11.03 ± 0.00 |
| MCF | Genie | 0.00 ± 0.00 | 0.00 ± 0.00 | 22.48 ± 4.65 | – | 11.03 ± 0.00 |
| MCF | Genie (entity) | – | – | – | – | – |
| zsRE | Base | 39.81 ± 1.84 | 36.44 ± 2.58 | 30.56 ± 2.48 | – | 11.03 ± 0.00 |
| zsRE | GA | 23.86 ± 3.43 | 23.47 ± 4.45 | 25.97 ± 1.40 | – | 13.26 ± 0.35 |
| zsRE | GD | 24.01 ± 3.26 | 23.67 ± 4.21 | 25.97 ± 1.58 | – | 13.89 ± 0.45 |
| zsRE | NPO | 21.43 ± 2.90 | 21.20 ± 4.72 | 25.02 ± 1.31 | – | 13.29 ± 0.51 |
| zsRE | RMU | 1.13 ± 2.53 | 0.93 ± 2.08 | 6.85 ± 5.52 | – | 10.74 ± 0.16 |
| zsRE | ROME | 34.64 ± 3.04 | 34.88 ± 3.57 | 30.53 ± 2.02 | – | 11.10 ± 0.02 |
| zsRE | MEMIT | 37.65 ± 4.03 | 37.27 ± 2.61 | 30.82 ± 2.08 | – | 11.07 ± 0.02 |
| zsRE | ZeroUnlearn | 33.77 ± 3.49 | 32.52 ± 3.13 | 30.43 ± 2.07 | – | 11.18 ± 0.09 |
| zsRE | **Ours** | 0.08 ± 0.18 | 3.27 ± 3.52 | 30.56 ± 2.48 | – | 11.03 ± 0.00 |
| zsRE | Genie | 0.08 ± 0.18 | 0.08 ± 0.18 | 30.56 ± 2.48 | – | 11.03 ± 0.00 |
| zsRE | Genie (entity) | – | – | – | – | – |
| MQuAKE | Base | 72.82 ± 3.31 | 63.48 ± 2.43 | 67.76 ± 0.41 | 70.10 ± 0.51 | 11.03 ± 0.00 |
| MQuAKE | GA | 2.38 ± 2.01 | 2.28 ± 1.95 | 3.82 ± 2.89 | 4.31 ± 3.37 | 2.02e+05 ± 4.47e+05 |
| MQuAKE | GD | 3.12 ± 2.87 | 2.45 ± 2.08 | 4.62 ± 3.21 | 5.16 ± 3.65 | 1969.26 ± 2946.11 |
| MQuAKE | NPO | 2.35 ± 2.45 | 2.43 ± 2.37 | 3.67 ± 2.82 | 3.98 ± 3.00 | 175.10 ± 195.11 |
| MQuAKE | RMU | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.10 ± 0.15 | 0.12 ± 0.18 | 10.78 ± 0.23 |
| MQuAKE | ROME | 7.88 ± 1.53 | 17.18 ± 1.47 | 32.05 ± 4.06 | 36.96 ± 4.83 | 11.19 ± 0.05 |
| MQuAKE | MEMIT | 28.02 ± 2.30 | 27.84 ± 2.42 | 55.17 ± 1.07 | 62.09 ± 1.09 | 11.16 ± 0.03 |
| MQuAKE | ZeroUnlearn | 24.96 ± 3.77 | 28.26 ± 4.53 | 55.30 ± 1.41 | 63.13 ± 1.19 | 11.20 ± 0.20 |
| MQuAKE | **Ours** | 0.79 ± 0.60 | 4.95 ± 1.84 | 58.41 ± 1.60 | 69.69 ± 0.72 | 11.03 ± 0.00 |
| MQuAKE | Genie | 0.79 ± 0.60 | 0.79 ± 0.60 | 58.75 ± 1.35 | 70.10 ± 0.51 | 11.03 ± 0.00 |
| MQuAKE | Genie (entity) | – | – | – | – | – |
| RWKU | Base | 78.40 ± 2.61 | 83.98 ± 2.54 | 84.71 ± 2.46 | – | 11.03 ± 0.00 |
| RWKU | GA | 15.20 ± 11.80 | 16.71 ± 11.33 | 14.83 ± 10.47 | – | 12.42 ± 0.53 |
| RWKU | GD | 15.60 ± 11.70 | 17.42 ± 11.57 | 15.68 ± 11.73 | – | 12.72 ± 0.52 |
| RWKU | NPO | 5.60 ± 2.97 | 11.56 ± 10.77 | 10.40 ± 7.41 | – | 12.64 ± 0.13 |
| RWKU | RMU | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | – | 10.85 ± 0.13 |
| RWKU | ROME | 6.40 ± 4.56 | 7.27 ± 7.18 | 51.81 ± 17.95 | – | 11.08 ± 0.03 |
| RWKU | MEMIT | 53.20 ± 10.26 | 62.22 ± 5.79 | 80.24 ± 4.05 | – | 11.07 ± 0.02 |
| RWKU | ZeroUnlearn | 33.20 ± 10.35 | 46.48 ± 4.96 | 75.04 ± 6.93 | – | 11.15 ± 0.05 |
| RWKU | **Ours** | 3.60 ± 5.90 | 49.21 ± 6.42 | 82.62 ± 3.79 | – | 11.03 ± 0.00 |
| RWKU | Genie | 1.60 ± 1.67 | 83.98 ± 2.54 | 84.71 ± 2.46 | – | 11.03 ± 0.00 |
| RWKU | Genie (entity) | 1.60 ± 1.67 | 27.65 ± 10.72 | 84.71 ± 2.46 | – | 11.03 ± 0.00 |

## Table 5 (Appendix A). Qwen3-4B

`outputs/paper_qwen3_4b_r26_w23_ours_v1`: classifier at block 26 (classifier sweep, MCF seed 1), Δe at
block 23 (lowest Gen averaged over MCF and zsRE among W in {16, 18, 21, 23, 26}, seed-1 smoke test,
`outputs/layer_smoke_qwen3_4b_r26_w*`). Baselines job 1483248 (RMU block 9). ROME diverges on MCF
and MQuAKE on this model. Genie leaves 9.3 of 30.2 points of zsRE and 16.1 of 58.8 of MQuAKE
request-prompt recall: on this model the remaining leak there is editing, not classification.

| Dataset | Method | Eff ↓ | Gen ↓ | Spe ↑ | Spe† ↑ | PPL |
|---|---|---|---|---|---|---|
| MCF | Base | 11.13 ± 2.74 | 9.78 ± 3.38 | 12.64 ± 2.90 | – | 14.07 ± 0.00 |
| MCF | GA | 0.02 ± 0.01 | 0.79 ± 1.08 | 0.72 ± 0.72 | – | 650.50 ± 847.86 |
| MCF | GD | 0.03 ± 0.03 | 0.79 ± 1.07 | 0.80 ± 0.73 | – | 470.91 ± 516.80 |
| MCF | NPO | 0.13 ± 0.15 | 0.38 ± 0.42 | 0.76 ± 0.78 | – | 175.93 ± 137.76 |
| MCF | RMU | 0.03 ± 0.03 | 0.04 ± 0.04 | 1.00 ± 1.08 | – | 13.67 ± 0.26 |
| MCF | ROME | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.00 ± 0.00 | – | 2139.96 ± 4736.62 |
| MCF | MEMIT | 0.74 ± 0.77 | 2.33 ± 0.81 | 12.44 ± 3.03 | – | 14.21 ± 0.04 |
| MCF | ZeroUnlearn | 0.04 ± 0.07 | 0.59 ± 0.47 | 9.72 ± 1.70 | – | 15.24 ± 0.85 |
| MCF | **Ours** | 0.00 ± 0.00 | 0.12 ± 0.15 | 12.52 ± 2.90 | – | 14.07 ± 0.00 |
| MCF | Genie | 0.00 ± 0.00 | 0.12 ± 0.15 | 12.64 ± 2.90 | – | 14.07 ± 0.00 |
| MCF | Genie (entity) | – | – | – | – | – |
| zsRE | Base | 30.21 ± 3.32 | 30.37 ± 3.22 | 35.59 ± 4.34 | – | 14.07 ± 0.00 |
| zsRE | GA | 21.85 ± 3.43 | 21.18 ± 4.19 | 33.54 ± 3.64 | – | 17.13 ± 0.35 |
| zsRE | GD | 21.98 ± 3.65 | 21.21 ± 4.11 | 33.32 ± 3.91 | – | 18.29 ± 0.36 |
| zsRE | NPO | 21.38 ± 4.25 | 20.27 ± 4.71 | 33.90 ± 3.91 | – | 17.53 ± 0.39 |
| zsRE | RMU | 10.33 ± 9.18 | 11.88 ± 9.86 | 27.37 ± 7.78 | – | 13.99 ± 0.25 |
| zsRE | ROME | 14.82 ± 4.63 | 14.90 ± 4.77 | 25.06 ± 3.32 | – | 15.93 ± 0.70 |
| zsRE | MEMIT | 27.45 ± 3.56 | 26.98 ± 3.13 | 35.04 ± 4.38 | – | 14.14 ± 0.06 |
| zsRE | ZeroUnlearn | 24.73 ± 3.07 | 25.39 ± 3.24 | 34.54 ± 5.14 | – | 15.36 ± 0.69 |
| zsRE | **Ours** | 9.33 ± 2.53 | 11.22 ± 2.95 | 35.59 ± 4.34 | – | 14.07 ± 0.00 |
| zsRE | Genie | 9.33 ± 2.53 | 9.43 ± 2.58 | 35.59 ± 4.34 | – | 14.07 ± 0.00 |
| zsRE | Genie (entity) | – | – | – | – | – |
| MQuAKE | Base | 58.82 ± 2.10 | 24.91 ± 3.95 | 51.85 ± 0.57 | 52.13 ± 0.43 | 14.07 ± 0.00 |
| MQuAKE | GA | 11.00 ± 2.02 | 15.66 ± 3.77 | 16.81 ± 2.36 | 18.52 ± 2.80 | 80.77 ± 15.39 |
| MQuAKE | GD | 11.19 ± 1.79 | 15.94 ± 3.95 | 17.16 ± 2.25 | 18.93 ± 2.66 | 99.91 ± 23.55 |
| MQuAKE | NPO | 11.31 ± 1.65 | 16.47 ± 3.67 | 17.08 ± 2.63 | 18.79 ± 3.24 | 60.55 ± 6.66 |
| MQuAKE | RMU | 1.81 ± 0.69 | 2.25 ± 0.99 | 5.35 ± 2.39 | 5.71 ± 2.71 | 14.04 ± 0.25 |
| MQuAKE | ROME | 0.00 ± 0.00 | 0.00 ± 0.00 | 0.01 ± 0.01 | 0.01 ± 0.02 | 8.23e+06 ± 8.14e+06 |
| MQuAKE | MEMIT | 26.95 ± 2.15 | 20.96 ± 3.74 | 48.16 ± 0.90 | 51.40 ± 0.35 | 14.25 ± 0.15 |
| MQuAKE | ZeroUnlearn | 20.43 ± 2.96 | 18.85 ± 3.38 | 37.51 ± 0.91 | 40.36 ± 1.04 | 17.52 ± 1.00 |
| MQuAKE | **Ours** | 16.05 ± 1.63 | 15.15 ± 3.23 | 45.98 ± 1.02 | 51.58 ± 0.68 | 14.07 ± 0.00 |
| MQuAKE | Genie | 16.05 ± 1.63 | 14.35 ± 3.10 | 46.44 ± 0.74 | 52.13 ± 0.43 | 14.07 ± 0.00 |
| MQuAKE | Genie (entity) | – | – | – | – | – |
| RWKU | Base | 26.00 ± 8.72 | 30.43 ± 12.37 | 24.89 ± 9.10 | – | 14.07 ± 0.00 |
| RWKU | GA | 12.40 ± 2.97 | 15.15 ± 4.01 | 12.59 ± 4.81 | – | 14.74 ± 0.16 |
| RWKU | GD | 12.40 ± 2.61 | 15.28 ± 4.12 | 12.81 ± 4.67 | – | 15.05 ± 0.17 |
| RWKU | NPO | 13.20 ± 2.28 | 15.74 ± 5.41 | 12.38 ± 4.35 | – | 15.99 ± 1.44 |
| RWKU | RMU | 0.80 ± 1.10 | 0.36 ± 0.81 | 0.07 ± 0.17 | – | 13.94 ± 0.09 |
| RWKU | ROME | 2.80 ± 6.26 | 4.09 ± 4.58 | 14.62 ± 6.52 | – | 14.10 ± 0.41 |
| RWKU | MEMIT | 18.40 ± 7.13 | 15.22 ± 6.42 | 23.07 ± 8.56 | – | 14.09 ± 0.06 |
| RWKU | ZeroUnlearn | 9.60 ± 5.55 | 13.55 ± 4.20 | 21.04 ± 8.27 | – | 14.46 ± 0.34 |
| RWKU | **Ours** | 4.00 ± 5.66 | 20.71 ± 6.40 | 24.53 ± 9.07 | – | 14.07 ± 0.00 |
| RWKU | Genie | 2.00 ± 1.41 | 30.43 ± 12.37 | 24.89 ± 9.10 | – | 14.07 ± 0.00 |
| RWKU | Genie (entity) | 2.00 ± 1.41 | 20.94 ± 6.90 | 24.89 ± 9.10 | – | 14.07 ± 0.00 |

## Table 2 (supplementary). Representation recoverability

Does the forgotten answer remain linearly readable from the residual stream? Motivated by Gao et al.
(2026, arXiv:2604.08271). Per block, the residual stream at the last prompt token is decoded by
(a) the logit lens, (b) a tuned lens (affine map to the final block fitted on the split's retain-side
prompts), (c) a supervised probe trained on retain prompts that share the forget answers but name no
forget subject. Each cell is the worst block, % of official forget prompts recovered (↓). Base and
Ours as in Table 1; 5 seeds; `scripts/representation_recoverability.py`.

| Readout | | MCF | ZsRE | MQuAKE | RWKU |
|---|---|---|---|---|---|
| logit lens | Base | 27.5 | 11.6 | 55.5 | 37.4\* |
| | Ours | 4.0 | 1.0 | 6.7 | 16.0\* |
| tuned lens (RR) | Base | 20.0 | 7.8 | 42.4 | 33.6\* |
| | Ours | 7.2 | 7.4 | 5.8 | 19.6\* |
| answer probe | Base | 80.2 | – | 93.9 | – |
| | Ours | 71.1 | – | 80.6 | – |

\* RWKU pending the rerun with the rank-fixed tuned lens. "–": too few retain prompts share a forget
answer for the probe (fewer than 5 answer classes). Baseline rows follow from the `l16_paper_v1`
recoverability run.

Reading. Below the write layer every block is the base model, so Ours' worst block for the lenses
is the one just below the write (block 15); from the write layer upward the logit lens and the tuned
lens read 0.0 on the forget prompts while control prompts stay at base. The supervised probe still
classifies 60–80% of forget prompts by the forgotten answer above the write layer (base 70–95%,
chance 2%): Δe moves the state off the answer's unembedding direction, not out of the region a
probe trained on other subjects assigns to that answer. This is the representation-level residue
the paper's critique predicts, and the number a white-box adversary should be assumed to have.
