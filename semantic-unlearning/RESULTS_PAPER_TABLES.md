# Paper tables (Llama-3.2-3B-Instruct, 5 seeds)

Table 1 is the main-paper table (`paper/table1_unlearning.tex`). Table 2 is supplementary and lives
here only. Numbers: Ours and Genie from `outputs/l16_ours_v1` (L19 router, Δe at L16; routers with
base-model rewrites, 20 positives / 20 negatives per fact), Base and baselines from
`outputs/l16_paper_v1`. Regenerate with `python scripts/l16_paper_table.py --root outputs/<tag>`;
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
| GA | 0.025 | 0.277 | 0.36 | 30.82 | 19.38 | 19.07 | 26.24 | 12.67 | 7.98 | 10.80 | 12.68 | 16.12 | 23.20 | 30.55 | 39.57 | 11.69 |
| GD | 0.032 | 0.321 | 0.48 | 31.00 | 19.52 | 19.40 | 26.36 | 12.91 | 9.30 | 11.92 | 14.18 | 16.48 | 23.60 | 30.46 | 39.72 | 11.80 |
| NPO | 0.026 | 0.160 | 0.12 | 29.02 | 17.55 | 17.52 | 25.40 | 12.72 | 4.65 | 7.78 | 8.39 | 16.58 | 18.80 | 24.08 | 30.22 | 14.00 |
| RMU | 0.008 | 0.039 | 1.00 | 11.50 | **0.00** | **0.05** | 7.55 | 11.49 | 0.30 | 6.89 | 0.67 | 11.48 | **0.80** | **0.27** | 0.14 | 11.43 |
| ROME | 4.749 | 5.636 | 18.36 | 11.46 | 33.46 | 31.94 | **29.90** | 11.42 | 10.23 | 15.90 | 32.35 | 11.97 | 2.40 | 4.32 | 30.74 | 11.36 |
| MEMIT | 9.563 | 8.546 | 19.76 | 11.38 | 32.94 | 31.11 | 29.80 | 11.36 | 31.36 | 26.24 | 55.05 | 11.47 | 36.80 | 39.78 | 59.96 | 11.35 |
| ZeroUnlearn | 0.270 | 2.313 | 15.84 | 11.55 | 27.81 | 27.37 | 29.23 | 11.54 | 18.93 | 22.76 | 49.79 | 11.45 | 18.80 | 22.53 | 49.49 | 11.45 |
| **Ours** | **0.001** | **0.027** | **19.84** | **11.31** | 0.17 | 1.24 | 29.79 | **11.31** | **0.10** | **1.49** | **55.07** | **11.31** | 2.00 | 33.61 | **61.89** | **11.31** |
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
