# Results: base, baselines, ours (L16), genie — 4 benchmarks × 5 seeds (round 2)

Model: Llama-3.2-3B-Instruct. Code: commits `4076d52` (round 2) and `a5a47d7` (baseline early
stopping). Outputs: `outputs/l16_paper_v1/<dataset>/seed<S>/<method>/`. Table:
`python scripts/l16_paper_table.py` → `outputs/l16_paper_v1/paper_table.{md,tex,json}`.
Round 1 (before the ZsRE router fix and trained zero rows): `RESULTS_L16_PAPER_ROUND1.md`.

Changes from round 1:
- **ZsRE router** trained on base-model rewordings of each direct question (`reworded_v2`:
  answer-consistency + near-duplicate filters; official rephrases never read).
- **`--oracle-unrouted`**: facts the router never routes are trained with their own row forced
  during training only (test-time routing unchanged). ZsRE: 0–2 such facts per seed; RWKU:
  7, 5, 6, 6, 12.
- **Baselines** GA, GD, NPO(+retain), RMU, ZeroUnlearn in the same harness (same forget views,
  seeds, splits and official evaluators; retain = base-model self-generated text). GA/GD/NPO/RMU
  stop as soon as the mean answer probability on the forget **training** views is ≤ 1e-4
  (a fixed 5-epoch budget had driven MCF PPL to 1e23 and Spe to 0).
- **MQuAKE Spe†**: retain Eff over retain facts that are not also forget facts.

All 160 runs complete (8 methods × 4 datasets × 5 seeds). Genie routed 100% of direct prompts on
every seed; v2.2 calibrated on all MCF seeds.

## Main table (mean ± std over seeds 1–5)

| Dataset | Method | Eff ↓ | Gen ↓ | Spe ↑ | Spe† ↑ | PPL |
|---|---|---|---|---|---|---|
| **MCF** | Base | 13.456 ± 2.609 | 10.262 ± 3.496 | 19.880 ± 4.660 | – | 11.313 |
| | GA | 0.025 ± 0.027 | 0.277 ± 0.231 | 0.360 ± 0.358 | – | 30.820 ± 26.323 |
| | GD | 0.032 ± 0.032 | 0.321 ± 0.271 | 0.480 ± 0.415 | – | 31.003 ± 23.602 |
| | NPO | 0.026 ± 0.028 | 0.160 ± 0.191 | 0.120 ± 0.268 | – | 29.020 ± 12.080 |
| | RMU | 0.008 ± 0.005 | **0.039 ± 0.045** | 1.000 ± 0.707 | – | 11.501 ± 0.061 |
| | ZeroUnlearn | 0.094 ± 0.089 | 1.507 ± 0.699 | 14.760 ± 3.366 | – | 11.587 ± 0.110 |
| | **Ours (L16)** | **0.001 ± 0.000** | 0.051 ± 0.106 | **19.240 ± 4.894** | – | **11.313** |
| | Genie | 0.001 ± 0.000 | 0.002 ± 0.001 | 19.880 ± 4.660 | – | 11.313 |
| **ZsRE** | Base | 32.371 ± 2.941 | 31.204 ± 2.589 | 29.788 ± 2.163 | – | 11.313 |
| | GA | 19.382 ± 2.919 | 19.066 ± 3.071 | 26.243 ± 1.859 | – | 12.667 ± 0.240 |
| | GD | 19.515 ± 2.892 | 19.399 ± 2.908 | 26.364 ± 2.110 | – | 12.909 ± 0.164 |
| | NPO | 17.548 ± 3.674 | 17.522 ± 3.855 | 25.396 ± 2.061 | – | 12.723 ± 0.163 |
| | RMU | **0.000 ± 0.000** | **0.050 ± 0.112** | 7.545 ± 1.524 | – | 11.488 ± 0.046 |
| | ZeroUnlearn | 27.140 ± 1.524 | 26.586 ± 2.014 | 29.282 ± 2.049 | – | 11.553 ± 0.089 |
| | **Ours (L16)** | 0.300 ± 0.671 | 1.440 ± 1.744 | **29.788 ± 2.163** | – | **11.313** |
| | Genie | 0.000 ± 0.000 | 0.000 ± 0.000 | 29.788 ± 2.163 | – | 11.313 |
| **MQuAKE** | Base | 73.227 ± 1.534 | 46.437 ± 3.481 | 66.129 ± 0.275 | 67.338 ± 0.482 | 11.313 |
| | GA | 7.978 ± 6.022 | 10.796 ± 4.837 | 12.682 ± 8.241 | 13.955 ± 8.822 | 16.119 ± 2.503 |
| | GD | 9.304 ± 5.858 | 11.919 ± 4.706 | 14.176 ± 8.166 | 15.604 ± 8.604 | 16.480 ± 2.433 |
| | NPO | 4.649 ± 1.283 | 7.782 ± 1.488 | 8.386 ± 2.543 | 9.341 ± 2.893 | 16.580 ± 1.456 |
| | RMU | 0.298 ± 0.211 | **6.891 ± 0.784** | 0.672 ± 0.185 | 0.641 ± 0.205 | 11.484 ± 0.084 |
| | ZeroUnlearn | 17.144 ± 2.387 | 20.640 ± 4.054 | 45.274 ± 1.727 | 52.216 ± 1.910 | 11.337 ± 0.097 |
| | **Ours (L16)** | **0.000 ± 0.000** | 7.201 ± 5.466 | **55.426 ± 1.141** | **66.401 ± 0.464** | **11.313** |
| | Genie | 0.000 ± 0.000 | 0.000 ± 0.000 | 56.205 ± 0.990 | 67.338 ± 0.482 | 11.313 |
| **RWKU** | Base | 57.600 ± 2.608 | 54.765 ± 3.954 | 63.697 ± 2.382 | – | 11.313 |
| | GA | 23.200 ± 16.162 | 30.553 ± 15.095 | 39.571 ± 13.214 | – | 11.692 ± 0.129 |
| | GD | 23.600 ± 17.344 | 30.456 ± 16.051 | 39.723 ± 14.581 | – | 11.795 ± 0.147 |
| | NPO | 18.800 ± 10.545 | 24.081 ± 18.154 | 30.221 ± 11.845 | – | 13.995 ± 4.787 |
| | RMU | **0.800 ± 1.789** | **0.267 ± 0.596** | 0.143 ± 0.321 | – | 11.430 ± 0.098 |
| | ZeroUnlearn | 12.000 ± 4.472 | 9.918 ± 3.610 | 28.835 ± 11.343 | – | 11.581 ± 0.034 |
| | **Ours (L16)** | 2.400 ± 4.336 | 39.380 ± 6.074 | **62.760 ± 2.811** | – | **11.313** |
| | Genie | 0.400 ± 0.894 | 54.765 (= base) | 63.697 ± 2.382 | – | 11.313 |

Bold = best among the trained methods (base and genie excluded). Eff, Gen ↓; Spe ↑; PPL ≈ base.
Spe: neighbourhood accuracy (MCF, ZsRE), retain Eff (MQuAKE), neighbour recovery (RWKU).
RWKU Gen = mean held-out recovery (Level 1, Level 2, Level-2 paraphrase); those probes ask facts
with no trained row, so RWKU genie Gen = base by construction.

## Forgetting vs locality

Share of base Eff removed / share of base Spe kept (from the means above):

| Method | MCF | ZsRE | MQuAKE | RWKU |
|---|---|---|---|---|
| GA | 99.8% / 1.8% | 40.1% / 88.1% | 89.1% / 19.2% | 59.7% / 62.1% |
| GD | 99.8% / 2.4% | 39.7% / 88.5% | 87.3% / 21.4% | 59.0% / 62.4% |
| NPO | 99.8% / 0.6% | 45.8% / 85.3% | 93.7% / 12.7% | 67.4% / 47.4% |
| RMU | 99.9% / 5.0% | 100.0% / 25.3% | 99.6% / 1.0% | 98.6% / 0.2% |
| ZeroUnlearn | 99.3% / 74.2% | 16.2% / 98.3% | 76.6% / 68.5% | 79.2% / 45.3% |
| **Ours** | **100.0% / 96.8%** | **99.1% / 100.0%** | **100.0% / 83.8%** (Spe† 98.6%) | **95.8% / 98.5%** |

## Reading the table

1. **Ours is the only method that forgets AND keeps locality on all four benchmarks**: ≥ 95.8%
   of base Eff removed and ≥ 96.8% of base Spe kept (MQuAKE: 83.8% raw, 98.6% on facts that are
   not also forget facts), with PPL exactly unchanged.
2. **Every weight-editing baseline trades locality for forgetting.** RMU forgets as much as ours
   (lower Gen on MCF, ZsRE, MQuAKE and RWKU) but keeps only 0.2–25% of Spe: it erases the
   relation/answer region, not the fact. GA/GD/NPO either collapse locality (MCF, MQuAKE) or,
   when stopped early enough to keep it, barely forget (ZsRE: 40–46% of Eff removed).
   ZeroUnlearn keeps the most locality among baselines but forgets least on ZsRE (16%) and loses
   26–55% of Spe on MCF, MQuAKE, RWKU.
3. **MCF/MQuAKE Spe structurally punishes global edits**: MCF neighbourhood prompts share the
   forget answer (other subjects, same answer token); a weight edit that suppresses the answer
   suppresses it there too. The router gate avoids this by never touching unrouted prompts.
4. **ZsRE router fix worked**: ours Gen 12.48 → 1.44 and Eff 2.34 → 0.30 vs round 1, Spe
   identical to base (the reworded router does not fire on neighbourhood prompts).
5. **MQuAKE Spe drop is the dataset overlap**: 14.9–18.1% of retain facts repeat a forget fact;
   on the rest, ours keeps 66.40 vs base 67.34 (−0.9).
6. **Gaps to genie are routing**: MCF Gen 0.051 vs 0.002 (one seed), ZsRE Gen 1.44 vs 0.00,
   MQuAKE Gen 7.20 vs 0.00 (seed 2: 15.2), RWKU Eff 2.4 vs 0.4.
7. **RWKU Gen is our weakest number** (39.4 vs base 54.8). Those held-out probes ask *other*
   facts about the same people, which a fact-level method does not target; the baselines that
   lower it further (RMU 0.3, ZeroUnlearn 9.9) do so by erasing the person entirely and lose
   55–100% of neighbour recovery.

## Diagnostics

| Dataset | Router (ours) | Oracle-trained rows (seeds 1–5) | Final on-policy leak | Genie direct routed | MQuAKE retain overlap |
|---|---|---|---|---|---|
| MCF | factorized v2.2 | 0, 0, 0, 0, 0 | 0.0, 0.3, 1.3, 1.7, 0.0 % | – | – |
| ZsRE | linear L19, reworded_v2 | 0, 2, 0, 0, 0 | 0.3, 0.0, 0.0, 1.7, 0.0 % | 100% | – |
| MQuAKE | linear L19 | 0, 0, 0, 0, 0 | 0.0, 0.0, 0.0, 1.4, 0.0 % | 100% | 15.1, 14.9, 16.9, 17.7, 18.1 % |
| RWKU | linear L19 (threshold) | 7, 5, 6, 6, 12 | 0.0, 0.0, 0.0, 2.0, 2.0 % | 100% | – |

## Caveats / open items

- Baselines use one operating point each (early stop at forget-view answer probability ≤ 1e-4;
  RMU with WMDP α = 1200 and steer = 4 × median ‖h₇‖). A forget–locality frontier per baseline
  (checkpoints along training) would make the comparison robust to that choice.
- ZsRE GA/GD/NPO reach the stop criterion (joint answer probability) while token-level Eff stays
  ~18–19%: the criterion is on the full answer, the metric scores each token under the gold
  prefix.
- Lens evaluation (answer unreadable inside the model) exists for MCF seed 1 only; the
  router + canned-reply baseline (C_idk) is not yet in this table.
