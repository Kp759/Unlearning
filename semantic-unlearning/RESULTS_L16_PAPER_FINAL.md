# Results (final): base, baselines, ours (L16), genie — 4 benchmarks × 5 seeds

Model: Llama-3.2-3B-Instruct. **One router on every benchmark**: per-fact linear heads on the
layer-19 boundary state behind exact subject-token eligibility (ZsRE: heads also trained on
base-model rewordings of each training question; official paraphrases never read). Δe rows
written at layer 16 (query mode). Supersedes the MCF rows of `RESULTS_L16_PAPER_ROUND2.md`,
which used the factorized v2.2 router for MCF (kept below as an ablation).

Outputs: base / baselines / ZsRE / MQuAKE / RWKU from `outputs/l16_paper_v1/<ds>/seed<S>/<method>/`;
MCF ours + genie from `outputs/l16_paper_mcf_linear19/mcf/seed<S>/`. Table:

```bash
python scripts/l16_paper_table.py --ours-from mcf=outputs/l16_paper_mcf_linear19
# -> outputs/l16_paper_v1/paper_table_mcf-l16_paper_mcf_linear19.{md,tex,json}
```
Code: `a5a47d7` (baseline early stopping), `ebe9a8b` (MCF linear-router run), `a97bcfd` (table).

## Main table (mean ± std over seeds 1–5)

| Dataset | Method | Eff ↓ | Gen ↓ | Spe ↑ | Spe† ↑ | PPL |
|---|---|---|---|---|---|---|
| **MCF** | Base | 13.456 ± 2.609 | 10.262 ± 3.496 | 19.880 ± 4.660 | – | 11.313 |
| | GA | 0.025 ± 0.027 | 0.277 ± 0.231 | 0.360 ± 0.358 | – | 30.820 ± 26.323 |
| | GD | 0.032 ± 0.032 | 0.321 ± 0.271 | 0.480 ± 0.415 | – | 31.003 ± 23.602 |
| | NPO | 0.026 ± 0.028 | 0.160 ± 0.191 | 0.120 ± 0.268 | – | 29.020 ± 12.080 |
| | RMU | 0.008 ± 0.005 | **0.039 ± 0.045** | 1.000 ± 0.707 | – | 11.501 ± 0.061 |
| | ZeroUnlearn | 0.094 ± 0.089 | 1.507 ± 0.699 | 14.760 ± 3.366 | – | 11.587 ± 0.110 |
| | **Ours (L16)** | **0.001 ± 0.000** | 0.086 ± 0.186 | **19.840 ± 4.672** | – | **11.313** |
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

Bold = best among trained methods (base and genie excluded; PPL: closest to base).
Spe: neighbourhood accuracy (MCF, ZsRE), retain Eff (MQuAKE), neighbour recovery (RWKU).
Spe†: MQuAKE retain Eff on retain facts that are not also forget facts (14.9–18.1% repeat one).
RWKU Gen = mean held-out recovery (Level 1, Level 2, Level-2 paraphrase); those probes ask
facts with no trained row, so RWKU genie Gen = base by construction. Baselines train on the
same training-visible forget views and are scored by the same evaluators; GA/GD/NPO/RMU stop
when the mean answer probability on the forget training views is ≤ 1e-4.

## Forgetting vs locality

Share of base Eff removed / share of base Spe kept:

| Method | MCF | ZsRE | MQuAKE | RWKU |
|---|---|---|---|---|
| GA | 99.8% / 1.8% | 40.1% / 88.1% | 89.1% / 19.2% | 59.7% / 62.1% |
| GD | 99.8% / 2.4% | 39.7% / 88.5% | 87.3% / 21.4% | 59.0% / 62.4% |
| NPO | 99.8% / 0.6% | 45.8% / 85.3% | 93.7% / 12.7% | 67.4% / 47.4% |
| RMU | 99.9% / 5.0% | 100.0% / 25.3% | 99.6% / 1.0% | 98.6% / 0.2% |
| ZeroUnlearn | 99.3% / 74.2% | 16.2% / 98.3% | 76.6% / 68.5% | 79.2% / 45.3% |
| **Ours** | **100.0% / 99.8%** | **99.1% / 100.0%** | **100.0% / 83.8%** (Spe† 98.6%) | **95.8% / 98.5%** |

**Ours removes ≥ 95.8% of base Eff and keeps ≥ 98.5% of base Spe on every benchmark**
(MQuAKE on facts that are not also forget facts; 83.8% raw), with PPL exactly base. Every
baseline loses ≥ 25% of Spe on at least one benchmark; RMU, the only one that forgets as
much (and has lower Gen), keeps 0.2–25% of Spe.

## Router ablation (MCF): linear L19 vs factorized v2.2

Same seeds, write layer, trainer, `--oracle-unrouted`, evaluator; only the router differs
(`scripts/compare_mcf_routers.py`). v2.2 = learned subject head (layer 6, MIL over
(token, next token) pairs) × relation softmax over MCF's 34 relation IDs (layers 19+23+27).

| MCF | Eff ↓ | Gen ↓ | Spe ↑ | PPL |
|---|---|---|---|---|
| Ours, v2.2 | 0.001 ± 0.000 | 0.051 ± 0.106 | 19.240 ± 4.894 | 11.313 |
| Ours, linear L19 | 0.001 ± 0.000 | 0.086 ± 0.186 | 19.840 ± 4.672 | 11.313 |
| Paired L19 − v2.2 | 0.000 | +0.036 ± 0.236 | **+0.600 ± 0.245** | 0.000 |
| Genie (either router's rows) | 0.001 | 0.002 | 19.880 | 11.313 |

| seed | v2.2 Gen / Spe | L19 Gen / Spe |
|---|---|---|
| 1 | 0.004 / 19.8 | 0.004 / 20.4 |
| 2 | 0.004 / 12.2 | 0.001 / 13.2 |
| 3 | 0.240 / 18.0 | 0.005 / 18.6 |
| 4 | 0.003 / 20.4 | 0.419 / 20.8 |
| 5 | 0.001 / 25.8 | 0.001 / 26.2 |

- Training is identical under both routers (same on-policy leak every seed, all 50 facts
  routed, no oracle-trained rows), so the difference is test-time routing.
- Linear L19 keeps more locality on 5/5 seeds (paired t ≈ 5.5, p ≈ 0.005; ≈ 3 of 500
  neighbourhood prompts per seed) and is ≈ base: exact subject matching cannot fire on a
  neighbour's subject, a learned subject head occasionally does. Gen is a tie within noise.
- v2.2's advantage — robustness to subject surface form (typos, case, unlisted aliases) —
  is not exercised by the official MCF prompts; report it with the robustness probes
  (`factorized_router.robustness_probes`) if needed. v2.2 also needs relation IDs, which
  ZsRE and RWKU do not have.

## Diagnostics

| Dataset | Router | Oracle-trained rows (seeds 1–5) | Final on-policy leak | Genie direct routed | MQuAKE retain overlap |
|---|---|---|---|---|---|
| MCF | linear L19 | 0, 0, 0, 0, 0 | 0.0, 0.3, 1.3, 1.7, 0.0 % | – | – |
| ZsRE | linear L19, reworded_v2 | 0, 2, 0, 0, 0 | 0.3, 0.0, 0.0, 1.7, 0.0 % | 100% | – |
| MQuAKE | linear L19 | 0, 0, 0, 0, 0 | 0.0, 0.0, 0.0, 1.4, 0.0 % | 100% | 15.1, 14.9, 16.9, 17.7, 18.1 % |
| RWKU | linear L19 (threshold) | 7, 5, 6, 6, 12 | 0.0, 0.0, 0.0, 2.0, 2.0 % | 100% | – |

## Reading the table

1. Ours is the only method that forgets and keeps locality on all four benchmarks.
2. Every weight-editing baseline trades locality for forgetting: RMU erases the
   relation/answer region (Spe 0.2–25% kept); GA/GD/NPO collapse locality (MCF, MQuAKE) or,
   stopped early enough to keep it, barely forget (ZsRE 40–46% of Eff removed); ZeroUnlearn
   keeps the most locality among baselines but forgets least on ZsRE (16%).
3. MCF and MQuAKE Spe structurally punish global edits: MCF neighbourhood prompts share the
   forget answer; a weight edit that suppresses the answer suppresses it there too. The
   router gate never touches unrouted prompts.
4. Remaining gaps to genie are routing: MCF Gen 0.086 vs 0.002 (seed 4: 0.419), ZsRE Gen
   1.44 vs 0.00, MQuAKE Gen 7.20 vs 0.00, RWKU Eff 2.4 vs 0.4.
5. RWKU Gen (39.4 vs base 54.8) is our weakest number: the held-out probes ask other facts
   about the same people, which a fact-level method does not target; RMU / ZeroUnlearn lower
   it by erasing the person and lose 55–100% of neighbour recovery.

## LaTeX

```latex
% \usepackage{booktabs,graphicx}
\begin{table*}[t]
\centering
\caption{Unlearning results on Llama-3.2-3B-Instruct (mean over 5 seeds). Eff, Gen $\downarrow$
(forgetting); Spe $\uparrow$ (locality); PPL should stay at Base (11.31). \textbf{Bold}: best trained
method (Base and Genie excluded; PPL: closest to Base). Ours uses one router on every benchmark
(per-fact linear heads on layer 19 behind exact subject matching) and writes $\Delta e$ at layer 16;
Genie: the same rows under oracle routing. On MQuAKE retain facts that are not also forget facts,
Spe is 66.40 (Ours) vs.\ 67.34 (Base).}
\label{tab:main}
\resizebox{\textwidth}{!}{%
\begin{tabular}{l cccc cccc cccc cccc}
\toprule
 & \multicolumn{4}{c}{MCF} & \multicolumn{4}{c}{ZsRE} & \multicolumn{4}{c}{MQuAKE} & \multicolumn{4}{c}{RWKU} \\
\cmidrule(lr){2-5} \cmidrule(lr){6-9} \cmidrule(lr){10-13} \cmidrule(lr){14-17}
Method & Eff$\downarrow$ & Gen$\downarrow$ & Spe$\uparrow$ & PPL
       & Eff$\downarrow$ & Gen$\downarrow$ & Spe$\uparrow$ & PPL
       & Eff$\downarrow$ & Gen$\downarrow$ & Spe$\uparrow$ & PPL
       & Eff$\downarrow$ & Gen$\downarrow$ & Spe$\uparrow$ & PPL \\
\midrule
\textit{Base} & 13.46 & 10.26 & 19.88 & 11.31 & 32.37 & 31.20 & 29.79 & 11.31 & 73.23 & 46.44 & 66.13 & 11.31 & 57.60 & 54.77 & 63.70 & 11.31 \\
\midrule
GA          & 0.025 & 0.277 & 0.36 & 30.82 & 19.38 & 19.07 & 26.24 & 12.67 & 7.98 & 10.80 & 12.68 & 16.12 & 23.20 & 30.55 & 39.57 & 11.69 \\
GD          & 0.032 & 0.321 & 0.48 & 31.00 & 19.52 & 19.40 & 26.36 & 12.91 & 9.30 & 11.92 & 14.18 & 16.48 & 23.60 & 30.46 & 39.72 & 11.80 \\
NPO         & 0.026 & 0.160 & 0.12 & 29.02 & 17.55 & 17.52 & 25.40 & 12.72 & 4.65 & 7.78 & 8.39 & 16.58 & 18.80 & 24.08 & 30.22 & 14.00 \\
RMU         & 0.008 & \textbf{0.039} & 1.00 & 11.50 & \textbf{0.00} & \textbf{0.05} & 7.55 & 11.49 & 0.30 & \textbf{6.89} & 0.67 & 11.48 & \textbf{0.80} & \textbf{0.27} & 0.14 & 11.43 \\
ZeroUnlearn & 0.094 & 1.507 & 14.76 & 11.59 & 27.14 & 26.59 & 29.28 & 11.55 & 17.14 & 20.64 & 45.27 & 11.34 & 12.00 & 9.92 & 28.84 & 11.58 \\
\midrule
Ours (L16)  & \textbf{0.001} & 0.086 & \textbf{19.84} & \textbf{11.31} & 0.30 & 1.44 & \textbf{29.79} & \textbf{11.31} & \textbf{0.00} & 7.20 & \textbf{55.43} & \textbf{11.31} & 2.40 & 39.38 & \textbf{62.76} & \textbf{11.31} \\
\midrule
\textit{Genie} & 0.001 & 0.002 & 19.88 & 11.31 & 0.00 & 0.00 & 29.79 & 11.31 & 0.00 & 0.00 & 56.21 & 11.31 & 0.40 & 54.77 & 63.70 & 11.31 \\
\bottomrule
\end{tabular}}
\end{table*}
```

## Open items

- Hybrid router smoke (learned subject head gating the linear heads; seed 1 × 4 benchmarks,
  gates `learned` and `learned_or_exact`; jobs 1432717 / 1432718): pending —
  `python scripts/compare_router_runs.py`.
- Llama-3.1-8B / Qwen3-4B tables: models not yet downloaded (`scripts/submit_paper_model.sh`).
- Baselines use one operating point each; a forget–locality frontier per baseline would make
  the comparison robust to that choice.
- Lens evaluation exists for MCF seed 1 only; router + canned-reply baseline (C_idk) not in
  the table.
