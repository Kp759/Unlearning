# Paper table: base, baselines, ours, genie — seeds 1-5

Eff, Gen ↓ (forgetting) · Spe ↑ (locality) · PPL ≈ base. mean ± std over seeds (n per row). Spe† = MQuAKE retain Eff on retain facts that are not also forget facts. RR = representation recoverability: worst-block tuned-lens top-1 (%) of the forgotten answer read from the residual stream (lower is better; – = stage 6 not run).

| Dataset | Method | n | Eff ↓ | Gen ↓ | Spe ↑ | Spe† ↑ | PPL | RR ↓ |
|---|---|---|---|---|---|---|---|---|
| MCF | Base | 5 | 17.089 ± 2.750 | 13.007 ± 4.123 | 22.480 ± 4.653 | – | 11.033 ± 0.000 | – |
| MCF | Ours | 5 | 0.000 ± 0.000 | 0.353 ± 0.363 | 22.440 ± 4.657 | – | 11.033 ± 0.000 | – |
| MCF | Genie | 5 | 0.000 ± 0.000 | 0.002 ± 0.002 | 22.480 ± 4.653 | – | 11.033 ± 0.000 | – |
| ZsRE | Base | 5 | 39.807 ± 1.839 | 36.436 ± 2.584 | 30.565 ± 2.481 | – | 11.033 ± 0.000 | – |
| ZsRE | Ours | 5 | 0.133 ± 0.298 | 4.093 ± 2.883 | 30.565 ± 2.481 | – | 11.033 ± 0.000 | – |
| ZsRE | Genie | 5 | 0.133 ± 0.298 | 0.133 ± 0.298 | 30.565 ± 2.481 | – | 11.033 ± 0.000 | – |
| MQuAKE | Base | 5 | 72.817 ± 3.315 | 63.482 ± 2.431 | 67.757 ± 0.413 | 70.098 ± 0.507 | 11.033 ± 0.000 | – |
| MQuAKE | Ours | 5 | 1.317 ± 1.069 | 4.940 ± 1.864 | 58.490 ± 1.519 | 69.803 ± 0.633 | 11.033 ± 0.000 | – |
| MQuAKE | Genie | 5 | 1.317 ± 1.069 | 1.264 ± 1.000 | 58.733 ± 1.380 | 70.098 ± 0.507 | 11.033 ± 0.000 | – |
| RWKU | Base | 5 | 78.400 ± 2.608 | 83.983 ± 2.541 | 84.711 ± 2.458 | – | 11.033 ± 0.000 | – |
| RWKU | Ours | 5 | 3.600 ± 5.899 | 43.938 ± 13.592 | 82.441 ± 4.256 | – | 11.033 ± 0.000 | – |
| RWKU | Genie | 5 | 1.200 ± 1.095 | 83.983 ± 2.541 | 84.711 ± 2.458 | – | 11.033 ± 0.000 | – |
| RWKU | Genie (entity) | 5 | 1.200 ± 1.095 | 50.288 ± 12.986 | 84.711 ± 2.458 | – | 11.033 ± 0.000 | – |

Spe is neighbourhood accuracy (MCF, ZsRE), retain Eff (MQuAKE) and neighbour recovery (RWKU). RWKU Gen = mean of held-out Level 1, Level 2 and Level-2 paraphrase recovery; those probes ask facts with no trained row, so RWKU genie Gen equals base by construction. Baselines (GA, GD, NPO, RMU, ZeroUnlearn) train on the same training-visible forget views as ours and are scored by the same evaluators. RR (scripts/representation_recoverability.py): an affine lens fitted on retain text under the evaluated system reads the forgotten answer's first token from the residual stream after each block; RR is the worst block, over all official forget prompts (Gao et al. 2026, arXiv:2604.08271).

## Diagnostics

| dataset | seed | router (ours) | oracle-trained rows | final on-policy leak | genie: direct prompts routed | MQuAKE retain overlap | genie table / hits / lookups |
|---|---|---|---|---|---|---|---|
| MCF | seed1 | linear_L22 | 0 | 2.3% | – | – | 150 / – / – |
| MCF | seed2 | linear_L22 | 0 | 2.3% | – | – | 150 / – / – |
| MCF | seed3 | linear_L22 | 0 | 4.0% | – | – | 150 / – / – |
| MCF | seed4 | linear_L22 | 0 | 2.0% | – | – | 150 / – / – |
| MCF | seed5 | linear_L22 | 0 | 1.3% | – | – | 150 / – / – |
| MQuAKE | seed1 | linear_L22 | 0 | 0.0% | 100% | 15.1% | 210 / – / – |
| MQuAKE | seed2 | linear_L22 | 0 | 0.9% | 100% | 14.9% | 216 / – / – |
| MQuAKE | seed3 | linear_L22 | 0 | 0.9% | 100% | 16.9% | 216 / – / – |
| MQuAKE | seed4 | linear_L22 | 0 | 0.0% | 100% | 17.7% | 176 / – / – |
| MQuAKE | seed5 | linear_L22 | 0 | 0.0% | 100% | 18.1% | 216 / – / – |
| RWKU | seed1 | linear_L22 | 5 | 0.7% | 100% | – | 50 / – / – |
| RWKU | seed2 | linear_L22 | 3 | 2.0% | 100% | – | 50 / – / – |
| RWKU | seed3 | linear_L22 | 2 | 0.0% | 100% | – | 50 / – / – |
| RWKU | seed4 | linear_L22 | 4 | 2.0% | 100% | – | 50 / – / – |
| RWKU | seed5 | linear_L22 | 9 | 0.0% | 100% | – | 50 / – / – |
| ZsRE | seed1 | linear_L22 | 0 | 0.0% | 100% | – | 100 / – / – |
| ZsRE | seed2 | linear_L22 | 1 | 0.0% | 100% | – | 100 / – / – |
| ZsRE | seed3 | linear_L22 | 0 | 0.0% | 100% | – | 100 / – / – |
| ZsRE | seed4 | linear_L22 | 0 | 1.7% | 100% | – | 100 / – / – |
| ZsRE | seed5 | linear_L22 | 0 | 0.0% | 100% | – | 100 / – / – |
