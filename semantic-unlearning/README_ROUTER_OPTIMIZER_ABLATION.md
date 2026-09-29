# Router optimizer ablation: SGD vs L-BFGS

Question: does fitting the linear-classifier router with minibatch SGD instead
of full-batch L-BFGS change the results? Seed 1, layer 19, regular SURE, on
MCF, ZsRE and MQuAKE.

```bash
sbatch optimizer_ablation_sgd.slurm            # 3 array tasks: mcf, zsre, mquake
# when done:
python scripts/compare_router_optimizers.py --collect outputs/optimizer_ablation_v1
cat outputs/optimizer_ablation_v1/optimizer_ablation_summary.md
```

Needs the finished reference runs `outputs/<dataset>_multiseed_regular_v1/seed1/L19/{prep,router,linear_global}`
(ZsRE reworded router: `--export=ALL,REF_TAG=multiseed_reworded_v2 --array=1`).

## What changes, what does not

Only `linear_router._fit_heads` (the stage-1 solver) is replaced;
`scripts/fit_linear_router_sgd.py` then runs `fit_linear_router.py` itself.

| | Reference | SGD run |
|---|---|---|
| objective | masked, class-balanced BCE + L2 (+1e-6 on b) | same, unbiased minibatch estimate (n/\|B\|)·Σ_B |
| solver | L-BFGS, full batch, strong Wolfe, float64 | `torch.optim.SGD`, batch 64, momentum 0.9, cosine decay, 300 epochs, float64, seed 0 |
| step size | line search | `1 / L`, L = worst-case minibatch smoothness bound (stable for every batch) |
| L2, PCA | grouped CV | **pinned** to the reference's CV choice (`--cv-grid full` reruns CV with SGD) |
| data, splits, negatives, calibration, bias folding, gate, rows init | — | identical (flags copied from the reference `linear_router_report.json` command) |

Change the SGD recipe with `SGD_ARGS`, e.g. vanilla SGD:
`sbatch --export=ALL,SGD_ARGS="--sgd-momentum 0 --sgd-epochs 600",ABL_TAG=optimizer_ablation_vanilla optimizer_ablation_sgd.slurm`.

## Stages (`scripts/run_optimizer_ablation_one.sh <dataset>`)

| dir | what | isolates |
|---|---|---|
| `router_sgd` | SGD router (+ in-process L-BFGS twin on the same features) | optimizer, classifier level |
| `router_lbfgs_rerun` | L-BFGS again through the same script | run-to-run noise (GPU query extraction) |
| `swap_sgd` | reference rows behind the SGD router, official eval | router effect at eval only |
| `full_sgd` | rows retrained under the SGD router, official eval | full pipeline |
| `full_lbfgs_rerun` | rows retrained under the rerun router, official eval | pipeline noise floor (`NOISE_FLOOR=0` skips it, ~half the time) |

Row training copies route, norm scale and time cap from the reference run's manifest.

## Reading `comparison.md`

1. **Stage-1 fit**: objective and max gradient per router. The twin line is the
   cleanest optimizer-only number (same features, same process): relative
   objective gap, weight cosine, norm ratio, sign flips on fit pairs.
2. **Parameters** in hidden space (PCA undone, so basis sign flips don't count).
3. **Routes** on every router prompt, per split: route changes, positives
   newly missed/correct, negatives newly firing. Compare with the rerun row.
4. **Training views** the row trainer could use under each router.
5. **Official metrics**: `identical` / `within noise` (|Δ| ≤ |rerun − reference|)
   / `differs`. The noise floor is one rerun, not a distribution.

Expected: with a well-regularized head (larger L2) SGD reaches the L-BFGS
optimum and routes are identical. With small L2 on near-separable data SGD
stops short of the optimum (smaller weight norm): the calibrated shift absorbs
most of it, but the fixed 0.5-logit ambiguity margin is scale-dependent, so a
few routes can move.
