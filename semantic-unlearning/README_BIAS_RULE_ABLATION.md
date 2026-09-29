# Bias-rule ablation: plain logistic regression vs calibrated bias

Question: fire at the standard logistic rule p ≥ 0.5 with the stage-1 bias
(one-stage fit), or at p ≥ 0.5 after folding the held-out calibrated cutoff
into the bias (shipped two-stage fit)? Seed 1, layer 19, regular SURE, MCF /
ZsRE / MQuAKE, for both router optimizers: 2 jobs × 3 benchmarks = 6 runs.

```bash
sbatch bias_rule_ablation_lbfgs.slurm
sbatch bias_rule_ablation_sgd.slurm     # or --dependency=afterany:<optimizer_ablation job> to reuse its SGD arm
# when both finish:
python scripts/compare_bias_rules.py --collect outputs/bias_rule_ablation_v1
cat outputs/bias_rule_ablation_v1/bias_rule_summary.md
```

## The two rules (same heads)

| rule | bias | fires when | calibration |
|---|---|---|---|
| **folded** (shipped) | b′ = b − t | w·φ + b′ ≥ 0 ⇔ z ≥ t | t from held-out calibration prompts |
| **raw** | b (stage 1) | w·φ + b ≥ 0 ⇔ z ≥ 0 | none |

With t < 0, a positive at z = −1.2 misses under raw and fires under folded;
a negative at z = −1.5 does the same. The report counts both.

## Arms per run (`outputs/bias_rule_ablation_v1/<optimizer>/<dataset>/seed1/L19/`)

| arm | what |
|---|---|
| folded | L-BFGS: the reference run itself. SGD: `optimizer_ablation_v1` router_sgd/full_sgd when finished (same reference, no SGD_ARGS), otherwise fit + trained here |
| `router_raw` | same weights, stage-1 bias, cutoff 0 (`make_raw_logistic_router.py`); both rules scored on every router prompt with the same features, plus runtime parity |
| `raw_swap` | folded rows behind the raw router + official eval: effect of the rule at eval only |
| `raw` | rows retrained under the raw router + official eval: the full one-stage method |

In regular mode the rows are trained only on views the router sends to their
own row, so `raw` can differ from `raw_swap` (the report lists views the
trainer could not use).

## Reading `comparison.md`

- Router table per split (fit / calibration / audit): correct route and false
  activation under both rules, positives rescued by calibration, positives
  whose own logit is below 0 and in [t, 0), false fires added by calibration.
  Use the **audit** split.
- Official metrics with the better rule per metric (forget Eff/Gen/AtomicGen
  and PPL lower; Spe and retain higher).
