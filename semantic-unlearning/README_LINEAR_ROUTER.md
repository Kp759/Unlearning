# Learned Linear Router (drop-in for Router V2)

`scripts/linear_router.py`, `scripts/fit_linear_router.py`, `tests/test_linear_router.py`

## What changes

Only the router's scorer. The frozen model, layer, request-boundary position,
subject eligibility, top-1 selection, residual rows and exact base path are
Router V2's own code and data.

| | Router V2 | Linear router |
|---|---|---|
| Score for association *i* | `max cos(q,P_i) − max cos(q,N_i)` | `z_i = w_i·φ(q) + b_i`, one output of `Linear(3072, N)` |
| Threshold | per-fact τ_i, in-sample (reduces to `−0.8(1−c_i)`) | one global logit threshold, set on a held-out calibration split |
| Stored for routing (MCF) | ~1,246 prototype vectors ≈ 15 MB | `W,b` ≈ 0.6 MB (+ PCA basis if selected) |
| Probability / ROC / operating point | none | yes |
| Route | deterministic | deterministic (hard gate; inactive path bit-exact) |

`φ(q) = normalize(q) − μ`, optionally projected onto a PCA basis.

**N independent binary classifiers.** The loss is a sum over heads of a
masked, per-head-normalised, class-balanced BCE, with L2 on each row of `W`.
The objective separates by head, so the joint fit gives the same weights as
fitting N logistic regressions one at a time. This is tested.

**Subject-masked loss.** Head *i* is trained only on prompts that contain
subject *i*, because those are the only prompts it scores at runtime. Its
negatives are therefore same-subject / different-relation prompts. That is the
boundary MQuAKE's same-subject retain set (V2: 63/81 fires) and RWKU's
same-person rows depend on.

**Hyperparameters are chosen by the data, per run.** The L2 strength and PCA
dimension (grid `1e-5…1` × `{none, 64, 256}`) are selected by grouped
cross-validation. Each fold holds out whole prompt families, and the score is
held-out class-balanced log-loss.

## Gates

| Benchmark | Default gate | Meaning |
|---|---|---|
| MCF, ZsRE, MQuAKE | `threshold` | Association-level. The best eligible head fires if its logit clears the global threshold; ambiguous top-1/top-2 (< 0.5 logit) abstains. |
| RWKU | `subject` | Entity-level. Every prompt naming a protected person fires; the heads choose which of that person's rows to inject. A threshold gate would learn to abstain on unseen questions about the same person, and those are RWKU's held-out forget probes. |

For RWKU, run both gates (`--gate threshold` and the default) and compare
held-out Level-2 recovery against neighbor locality.

## Data (training-visible only)

| Split | MCF | ZsRE / MQuAKE / RWKU | Used for |
|---|---|---|---|
| fit | train families (646 prompts) + donor group 0 | canonical prompts + context prefixes 0,1 + donor group 0 | weights, CV |
| calibration | development families `authored_0, authored_2` + donor group 1 | context prefix 2 + donor group 1 | the global threshold |
| audit | development families `authored_1, authored_3` + donor group 2 | context prefix 3 + donor group 2 | reported numbers only |

A negative control for fact *i* is either a real prompt of another fact with
the same subject, or a subject transplant (subject *i* put into another fact's
prompt). Transplants are type-checked, limited to 3 per donor, and skipped when
(subject *i*, donor relation) is a protected pair.

That last rule fixes a V2 issue. **In the MCF seed-1 sample, fact 27 (P101) takes all 12 of its V2
"negatives" from a same-relation donor, so they are really paraphrases of its own positive.**
Official paraphrase, neighborhood, retain, utility and evaluation fields are
never read.

The ZsRE/MQuAKE/RWKU audit measures robustness to an unseen lead-in context,
not to paraphrase. The official evaluation is the generalization test for
every benchmark.

## Run

```bash
# 1. fit on an existing V2 run (reuses its residual rows; new dir must not exist)
python -u scripts/fit_linear_router.py \
  --run-dir  outputs/mcf_fact_assoc_router_v2_seed1 \
  --output-dir outputs/mcf_fact_assoc_linear_router_seed1

# 2. evaluate exactly as for V2; every evaluator dispatches on the artifact
python -u scripts/evaluate_static_overlap_fact_association_embeddings_official.py \
  --run-dir outputs/mcf_fact_assoc_linear_router_seed1 --mcf-path data/multi_counterfact.json
```

The same two steps apply to ZsRE, MQuAKE and RWKU with their own run dirs and
evaluators:

- `evaluate_zsre_fact_association_embeddings_official.py`
- `evaluate_mquake_fact_association_embeddings_official.py`
- `evaluate_rwku_fact_association_embeddings_seed1.py`

`evaluate_router_decomposition.py` also works: its `v2` arm becomes the linear
router, and the oracle, subject-only and random arms reuse the same rows.

Useful flags:

- `--gate {auto,threshold,subject}`
- `--target-fpr 0.0` (calibration false activation; the report includes the operating curve at 0/1/2/5%)
- `--ambiguity-margin 0.5`
- `--split-rule rebalanced` (MCF: relation-alternates become development)
- `--pca-dims`, `--lambdas`
- `--dtype` (for feature extraction; V2 fitting used float32)

## Outputs (new run dir)

- `fact_association_embeddings.pt`: architecture `linear_classifier_fact_association_bank_v1`
- `association_manifest.json`: the source manifest plus router metadata
- `linear_router_report.json`, containing:
  - the CV table
  - calibration and operating curve
  - route outcomes per split, with Wilson intervals
  - **V2 on the same prompts**
  - runtime parity (hook vs offline decisions; must be 0 mismatches)
  - neutral-prompt exact-base check
  - dataset diagnostics
- `linear_router_dataset.json`: every prompt with its split, owner, kind and donor relation
- the source run's other non-`.pt` files, copied

Report the **audit** split, not calibration: calibration numbers come from the
split that chose the threshold.

## Caveats

- Fitting extracts features in float32 while evaluators default to bf16.
  Prompts near the threshold can flip. V2 has the same mismatch.
- `target_fpr=0` sets the threshold from the hardest calibration control. Read
  the operating curve before freezing it.
- Freeze gate, threshold rule, margin and grids before confirmatory seeds.
- `router_fitting_data_v2.rebalanced_split` reads the family from `role`, but
  MCF examples store it in `group` (`role` is `"forget"`), so as shipped it moves
  nothing. `fit_linear_router.py --split-rule rebalanced` passes the family through.
