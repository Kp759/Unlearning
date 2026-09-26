# Learned Linear Router (drop-in for Router V2)

`scripts/linear_router.py`, `scripts/fit_linear_router.py`, `tests/test_linear_router.py`

## What changes

Only the router's scorer. The frozen model, layer, request-boundary position,
subject eligibility, top-1 selection, residual rows and exact base path are
Router V2's own code and data.

| | Router V2 | Linear router |
|---|---|---|
| Score for association *i* | `max cos(q,P_i) − max cos(q,N_i)` | `z_i = w_i·φ(q) + b_i`, one output of `Linear(3072, N)` |
| Threshold | per-fact τ_i, in-sample (reduces to `−0.8(1−c_i)`) | none at runtime: the cutoff calibrated on a held-out split is folded into each bias, and the head fires at p ≥ 0.5 |
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
dimension (grid `1e-7…1` × `{none, 64, 256}`) are selected by grouped
cross-validation. Each fold holds out whole prompt families, and the score is
held-out class-balanced log-loss.

## Two-stage fit: the calibrated bias

Each head is a logistic regression, zᵢ = wᵢ·φ + bᵢ, and fires under the standard rule σ(zᵢ) ≥ 0.5. There is no separate threshold.

1. **Stage 1 (fit split):** wᵢ and bᵢ by masked, class-balanced BCE on the training templates.
2. **Stage 2 (calibration split, weights frozen):** the bias is re-fit on held-out prompts, bᵢ′ = bᵢ − t. Here t is the lowest cutoff that routes ≥ 98% of held-out forget prompts correctly with the fewest false activations. It is one value for all heads (`--threshold-policy global`) or one per head (`per_head`).

Why stage 2 is needed: the training templates are almost separable, so the stage-1 bias only has to put 0 somewhere in a wide gap between training positives and negatives. The class balancing also assumes half the prompts are positives. Unseen paraphrases land inside that gap. For example, MCF's calibrated shift is +3.78 logits, so bᵢ′ = bᵢ + 3.78.

**Equivalence with the earlier explicit threshold.** zᵢ ≥ t is the same as zᵢ − t ≥ 0.
- *Global:* every bias moves by the same amount, so every route is identical. The fit reports `fold_route_changes_vs_explicit_cutoff` (expected 0), and the official MCF evaluator gives identical numbers on both forms.
- *Per head:* each head fires on exactly the same prompts, but when two heads qualify, the best one is ranked by its calibrated logit zᵢ − tᵢ. Re-evaluate once.

Commands:
- `fit_linear_router.py` uses this by default (`--decision-rule calibrated_bias`). `--decision-rule explicit_threshold` reproduces earlier runs.
- `scripts/fold_linear_router_bias.py --run-dir <old run> --output-dir <new>` folds an existing run without refitting.
- The artifact keeps the stage-1 bias and the shift under `bias_calibration`.

## Gates

| Benchmark | Default gate | Meaning |
|---|---|---|
| MCF, ZsRE, MQuAKE | `threshold` | Association-level. The best eligible head fires if its calibrated probability is ≥ 0.5; ambiguous top-1/top-2 (< 0.5 logit apart) abstains. |
| RWKU | `subject` | Entity-level. Every prompt naming a protected person fires; the heads choose which of that person's rows to inject. A threshold gate would learn to abstain on unseen questions about the same person, and those are RWKU's held-out forget probes. |

For RWKU, run both gates (`--gate threshold` and the default) and compare
held-out Level-2 recovery against neighbor locality.

## Data (training-visible only)

| Split | MCF | ZsRE / MQuAKE / RWKU | Used for |
|---|---|---|---|
| fit | train families (646 prompts) + donor group 0 | canonical prompts + context prefixes 0,1 + donor group 0 | weights, CV |
| calibration | development families `authored_0, authored_2` + donor group 1 | context prefix 2 + donor group 1 | the bias shift (stage 2) |
| audit | development families `authored_1, authored_3` + donor group 2 | context prefix 3 + donor group 2 | reported numbers only |

A negative control for fact *i* is either a real prompt of another fact with
the same subject (from every split), or a subject transplant (subject *i* put
into another fact's prompt).

**Transplants are split-matched.** A calibration or audit negative is built from
the donor's prompts in that same development family. So every held-out
negative is a minimal pair with a held-out positive: same unseen template, same
subject, different relation.

Transplants are also:

- type-checked;
- limited to 3 per donor, rotated across template families;
- skipped when (subject *i*, donor relation) is a protected pair;
- skipped when the donor relation is in the fact's answer group
  (`RELATION_ANSWER_GROUPS`, e.g. native language / language used for writing /
  official language). Those prompts can leak the same answer, so they are
  neither clean positives nor clean negatives. Ablate with `--no-answer-groups`.

The protected-pair rule fixes a V2 issue. **In the MCF seed-1 sample, fact 27 (P101) takes all 12 of its V2
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
- `--min-recall 0.98`: recall-first. Picks the lowest calibration false
  activation with correct-route recall ≥ 0.98. Use this when the benchmark
  scores forgetting and never shows same-subject prompts (MCF, ZsRE).
- `--target-fpr 0.0`: selectivity-first. Picks the most correct routes with
  calibration false activation ≤ target. Use this when same-subject retain
  prompts are scored (MQuAKE).
- The report's operating curve covers FPR 0–50% and recall 90–100%.
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
  - `audit_frontier`: both routers swept over all thresholds on the audit
    split (V2 by shifting every τᵢ together). Includes route AUC, recall at
    V2's false-activation rate, and false activation at V2's recall. Use it to
    compare routers at matched operating points only; it is not an operating
    point.
  - runtime parity (hook vs offline decisions; must be 0 mismatches)
  - neutral-prompt exact-base check
  - dataset diagnostics
- `linear_router_dataset.json`: every prompt with its split, owner, kind, donor
  relation, linear score, and both routers' decisions (to inspect the
  highest-scoring negatives)
- `association_examples.json`, copied. Nothing else from the source run is
  copied: the evaluators read only the manifest and the artifact, and copying
  V2's evaluation outputs would place V2 results in this directory.

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

## Threshold policies and the 2×2

One fit run can separate two questions:

1. Does the scorer matter (V2's cosine margin dᵢ vs the linear logit zᵢ)?
2. Does the calibration policy matter (one global threshold vs per-association thresholds)?

| | Global threshold | Per-head threshold |
|---|---|---|
| **Cosine dᵢ** (V2 prototypes, bank, rows) | `arms/cosine_global` | `arms/cosine_per_head` |
| **Linear zᵢ** | `arms/linear_global` | `arms/linear_per_head` |

- **Same data and rules for every arm.** All four use the same held-out calibration split and the same rules. The per-head rule is Router V2's own τ rule: tᵢ = hardest negative + 0.1·gap. The difference is that here it is applied to prompts the scorer never saw.
- **Cosine arms change only τᵢ.** They keep V2's prototypes, runtime bank and residual rows, and load with the unchanged evaluators.
- **Shipped V2 is a fifth reference row** (`v2_shipped_in_sample_tau`). Its τᵢ come from the prototypes' own prompts.

```bash
python -u scripts/fit_linear_router.py --run-dir <V2 run> --output-dir <new> \
  --min-recall 0.98 --threshold-placement-fraction 0.1 \
  --threshold-policy per_head --emit-2x2 --local-files-only
# then run the benchmark's official evaluator on each <new>/arms/<arm>
```

**Per-head rule for head i** (on calibration):

- **Separable:** tᵢ = max(negatives) + f·(min(positives) − max(negatives)). Set f with `--per-head-fraction` (default 0.1).
- **Non-separable, or no negatives:** tᵢ = min(positives) − slack. Set slack with `--per-head-slack` (0.5 logits for the linear scorer; 0.02 for cosine, as in V2).
- **No positives:** fall back to the global threshold.
- **Shrinkage:** `--per-head-shrink κ` pulls tᵢ toward the global threshold with weight mᵢ/(mᵢ+κ), where mᵢ is the number of calibration prompts for head i.

Where the global threshold sits within its admissible gap is set by `--threshold-placement-fraction`: 0.5 is the midpoint, 0.1 is V2's placement.

**Other flags:**

- `--fit-device` runs the cross-validation fits on a GPU (it defaults to `--device`).
- The fit prints one progress line per cross-validation cell.
