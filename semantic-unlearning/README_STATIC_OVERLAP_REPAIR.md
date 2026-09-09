# Static overlap pilot repair

The September 8 pilot is not evidence of successful forgetting. Its +0.010458
mean forget NLL change is small, and 10 accepted steps only mean that the old
retention checks passed. Re-score the base and edited models with the same
metric implementation before comparing them.

## Matched-base result and active retention projection

The recovered 25-step run failed the forgetting target: Eff changed from
12.205893% to 11.955799%, Gen from 7.786824% to 7.735833%, and released answer
accuracy stayed at 22% / 16%. Successful FP32 export established checkpoint
parity and finite-anchor retention, not forgetting. Re-exporting those saved
factors cannot improve these scores.

The trainer had a mismatch between its projection and acceptance checks. It
projected against eight rotating protected examples but validated proposals
against every training anchor. If an omitted anchor was tight, backtracking
could shrink an otherwise useful update repeatedly. The observed tiny steps
and near-boundary NLL are consistent with this mechanism; the old report does
not identify the responsible anchors, so its contribution to that run cannot
be quantified from the supplied results alone.

`active_retention_projection_v2` includes all anchors near either budget in
addition to rotating coverage. After an actual retention violation, it adds the
worst omitted anchors' NLL/KL gradients and reprojects the same optimizer
proposal. Gradients are computed after restoring the pre-step parameters.
There is no extra Adam update. At most `max_constraint_refinements` (default 4)
rounds add `protected_batch_size` (default 8) anchors each. All near-limit
anchors are included initially without that cap. Nonlinear checks and
backtracking still apply, and every accepted step must pass all training
anchors at the original 0.05 NLL / 0.01 KL budgets.

History now records `active_anchor_ids`, `projected_anchor_ids`,
`discovered_anchor_ids`, `encountered_violating_anchor_ids`,
`constraint_refinements`, `nonlinear_checks`, and `retention_rejections`.
Retention diagnostics identify the maximum-NLL and maximum-KL anchors.
`training_protection` reports final training-anchor maxima separately from
validation. These fields show whether omitted constraints caused rejections,
whether projection repaired them, or whether another bottleneck remains.
The metric definitions, target probability, model capacity and learning rate
are unchanged.

This changes optimization and needs a fresh training pilot from the original
base, in a new output directory. Preserve the recovered checkpoint and its
evaluation for comparison. Use the training/evaluation commands below, with
`--require-zero`; do not infer MCF success from local regression tests. Official
paraphrases remain held out. The small synthetic suppression test verifies
the optimizer, not Llama generalization or feasibility of 0/0 on the 50 facts.

## Recover an export failure without retraining

The training runner saves `training_factors.pt`, `training_report.json`, and
`manifest.json` before export. A failure after step 25 does not require repeating
those steps. Keep `REPAIR_OUT` pointing to that completed run and recover into a
new directory:

```bash
git pull --ff-only origin feat/static-overlap-constrained-editing
python scripts/export_static_overlap_edit.py \
  --training-run "$REPAIR_OUT" \
  --model-path "$MODEL_PATH" --training-bundle "$TRAIN_BUNDLE" \
  --deployment-dtype float32 --device cuda --local-files-only
```

The checkpoint is written to `$REPAIR_OUT/checkpoint_float32`. Use that path for
`--checkpoint` in the evaluation commands below. Recovery checks the exact
bundle hash, editable indices and factor shapes, then reproduces the saved
base/edited NLL and KL statistics before merging. It uses the original model
and tokenizer and performs zero optimizer steps. Existing checkpoint directories
and training files are not overwritten; specify a new `--output-dir` for another
attempt. The numerical reproduction check does not retroactively establish a
cryptographic identity for an older run's unhashed base weights.

Export now verifies merging in the training dtype first, casting second, and
reload last. Each failed check writes `static_edit_export_failure.json` with its
stage, example, error magnitude and failing-logit count. The success marker is
absent on failure. Logit parity tolerances and nominal retention budgets remain
unchanged.

FP32 export verification allows an explicit absolute numerical slack of `5e-6`
for NLL increase and KL, after merge and after reload. Training constraints and
the factor-recovery checks remain strict; other model dtypes receive no slack.
The decision uses the unrounded observed value <= nominal budget + slack.
For example, NLL increase `0.05000114440917969` with KL
`9.294498158851638e-05` is a `numerical_boundary_pass` under nominal budgets
`0.05` and `0.01`. An excess greater than `5e-6` still fails.

Every export stage's `protection` report preserves the original maxima and adds
`observed_max_retained_nll_increase`, `observed_max_retained_kl`,
`nominal_retain_nll_budget`, `nominal_retain_kl_budget`, `numerical_slack`,
`nominal_budgets_passed`, `passed_with_numerical_slack`, and `classification`.
`passed_with_numerical_slack` is true only if the check needed the allowance.
This records numerical acceptance explicitly, rather than claiming the raw
value met the nominal budget.

If `checkpoint_float32` already exists from the boundary failure, preserve it
and rerun export into a new directory:

```bash
python scripts/export_static_overlap_edit.py \
  --training-run "$REPAIR_OUT" \
  --model-path "$MODEL_PATH" --training-bundle "$TRAIN_BUNDLE" \
  --output-dir "$REPAIR_OUT/checkpoint_float32_numeric" \
  --deployment-dtype float32 --device cuda --local-files-only
```

After verification succeeds, use `$REPAIR_OUT/checkpoint_float32_numeric` as
`--checkpoint` for the matched-base evaluation. No training rerun is needed.

Float32 is the default export dtype because converting the entire model to
bfloat16 can change logits independently of the edit and can round small weight
updates away. Bfloat16 export remains available when its verification passes.
Float32 uses more storage and inference memory. The old combined failure cannot
establish whether merging or casting caused the discrepancy; the staged check
now distinguishes them.

Recovered export does not imply successful unlearning. Evaluate the saved run
against its base. Future training logs also include the proposal radius,
backtrack count, update direction, forgetting progress and retention maxima.

## Verified issues and changes

- `evaluate_static_overlap_edit.py` called the legacy CounterFact summary,
  bypassing the probability helper already present in
  `feat/retain-anchored-context-quotient-head`. Its 84/86 Eff/Gen were sensitive
  answer preference percentages, not the paper's answer probabilities.
- The earlier helper computes a geometric mean of token probabilities. The new
  static evaluator reports full answer probabilities as Eff/Gen and preserves
  the geometric mean, released-code accuracy, and pairwise preferences under
  separate names. Old consumers of the shared evaluator retain an explicitly
  versioned legacy contract; their margin thresholds are not silently changed.
- Fast-tokenizer scoring now selects actual answer tokens by character offsets.
  It no longer strips a token merely because the model is Llama-like. Tests
  cover BOS present/absent, left/right padding, and one/multiple answer tokens.
- The old forget hinge stopped at base NLL + 2, which only reduces geometric
  token probability by a factor of `exp(-2)`. Its new target is
  `max(base_NLL + 2, -log(1e-6))` for each fitting example.
- Forget batches cycle without replacement and use the matching abstention
  examples. Ten batches of four still cannot cover 100 forget examples; one
  complete pass needs 25 steps in that case. Coverage is recorded in the report.
- The initial global update radius is now 0.25 and can grow to 2.0 after accepted
  steps. Learning rate is 0.005. These are candidate settings for a new pilot,
  not validated MCF hyperparameters.
- A step must improve the sampled forget objective as well as satisfy all
  training-anchor NLL/KL budgets. If the joint Adam direction harms forgetting,
  a forget-gradient fallback is projected and checked under the same budgets.
  Rejected updates restore parameters and optimizer state; fallback updates
  discard the unused Adam moment update.
- Projection now includes both NLL and KL gradients and their remaining budget
  allowances. NLL-only projection could stall at the KL boundary even when a
  useful feasible direction existed. The hard budgets remain NLL increase <=
  0.05 and KL <= 0.01, measured against the original model.
- Export verification reports forgetting separately from native checkpoint
  parity/retention. It does not label a successfully exported pilot as unlearned.

Official forget paraphrases and neighborhood probes remain excluded from
training. Held-out Gen=0 cannot be guaranteed by these changes. If model capacity
and retention constraints prevent suppression, the run must report failure.

## Metric definitions

[ZeroUnlearn, section 6.1, Eq. (16)](https://arxiv.org/html/2605.18879#S6.SS1)
defines efficacy as residual original-answer likelihood. Generalization applies
the same measure to paraphrases. For a multi-token answer `y`, this implementation
uses `P(y|x) = exp(sum_t log P(y_t|x,y_<t))`, without appending an EOS token.

| JSON field | Definition, in percent |
| --- | --- |
| `Eff` | `100 * mean_case(P(target_true | rewrite))` |
| `Gen` | `100 * mean_case(mean_paraphrase(P(target_true | paraphrase)))` |
| `TokenGeometricMean_Eff/Gen` | Earlier branch's `100 * mean_case(mean_prompt(exp(-mean_token_NLL)))` |
| `ReleasedAccuracy_Eff/Gen` | All original answer tokens argmax-correct, averaged by prompt then case |
| `Spe` / `ReleasedAccuracy_Spe` | The same correctness statistic on neighborhood prompts |
| `SensitivePref_Eff/Gen` | Legacy fraction preferring target_true over target_new |
| `CF_EditSuccess_Eff/Gen` | Fraction preferring target_new over target_true |

The [released ZeroUnlearn evaluator](../ZeroUnlearn/experiments/py/eval_utils_counterfact.py)
also records teacher-forced correctness, aggregated by its
[summary code](../ZeroUnlearn/experiments/summarize_list.py). This is a distinct
metric from Eq. (16). For unequal paraphrase counts, cases retain equal weight;
ordinary MCF records have two paraphrases.

Raw probabilities are not rounded to zero. Finite softmax logits have positive
probability mathematically. `--require-zero` checks both Eff/Gen < 0.005 percent
(displayed as 0.00 at two decimals) AND both released accuracies exactly zero.
It saves JSON before exiting nonzero on failure. This is a forgetting check,
not a guarantee of overall utility or absence of disclosure in long generations.
The existing counterfactual margin is retained as a diagnostic, not substituted
for answer probability. `--max-new-tokens 1` affects only free generation and
does not affect any of these teacher-forced MCF metrics.

Old raw JSON lacks answer token counts, so full sequence likelihood cannot be
reconstructed exactly from it. Re-run inference to obtain the new metrics; do
not compare 84/86 legacy preferences directly to the new Eff/Gen.

## Re-score the existing pilot against its base

Run from `semantic-unlearning` after applying the code changes on EC2:

```bash
set -o pipefail
export PILOT_OUT="$PWD/outputs/static_overlap_mcf_seed1_pilot_20260908_221239"
export MODEL_PATH="$(python -c 'import json,os; print(json.load(open(os.path.join(os.environ["PILOT_OUT"], "manifest.json")))["model_path"])')"

python scripts/evaluate_static_overlap_edit.py \
  --checkpoint "$PILOT_OUT/checkpoint" \
  --base-model "$MODEL_PATH" \
  --evaluation-bundle data/static_overlap_mcf_seed1_eval.json \
  --out "$PILOT_OUT/evaluation_probability_v2.json" \
  --device cuda --max-new-tokens 1 \
  --mcf-path data/multi_counterfact.json --wikidata-dir data/wikidata \
  --seed 1 --unlearn-num 50 --retain-num 1000 --skip-official-ppl \
  2>&1 | tee static_overlap_mcf_seed1_probability_v2.log
```

Inspect `official_mcf.forget`, `base.official_mcf.forget`,
`change_vs_base`, and `forgetting_check` in the saved report. Base inference
uses the same deployment dtype, tokenizer pipeline, and sampled records.

## Run a fresh coverage pilot

Use the original training bundle; adjust `TRAIN_BUNDLE` if its filename differs.
The output must be a new directory, and training starts from the original model.
For the reported 100 forget examples and batch size four, 25 steps cover one pass.

```bash
export TRAIN_BUNDLE="data/static_overlap_mcf_seed1_train.json"
export REPAIR_OUT="$PWD/outputs/static_overlap_mcf_seed1_repair_$(date +%Y%m%d_%H%M%S)"
python scripts/run_static_overlap_edit.py \
  --model-path "$MODEL_PATH" --training-bundle "$TRAIN_BUNDLE" \
  --output-dir "$REPAIR_OUT" --config config/static_overlap_edit.json \
  --steps 25 --device cuda --dtype float32 --deployment-dtype float32 \
  2>&1 | tee static_overlap_mcf_seed1_repair_train.log

python scripts/evaluate_static_overlap_edit.py \
  --checkpoint "$REPAIR_OUT/checkpoint" --base-model "$MODEL_PATH" \
  --evaluation-bundle data/static_overlap_mcf_seed1_eval.json \
  --out "$REPAIR_OUT/evaluation_probability_v2.json" \
  --device cuda --max-new-tokens 1 \
  --mcf-path data/multi_counterfact.json --wikidata-dir data/wikidata \
  --seed 1 --unlearn-num 50 --retain-num 1000 --skip-official-ppl \
  --require-zero 2>&1 | tee static_overlap_mcf_seed1_repair_eval.log
```

A nonzero final exit means the goal is unmet; inspect the saved diagnostics.
Training acceptance alone does not justify a 200-step launch. Check matched-base
retention and neighborhood accuracy too. A one-token generation probe cannot
assess full answers, mixed requests, or disclosure; use the normal generation
length for that assessment and a separate held-out corpus for final PPL.

## Local verification

Regression tests cover exact probability aggregation, token alignment, raw versus
rounded success decisions, original-model comparisons, actual CLI failure after
saving results, projection against independent constraints, optimizer rollback,
and float32/bfloat16 native merge/reload.

A synthetic model with two contexts sharing an answer is also trained to the
1e-6 suppression target while preserving its retained association and language
anchor within the same NLL/KL budgets. That test verifies optimization behavior;
it is not an MCF or Llama result.

```bash
python -m pytest -q \
  tests/test_static_overlap_active_constraints.py \
  tests/test_static_overlap_edit.py \
  tests/test_static_overlap_export_slack.py \
  tests/test_static_overlap_probability_metrics.py \
  tests/test_static_overlap_mcf_language.py \
  tests/test_mcf_zero_unlearn_metric_parity.py \
  tests/test_mcf_zero_unlearn_released_table_accuracy.py \
  tests/test_gagd_active_case_repair.py
```
