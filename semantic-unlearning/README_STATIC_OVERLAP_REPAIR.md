# Static overlap pilot repair

The September 8 pilot is not evidence of successful forgetting. Its +0.010458
mean forget NLL change is small, and 10 accepted steps only mean that the old
retention checks passed. Re-score the base and edited models with the same
metric implementation before comparing them.

## Runtime: cache the immutable base references

The replay pilot added 326 retention rows. Every nonlinear candidate check
previously made a base and an edited forward for every anchor, individually.
For roughly 489 retained-answer contexts, thirteen checks entail about 12,700
forwards before language anchors, gradient computation, all-forget scoring and
validation selection. A logged optimizer step therefore contains much more
work than a conventional minibatch update. Projection refinements/backtracking
repeat those checks, and full FP32 execution remains in use.

The replay preset now sets `base_cache_mb=1024`. A training-call-local cache
stores the original base NLL and FP32 full-vocabulary log-probabilities at the
labeled answer positions, in CPU RAM. Cache keys contain the actual input IDs
and labels. It never caches edited outputs, drops vocabulary entries, quantizes
references, changes anchors, changes precision, or relaxes an acceptance check.
References for retain/language examples are reused in candidate checks,
protection gradients, the KL objective and checkpoint selection. Input/label
changes miss the cache; LRU eviction and oversized entries fall back to original
base computation. The cache's tensor storage is bounded by the configured
limit (plus transient tensors and ordinary Python metadata). Set
`base_cache_mb=0` to disable it; existing non-replay presets retain that default.

With a warm cache that fits the references, retention checking needs one model
forward per anchor instead of two. This does **not** promise a twofold total
speedup: edited forwards, backward passes, host-device transfers, solver work,
and cache misses remain. Local tests require exact cached/uncached NLL, KL and
KL-gradient parity, correct input/label invalidation, bounded eviction, and
successful fresh training with caching both enabled and disabled. Actual EC2
wall-clock speedup has not been measured here.

Each step now reports `timing_seconds` and cumulative `base_reference_cache`
statistics. `nonlinear_checks_within_search` is a **subset** of
`candidate_search`; do not add both when attributing runtime. These use host
wall-clock timings, with the trainer's existing scalar reads synchronizing
model results; they are not dedicated GPU kernel timings. `step_wall` excludes
subsequent epoch-end diagnostics and final report measurement, while
`training_wall_seconds` includes the full `train()` call. Neither includes
earlier model loading/localization in the runner. A Python process already
running the older code is unaffected by fetching this change.

## Adaptive replay and retention context experiment

The fresh hard-example run made 25 accepted updates and covered all 100 fitting
forget examples. Before checkpoint restoration, mean token probability fell
from 6.6954% to 5.1017%, but maximum probability ended at 82.5591% versus
82.8683% at base. Validation retention failed at NLL increase 0.201105 and KL
0.012736. Only steps 1 and 2 were eligible; step 1 was restored because step 2's
worst fitting score regressed. This is partial average suppression, not a
near-zero forgetting result.

`config/static_overlap_replay.json` adds three opt-in changes while preserving
the two-layer/rank-8/64-channel architecture, the repaired projection solver,
detached capped weights, best-valid checkpoint selection, and a fresh start:

- **Adaptive replay:** four ordinary coverage examples plus up to two extra
  hard examples per step. Hardness is recomputed from all fitting forget NLLs
  after accepted steps. The largest target gap and largest residual probability
  receive priority, then distinct facts. Replay does not consume coverage slots
  or duplicate an example inside the batch.
- **Worst-target objective and acceptance:** add a separate hinge loss for
  the currently largest fitting target gap (`lambda_worst_forget=1`). Its
  gradient participates in both the mixed Adam and pure forget proposals. This
  term retains its scale even when capped batch weights normalize similarly.
  Candidates must still improve the full weighted fitting objective. They
  must also keep both the largest target gap and largest token probability
  nonincreasing over **all** fitting forget examples. There is no numerical
  allowance that can accumulate regression. This is a check on global extrema,
  not a guarantee that every individual fact improves at every step.
- **Solver support for the new check:** initially include the gradients of
  the two fitting extrema. For example `i`, linearize the NLL floor
  `max(target_i - current_max_gap, current_min_NLL)`. Add missing fitting
  constraints when nonlinear rechecks find another violating example. Actual
  nonlinear forgetting and retention checks remain authoritative. Among passing
  Adam/forget candidates, maximize weighted-average gap reduction plus
  `lambda_worst_forget * worst_gap_reduction`.

`scripts/augment_static_overlap_retention.py` prepares the retention contexts
for this experiment. It uses only original **training** retention rows and two
fixed instruction prefixes. Mixed companion answers are protected after both
true-answer and neutral/abstention completions. All added labels are retain
labels; the original forget supervision, facts, language anchors and validation
examples remain unchanged. It reads no official evaluation file or MCF probes.
Prompt collisions with validation and repeated augmentation are rejected.
The sidecar records source/output hashes, templates and every source training
row. This broadens context coverage; it does not guarantee validation retention.

The scientific validation limits remain **0.05 NLL / 0.01 KL**, with the same
internal training margins giving **0.04 / 0.008**. Validation is used only to
determine checkpoint eligibility; official Gen remains entirely held out.
Stricter worst-target checks can reveal a lack of useful feasible updates.
The trainer reports that failure instead of weakening its checks.

Run fresh on EC2. The augmented bundle has a new path and hash; do not resume
the prior factors or overwrite their input bundle:

```bash
cd /home/ec2-user/workspace/Unlearning-static-overlap/semantic-unlearning
export MODEL_PATH="/home/ec2-user/models/Llama-3.2-3B-Instruct"
export TRAIN_BUNDLE="$PWD/data/static_overlap_mcf_seed1_train.json"
export REPLAY_OUT="$PWD/outputs/static_overlap_replay_$(date +%Y%m%d_%H%M%S)"
export AUG_BUNDLE="$REPLAY_OUT.training.json"
set -o pipefail

python scripts/augment_static_overlap_retention.py \
  --training-bundle "$TRAIN_BUNDLE" --out "$AUG_BUNDLE" &&
python scripts/run_static_overlap_edit.py \
  --model-path "$MODEL_PATH" --training-bundle "$AUG_BUNDLE" \
  --output-dir "$REPLAY_OUT" --config config/static_overlap_replay.json \
  --steps 25 --device cuda --dtype float32 --local-files-only --training-only \
  2>&1 | tee "$REPLAY_OUT.log"
```

The new terminal summary includes `last_iterate`, so the last state cannot be
confused with the restored selected state. History adds `coverage_batch_ids`,
`replay_batch_ids`, `worst_gradient_id`, `projected_forget_ids`,
`worst_forget_before`, `worst_forget_after`, and `worst_forget_progress`.
The report's `forget_gradient_visits` counts coverage/replay and the separate
worst-target loss visits, including attempted steps. It excludes gradients used
only for projection constraints. Candidate scores in replay mode include the
worst-target reduction and are not the same as `global_forget_progress` alone.

Once the selected state meets the fitting suppression target and both retention
checks, recover a native FP32 checkpoint using the manifest's saved augmented
bundle path:

```bash
python scripts/export_static_overlap_edit.py \
  --training-run "$REPLAY_OUT" --model-path "$MODEL_PATH" \
  --deployment-dtype float32 --device cuda --local-files-only
```

Then use the unchanged official base-versus-edit evaluator on the original
evaluation bundle. A fitting target pass is not an official Gen pass.
Local tests exercise two-fact suppression below `1e-6`, monotonic worst fitting
probability, retention checks, disjoint augmentation, real tiny-Llama CLI
training, and selected-factor recovery. These tests do not establish Llama-3.2-3B
Eff/Gen results, and neither evaluation metrics nor inference behavior are
modified by this experiment.

## Fresh hard-example experiment with retention margins

The five-step solver continuation accepted all steps, but mean fitting token
probability changed only from 5.0648% to 5.0248%, and the worst example worsened
from 72.55% to 73.02%. Its final KL was 0.009999802, effectively at the training
limit. Validation retention remained invalid (NLL increase 0.211648, KL 0.021239).
These are training-bundle diagnostics, not official Eff/Gen.

The separate `config/static_overlap_hard_examples.json` preset starts from the
original base and retains the two-layer, rank-8, 64-channel architecture for a
controlled optimization comparison. It explicitly rejects continuation.
Existing presets retain their earlier behavior unless the new options are set.

The new mode implements:

- **Detached capped weights.** For each fact, use the maximum token probability
  across its fitting views, `P_f = max_view(exp(-mean_answer_NLL))`. With `m`
  equal to `hard_example_mix` and `C` to `hard_example_cap`, the raw fact weight
  is `1 + m * min(C - 1, P_f / mean_fact(P_f))`. Weights are computed from
  detached scalars once per optimizer step and held fixed for every candidate
  and final recheck. With `m=1, C=4`, raw fact weights lie in `[1,4]`. Each
  fact's weight is divided among its views, then losses are normalized by the
  included weights. The floor prevents low-probability facts from disappearing.
- **Coverage.** Each epoch orders the most remembered facts first, then visits
  every fitting view without replacement. Hard-example mode does not stop for
  consecutive rejected steps before trying every fitting example once, unless
  the requested step budget ends first.
- **Candidate comparison.** Mixed Adam and pure forget descent each search for
  their first feasible step. Feasible candidates must improve the same fixed,
  weighted hinge objective over **all training forget examples**, in addition
  to batch progress and every training retention constraint. The larger global
  improvement wins. The selected parameters are restored and checked again;
  Adam state is discarded if pure forget descent wins. This compares two
  searched directions, not every possible feasible update.
- **Internal margins.** `retain_nll_safety_margin=0.01` and
  `retain_kl_safety_margin=0.002` yield fitting limits of **0.04 NLL / 0.008 KL**.
  They apply to active-anchor selection, remaining projection allowances, and
  complete nonlinear training checks. Scientific validation limits stay
  **0.05 / 0.01**. Export retains its separately documented FP32 numerical
  allowance; the margins do not loosen validation or export checks.
- **Best valid checkpoint.** After each accepted step, check validation
  retention without gradients. Eligible states must pass both internal
  training limits and nominal validation limits. Among eligible states,
  minimize: largest remaining NLL gap to a fitting suppression target, then
  maximum fitting token probability, then mean fitting token probability. This
  score is consistent across steps and does not use the changing hard-example weights.
  Validation forget examples and official MCF Gen are never selection scores.

Every accepted state is saved under `accepted_checkpoints/step_XXXXXX.pt`, with
a matching JSON record. At completion, the best eligible state is restored and
its metrics are recomputed; `training_factors.pt` and the main report refer to
that state. If it differs from the last accepted state, the latter is preserved
as `last_training_factors.pt` with `last_training_statistics.json`. The main
report records `checkpoint_selection.selected_step` and `last_iterate`.
If no eligible state exists, `selected_step` is null and native export is refused;
the unedited base is never substituted as a successful result.

Run the next pilot fresh on EC2, with all paths explicit:

```bash
cd /home/ec2-user/workspace/Unlearning-static-overlap/semantic-unlearning
export MODEL_PATH="/home/ec2-user/models/Llama-3.2-3B-Instruct"
export TRAIN_BUNDLE="$PWD/data/static_overlap_mcf_seed1_train.json"
export HARD_OUT="$PWD/outputs/static_overlap_hard_$(date +%Y%m%d_%H%M%S)"
set -o pipefail

python scripts/run_static_overlap_edit.py \
  --model-path "$MODEL_PATH" --training-bundle "$TRAIN_BUNDLE" \
  --output-dir "$HARD_OUT" --config config/static_overlap_hard_examples.json \
  --steps 25 --device cuda --dtype float32 --local-files-only --training-only \
  2>&1 | tee "$HARD_OUT.log"
```

Inspect `initial_training_forgetting` (not `resume_before_training_forgetting`),
`training_forgetting`, `training_protection`, `validation_protection`, and
`checkpoint_selection` in `training_report.json`. History adds
`forget_batch_ids`, `forget_batch_weights`, `global_forget_progress`, and
`candidate_results`. `projection_violation` reports the chosen line-search
step's linear residual. `forget_progress` alone remains a minibatch statistic.

This is an opt-in experiment, not an established Llama 0/0 configuration.
Margins may help validation generalization but do not guarantee it. Official
Gen stays entirely held out from fitting, weighting, candidate comparison and
checkpoint selection. Only evaluate official Eff/Gen after a retention-valid
checkpoint is available. Local regression experiments verify capped gradients,
complete coverage, stronger feasible candidate selection, restoration of an
earlier valid checkpoint, and exact selected-factor recovery.

## Projection repair and continuation diagnostic

The active-projection run reported nine rejected steps, all ending with
`projection_converged=false` at 1,000 iterations. Its final training-anchor
retention passed, but validation retention failed at NLL increase 0.207899 and
KL 0.021255. It therefore produced no verified native checkpoint and no
evaluation JSON. Its worst training forget token probability was 0.725545;
neither those bundle statistics nor the rejected trial losses establish
official Eff/Gen success.

The solver now tries 100 Dykstra iterations, then solves the **same** constrained
projection using FP64 QR reduction and SLSQP (SciPy is already a dependency).
The solution lies in the span of the proposal and constraint normals, so the
reduction has at most `number_of_constraints + 1` variables. It does not drop
constraints or change their allowances. Solver success and the original
halfspace/radius checks must hold after casting back to the parameter dtype.
All nonlinear training retention checks still apply before accepting a step.
Three correlated-constraint regression cases that fail after 1,000 Dykstra
iterations even in FP64 converge in 10–16 reduced-solver iterations. This is a
solver test, not evidence of success on the EC2 model.

If both solvers fail after adding constraints, the last converged direction
may still be backtracked, provided it passes **all expanded linear constraints**
and the complete nonlinear checks. Failed solver iterates are never applied.
History includes `projection_attempts` with method, precision, convergence,
iterations and violation. Rejected trials' measurements are nested under
`last_rejected_trial`; their top-level `forget_progress` is zero and retention
maxima describe the unchanged model. A rejected trial with a large forgetting
loss improvement is not reported as applied progress.

Follow this order: fix the solver, test continuation, measure fitting forgetting,
then address validation retention if necessary, then evaluate official Eff/Gen.
Checkpoint selection, safety margins, hard-example weighting, and soft penalties
are not combined into this solver change. Validation examples still produce no
gradients. Scientific limits remain 0.05 NLL and 0.01 KL.

To test continuation from the failed run without repeating localization or
creating an invalid checkpoint, preserve `ACTIVE_OUT` and use a new directory:

```bash
cd /home/ec2-user/workspace/Unlearning-static-overlap/semantic-unlearning
export MODEL_PATH="/home/ec2-user/models/Llama-3.2-3B-Instruct"
export TRAIN_BUNDLE="data/static_overlap_mcf_seed1_train.json"
export ACTIVE_OUT="$PWD/outputs/static_overlap_active_20260909_030026"
export SOLVER_OUT="$PWD/outputs/static_overlap_solver_$(date +%Y%m%d_%H%M%S)"
set -o pipefail

python scripts/run_static_overlap_edit.py \
  --model-path "$MODEL_PATH" --training-bundle "$TRAIN_BUNDLE" \
  --resume-training-run "$ACTIVE_OUT" \
  --output-dir "$SOLVER_OUT" --config config/static_overlap_edit.json \
  --steps 5 --device cuda --dtype float32 --local-files-only --training-only \
  2>&1 | tee "$SOLVER_OUT.log"
```

These assignments are required again in a new shell session. An unset
`TRAIN_BUNDLE` expands to an empty string, which `Path("")` interprets as `.`.
The CLI now rejects empty path arguments before reading files or loading a
model, including an explicitly empty resume path. It also checks that the
bundle/config are files and that all three resume artifacts exist.

Continuation verifies the bundle hash, architecture, editable rows, dtype,
base configuration and reproduction of saved base/edited statistics before
fitting. The original base remains the reference for targets and retention;
budgets are not reset around the already edited model. Initial **training**
retention must pass. Validation failure may remain during this diagnostic.
Adam is explicitly reset because the earlier artifacts do not contain optimizer
state. Parent files remain unchanged, and parent hashes and accepted-step
counts are recorded in the new manifest/report.

`--training-only` saves `training_factors.pt`, `training_report.json`, and
`manifest.json`. A successful exit means the diagnostic completed, not that
forgetting or validation passed. No native checkpoint or evaluation JSON is
created. Compare `initial_training_forgetting` with `training_forgetting`, and
`initial_training_forget_loss` with `training_forget_loss`. Inspect
`training_protection`, `validation_protection`, and `projection_attempts` before
deciding on more training. Five steps diagnose solver behavior but do not cover
all 100 training examples at batch size four.

## One localized MLP layer ablation

`config/static_overlap_one_layer.json` differs from the two-layer configuration
only in `architecture.blocks=1`: rank 8, 64 channels, editable subject/alias
embedding rows, and true-answer/abstention head rows. The selected layer has the
highest training forget-versus-retain activation-gradient contrast among its
top channels. It is not chosen arbitrarily or using official paraphrases.

The bounded forget hinge performs GA on forgotten-answer NLL while an example
is below its suppression target. Retain GD, KL, abstention, and delta penalties
stay unchanged for this comparison. A single layer may improve or reduce
feasible forgetting; fewer parameters alone do not guarantee easier constraints.

Run that architecture from the original base with a fresh output directory and
`--config config/static_overlap_one_layer.json --steps 25 --training-only`.
Do not resume two-layer factors into it; continuation rejects that mismatch.
Compare both architectures under the repaired solver and the same retention
limits. An abstention-off or soft-penalty experiment should be a separate
ablation, and official held-out paraphrases remain excluded from fitting.

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
  tests/test_static_overlap_forgetting_priority.py \
  tests/test_static_overlap_active_constraints.py \
  tests/test_static_overlap_edit.py \
  tests/test_static_overlap_export_slack.py \
  tests/test_static_overlap_probability_metrics.py \
  tests/test_static_overlap_mcf_language.py \
  tests/test_mcf_zero_unlearn_metric_parity.py \
  tests/test_mcf_zero_unlearn_released_table_accuracy.py \
  tests/test_gagd_active_case_repair.py
```
