# Fixed-mask tied embedding/head gradient ascent

This is a separate exploratory follow-up to the completed head experiment and
one-MLP pilot. It tests stronger forgetting capacity inside the **existing
endpoint row mask**. It does not claim that the earlier held-out preservation
failure was solved, or that perfect forgetting is guaranteed.

## What changes

The runner starts from the original FP32 tied Llama base. It reads the exact
`input_rows` and `output_rows` from the earlier static-overlap replay manifest.
Their union is the physical support of **one shared dense delta**. The same
delta participates in input embedding lookup and output prediction. Each
selected row can move in every hidden dimension: there is no rank-8 constraint.
All transformer parameters and all unselected endpoint rows remain frozen.
No token suppression mask, router, output filter, or inference adapter remains
in an exported native model. The embedding and head stay tied after merging
and reloading. A saved hash audit checks every frozen parameter and every
unselected row after reload.

The old replay already included low-rank endpoint updates and capped NLL ascent;
this is not the first use of endpoint GA. The new variables are the unrestricted
within-row delta and a forgetting-first optimizer. The row set, data, semantic
overlap categories, thresholds and final tests are not redesigned. A fixed row
set does not by itself guarantee preserved behavior in every shared-token
context; full preservation is still measured.

## Exact optimization

For each fitting forget example, minimize
`max(0, max(base_NLL + 2, -log(1e-6)) - edited_NLL)` over its answer tokens.
While this hinge is active, descent on it is ascent on the answer NLL. Once a
particular example reaches its target, its hinge gradient is zero. Adam uses
**only this forgetting gradient**, at learning rate 0.003, with gradient norm
clipped to 1. There is no large retention penalty mixed into Adam's gradient.

At each step, eight fitting preservation examples supply separate NLL and
full-vocabulary forward-KL gradients. A Dykstra projection constrains the Adam
proposal using the remaining internal 0.04 NLL / 0.008 KL allowances and an L2
step radius of 0.5. A maximum of five actual-model trials (the proposal plus
four halvings) must pass those **minibatch** limits and improve the same forget
batch. A failed projection or line search restores both the delta and Adam
state. Projection tolerance is 1e-6; it never replaces the actual model check.

If a newly visited fitting preservation example already violates an internal
limit, that step instead follows the negative gradient of the sum of squared
normalized excesses. An actual-model line search must reduce the maximum
normalized violation on that batch. Accepted repair clears Adam momentum.
Such a repair can worsen forgetting; this is explicitly logged and complete
forgetting gates remain authoritative. No claim of per-step global feasibility
is made. Other examples can still be affected by a minibatch-accepted update.

Forget batches contain 16 examples, initially rotating through all fitting
data. After a full gate, half the slots replay the lowest-NLL fitting examples;
preservation replays its worst fitting violations. Rotating cursors continue
across gate changes. Development examples never determine gradients, replay or
minibatch acceptance. Development scores only qualify a checkpoint. There are
no hidden-state caches from the head fit: changing embeddings changes hidden
states. Only immutable original-model reference probabilities are cached.

## Bounded execution and gates

- At most 80 optimizer attempts, with a full gate every 20 attempts.
- Soft 1,200-second **training loop** limit, checked between attempts. A pending
  attempt and final full gate can extend it. Model loading, base references,
  export and final evaluation take additional time.
- Stop after 12 consecutive rejected steps.
- Stop after two consecutive gates each gaining less than 0.05 in mean fitting
  forget NLL since the previous gate. The first comparison uses a saved base
  measurement. This is an efficiency heuristic, not proof of infeasibility.
- Export only the first scheduled state satisfying **all** fitting/development
  forgetting targets plus full preservation at 0.05 NLL / 0.01 KL, with no
  added numerical retention allowance.
- A failed gate returns `no_development_valid_edit` and exit status 2. The shell
  stops before export or final model evaluation. It never substitutes the base
  model as a successful forgetting result.

The development data and their hashes are reused from
`outputs/static_overlap_mlp_pilot_seed1/pilot_protocol.json`; no new dataset is
downloaded. The earlier MLP/head contracts and reports remain unchanged. The
new experiment registers under `outputs/static_overlap_endpoint_ga_seed1` and
refuses to overwrite or restart an existing training run.

Successful export performs native merge, save, reload, logit parity and all
development gates again. It then runs the same frozen final retention and
separate evaluation bundles and official MCF metrics once for that checkpoint.
These final sets have already been observed in the head experiment, so this is
exploratory evidence, not an independent confirmatory experiment. Official Gen
remains excluded from fitting. No threshold or metric definition changes.

## EC2 command

After the old GPU process finishes, from `semantic-unlearning`:

```bash
bash scripts/run_static_overlap_endpoint_ga.sh \
  /home/ec2-user/models/Llama-3.2-3B-Instruct \
  "$PWD/outputs/static_overlap_replay_cached_20260909_215959/manifest.json"
```

The second argument must be the **original replay manifest**, not a head export
or MLP manifest. It is required so the previous editable row mask is preserved
exactly rather than regenerated with a potentially different tokenizer.

The terminal prints each attempt's `mode`, raw forgetting and repair gradient
norms, actual batch gap before/after, projection status and elapsed time. A
full gate prints probability extrema and the number of preservation failures;
full failure IDs stay in JSON files rather than flooding the terminal.

Files under `outputs/static_overlap_endpoint_ga_seed1/`:

- `baseline_metrics.json`: original-model probabilities on these same prompts.
- `endpoint_mask.json`: immutable support and original-manifest checksum.
- `training.jsonl`, `training_report.json`: attempts, gates, stopping reason.
- `last_metrics.json`, `last_endpoint_delta.pt`: last measured state, including
  failures. These are diagnostic artifacts, not a qualified checkpoint.
- `checkpoint/`: created only after a training/development gate passes. Only
  `static_edit_export.json` with `verified: true` establishes completed export.
- `exploratory_summary.json`: produced only after qualified final evaluation.

Local tests cover actual tiny-Llama endpoint/native gradients, tied merge and
mask invariants, the sign of GA, optimizer rollback, data isolation, early
stopping, strict export rejection and the qualified native export path. They do
not establish near-zero forgetting on the EC2 Llama-3.2-3B model. Production
Eff, Gen and preservation results must be measured by the run above.
