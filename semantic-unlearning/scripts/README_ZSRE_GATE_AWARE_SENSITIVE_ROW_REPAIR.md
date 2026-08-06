# Gate-aware sensitive-row LM-head repair for ZsRE

This is a new repair-only ZsRE experiment. It does not modify or supersede the
authoritative v2 record in `config/best_runs/zsre/`, and it does not rerun the
600-step Setting 5e stage.

The method name is:

> Gate-aware sensitive-row LM-head repair

Its protocol status is:

> `native_data_and_metrics_but_evaluation_conditioned_repair`

That label is required because official rewrite/paraphrase correctness selects
the residual active and anti-regression cases, official forget-neighborhood
and retain correctness select protected constraints, and native metrics select
the BF16 candidate scale. This is not a blind or held-out repair protocol.

## Motivation

The v2 repair changes only the global `Unknown` output row. It reaches zero
forget Eff and Gen, but all ten recorded seeds fail the strict retain/utility
gates. This experiment takes the opposite local intervention: it leaves
`Unknown` unchanged by default and lowers only sensitive answer-token rows that
remain top-1 after Setting 5e.

For a baseline-correct sensitive token `s` with final hidden state `h`, the
active constraint is

```text
best_other_logit - (sensitive_logit + h @ delta_s) >= active_margin
```

The default active margin is `0.02`, which avoids relying on a BF16 tie.
Every rewrite/paraphrase target position that is already incorrect receives a
separate anti-regression constraint. This prevents repair from making an
incorrect target correct by lowering an edited competitor row.

## Immutable starting point

Each run requires:

- `seedN/setting5e/checkpoint`, produced by the existing 600-step Setting 5e
  pipeline;
- the matching `seedN/zsre_results.json`, used to verify the step count, seed,
  data counts, dataset hash, Base metrics, and saved Setting 5e metrics;
- the checkpoint's `zsre_neutral_target.json`, which must resolve `Unknown` to
  the tokenizer's current single token ID.

The repair refuses a checkpoint if those identities disagree. Stage 1 is never
invoked by the Python script or shell runner.

The official sampler remains unchanged: with the seeded RNG, forget examples
are sampled first from the second half and retain examples second from the
first half. Defaults are 50 forget records and 1,000 retain records.

## Editable state

The implementation calls the existing safe output-head freezer. It:

1. unties `lm_head` if it shares storage with input embeddings;
2. freezes every model parameter;
3. learns a separate FP32 selected-row delta;
4. materializes only explicitly selected output rows.

Input embeddings and transformer parameters are never trainable. Every
tokenizer special ID is excluded. The `Unknown` row is also excluded by
default. `--edit-unknown-row` exists only as an explicitly labeled exploratory
override and is not part of the primary configuration.

Selected row IDs, decoded token pieces, active-case counts, excluded cases, and
special IDs are written to `selected_sensitive_rows.json`.

## Active, anti-regression, and protected constraints

Active constraints cover every officially baseline-correct rewrite and
paraphrase token remaining in the loaded Setting 5e checkpoint. The cache uses
the same prompt expansion, target-token construction, case order, BF16 model,
and batch size as the official evaluator.

Anti-regression constraints cover every other forget rewrite/paraphrase target
position. For each baseline-incorrect target, the repair requires

```text
best_corrected_competitor_logit - corrected_target_logit
    >= forget_regression_margin
```

The competitor is the maximum of the strongest unchanged vocabulary row and
every corrected selected row. If the target row is selected, its correction is
applied to the target and that row is excluded from the competitor set. The
default anti-regression margin is `0.02`.

Protected constraints cover:

- every officially correct forget-neighborhood token;
- every officially correct rewrite token from all 1,000 retain records;
- every officially correct paraphrase token from all 1,000 retain records;
- every officially correct neighborhood token from all 1,000 retain records.

For each protected target, the required margin is

```text
min(original_top1_margin, protected_margin_cap)
```

with a default cap of `0.05`.

When the protected target row is edited, its corrected target logit must remain
above both the strongest unchanged competitor and every other corrected edited
row. When the target row is not edited, its unchanged target logit must remain
above every corrected edited row. Constraint counts and violations before and
after FP32 optimization are recorded in
`optimization/constraint_summary.json`.

## Retain KL and objective

All prompt-token decisions from all 1,000 official retain records contribute
to selected-row KL. The loaded Setting 5e distribution is the reference. The
implementation reuses the low-memory exact selected-row correction formula in
`gagd_active_case_repair.py`; optimizer state scales only with selected rows.

The objective is

```text
active_hinge_weight    * active_squared_hinge
+ forget_regression_hinge_weight * forget_regression_squared_hinge
+ protected_hinge_weight * protected_squared_hinge
+ retain_kl_mu          * retain_KL
+ delta_l2_lambda       * ||delta||^2
```

Primary defaults:

| setting | default |
| --- | ---: |
| repair steps | 3000 |
| repair learning rate | 1e-3 |
| optimizer | AdamW |
| active margin | 0.02 |
| forget anti-regression margin | 0.02 |
| protected margin cap | 0.05 |
| active hinge weight | 2.0 |
| forget anti-regression hinge weight | 2.0 |
| protected hinge weight | 50.0 |
| retain KL weight | 10.0 |
| delta L2 weight | 1e-4 |
| retain calibration records | 1000 |
| retain calibration seed | 1729 |
| stop when all satisfied | false |
| repair rank | 0 (unrestricted) |
| model/evaluation dtype | BF16 |
| evaluation batch size | 8 |
| cache batch size | 8 |
| protected optimization batch | 256 constraints |
| retain-KL optimization batch | 32 complete retain records |
| terminal/live progress interval | 10 steps |
| full-constraint check interval | 100 steps |

Because early stopping defaults to false, optimization continues after active
constraints first become satisfied so KL and L2 can reduce collateral damage.
Cache and evaluation batch sizes must remain equal so the BF16 cache reproduces
the evaluator's padding and batch composition exactly.
The calibration seed is recorded for reproducibility; the primary protocol
uses the entire retain set, so it does not subsample membership.

Optimization always uses the complete active set. Anti-regression and
protected constraints plus retain-record KL use independent deterministic
cyclic mini-batches: every item is visited once before its cycle repeats. The
anti-regression bank is complete on every step when it fits in one configured
batch. This batching is only an optimization approximation. The complete
active/anti-regression/protected sets and all 1,000 retain records are rechecked
before final acceptance; all constraint banks are also checked in bounded
no-gradient chunks every 100 steps. Exact official Eff/Gen/Spe plus PPL gates
remain unchanged. Progress is printed with flushing and appended incrementally
to `optimization/live_progress.jsonl`.

## Exact BF16 scale sweep

The default scale grid is `1.0, 0.975, ..., 0.025, 0.0`. Every scale is set
from the original selected rows (never accumulated), cast into the actual
output-weight dtype, and evaluated with the official ZsRE evaluator without
PPL first.

A non-PPL candidate survives only if:

```text
forget Eff <= 0.0
forget Gen <= 0.0
forget Spe >= Setting5e forget Spe - 0.10
retain Eff >= Setting5e retain Eff - 0.10
retain Gen >= Setting5e retain Gen - 0.10
retain Spe >= Setting5e retain Spe - 0.10
original active violations = 0
forget anti-regression violations = 0
newly correct forget target tokens = 0
```

Each materialized scale is also rerun with the evaluator's exact BF16 forget
batch construction. The sweep records baseline-correct rewrite/paraphrase
tokens that remain correct, baseline-incorrect tokens that become newly
correct, and full token identities/logits/margins for every regression.

Only survivors are evaluated on the official Wikipedia PPL corpus. The final
gate is:

```text
candidate PPL <= Setting5e PPL * 1.02
```

The fixed `0.10` percentage-point utility tolerance, `1.02` PPL ratio, and
zero Eff/Gen targets cannot be weakened through the CLI. Among candidates that
pass every gate, the materialized candidate with the smallest selected-row
delta norm is chosen. Scale zero is ineligible unless Setting 5e itself already
has zero Eff and Gen.

PPL text, unseen prompts, and held-out prompts never contribute gradients or
constraint/scale optimization. PPL is opened only for a non-PPL survivor.

## Failure behavior

A rejected seed has:

- `repair.candidate_accepted = false`;
- `selected = null`;
- `repair.selected_scale = null`;
- no `selected_checkpoint/` directory;
- a nonzero process exit when `--fail-if-target-missed` is active.

The code never restores Setting 5e and calls that fallback "selected." A raw
full-scale optimizer checkpoint is still written to
`active_candidate_checkpoint/` for diagnosis.

The shell runner executes every requested seed before deciding whether the
suite failed. When any seed is rejected and
`FAIL_IF_ANY_SEED_REJECTED=1`, it exits nonzero and emits no aggregate. The
aggregator is called only when every requested seed is accepted, and no
fallback-selected option is used.

## Output layout

```text
OUTPUT_ROOT/seedN/
├── config_used.json
├── sampled_case_ids.json
├── setting5e_official_eval.json
├── selected_sensitive_rows.json
├── optimization/
│   ├── active_cases.jsonl
│   ├── interrupted.json          # Ctrl+C only; no selected checkpoint survives
│   ├── live_progress.jsonl
│   ├── newly_correct_forget_cases.jsonl
│   ├── repair_log.jsonl
│   └── constraint_summary.json
├── scale_sweep/
│   ├── non_ppl_gate_sweep.json
│   └── full_gate_sweep.json
├── active_candidate_checkpoint/
├── selected_checkpoint/          # accepted seeds only
├── selected_official_eval.json   # accepted seeds only
└── zsre_results.json
```

`zsre_results.json` retains the existing aggregator's `base`, `setting5e`,
`active_candidate`, and `selected` metric blocks. It also records fixed gate
reports, selected-row and constraint counts, protocol status, selected scale,
and the deterministic selected-checkpoint tree SHA-256.

ZsRE aggregation reports sample standard deviation (`ddof=1`; zero for one
seed), matching the authoritative v2 record's uncertainty convention.

## One-seed smoke run

From `semantic-unlearning/` on the NJIT HPC login or allocated GPU node:

```bash
PYTHON_BIN=/mnt/train/venvs/unlearning/bin/python \
CUDA_VISIBLE_DEVICES=0 \
SEEDS="1" \
FAIL_IF_ANY_SEED_REJECTED=1 \
bash scripts/run_zsre_gate_aware_sensitive_row_repair.sh \
  outputs/zsre_cal384_uniform_unknown_seeds1_10 \
  outputs/zsre_gate_aware_sensitive_row_repair_smoke
```

The first argument must contain
`seed1/setting5e/checkpoint` and `seed1/zsre_results.json`.

## Seeds 1–10

```bash
PYTHON_BIN=/mnt/train/venvs/unlearning/bin/python \
CUDA_VISIBLE_DEVICES=0 \
SEEDS="1 2 3 4 5 6 7 8 9 10" \
FAIL_IF_ANY_SEED_REJECTED=1 \
bash scripts/run_zsre_gate_aware_sensitive_row_repair.sh \
  outputs/zsre_cal384_uniform_unknown_seeds1_10 \
  outputs/zsre_gate_aware_sensitive_row_repair_seeds1_10
```

Override any non-fixed optimization setting through the environment, for
example `REPAIR_STEPS`, `REPAIR_LR`, `REPAIR_OPTIMIZER`, `ACTIVE_MARGIN`,
`FORGET_REGRESSION_MARGIN`, `PROTECTED_MARGIN_CAP`, `ACTIVE_HINGE_WEIGHT`,
`FORGET_REGRESSION_HINGE_WEIGHT`,
`PROTECTED_HINGE_WEIGHT`, `RETAIN_KL_MU`, `DELTA_L2_LAMBDA`,
`RETAIN_CALIBRATION_NUM`, `RETAIN_CALIBRATION_SEED`,
`PROTECTED_BATCH_SIZE`, `RETAIN_KL_BATCH_SIZE`, `PROGRESS_EVERY`,
`FULL_CONSTRAINT_CHECK_EVERY`, `REPAIR_RANK`, `CANDIDATE_SCALE_STEP`,
`DTYPE`, `EVAL_BATCH_SIZE`, or `CACHE_BATCH_SIZE`.
