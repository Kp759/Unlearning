# Last development-preservation adaptation

This experiment is conditional on the saved **actual-model parity check passing**.
The EC2 JSON must contain `training_only_verification.cache_model_parity_passed: true`
with finite numerical errors consistent with the original `1e-4` absolute / `1e-5`
relative tolerance. The launcher checks that evidence and the source file hashes
before creating protocol files. It aborts on missing, stale or failed evidence.
This repository does not contain the user's EC2 audit result.

## Run once on EC2

From `semantic-unlearning`, after checking out the new commit:

```bash
bash scripts/run_static_overlap_final_experiment.sh \
  /home/ec2-user/models/Llama-3.2-3B-Instruct \
  "$PWD/outputs/static_overlap_cached_head_20260910_004019"
```

The launcher checks the existing audit; it does not rerun its boundary search.
If parity fails, stop and debug that discrepancy before proceeding. The launcher
requires the original `retention_boundary_audit.json`, `head_cache.pt`,
`training_report.json`, `head_preparation.json`, and `training_bundle.json`.

The workflow freezes `outputs/static_overlap_final_protocol_seed1/protocol.json`
and `final_retention.json` **before fitting or looking at final results**. A source
run lock prevents generating another test under a new output-directory name.
Do not delete these locks to retune after observing final results. Failed
measurements remain part of the report.

## Dataset roles

| Data | Role in this experiment |
|---|---|
| Original training forget prompts and existing training context augmentations | Forget objective |
| Original training retain and language anchors | Preservation constraints |
| Former validation retain/language spans, including both mixed completion views | **Development preservation constraints** |
| Former validation forget and abstention supervision | Excluded from solve and selection |
| New frozen retain set | Final test only |
| Separate evaluation bundle | Final evaluation only |
| Official MCF scores and official Gen prompts | Final evaluation only; no tuning on scores or Gen prompts |

The active dataset is saved as `development_examples.json`; each entry carries
`split: train` or `split: development`. Historical IDs may still begin with
`validation_` for traceability. `dataset_membership.json` records the same mapping.
`source_bundle.json` is an immutable archive, not the active fitting dataset.
Reports use `development_protection`, never `validation_protection`, for reused
contexts. These contexts no longer supply held-out preservation evidence.

The new factual test contains **256 independent subject-relation pairs**, each
with its direct prompt and first distinct genuine MCF paraphrase (512 prompts).
It comes from the first-half MCF pool, with all development associations, separate
evaluation associations, and official sampled cases excluded. Prompt/full-text
collisions are also excluded. Selection uses SHA-256 ordering with seed 20260910,
never model scores. This is an independently reserved factual test from the same
source corpus; it is not evidence of preservation on every language domain.
Separate-bundle language anchors are scored separately as part of final retention.

## Solve and selection

The backbone and original embeddings stay frozen. The original base model is
loaded fresh and its tied head is explicitly copied with base-function parity.
The original base feature cache is reused; **no prior edit delta is resumed**.
The retention feature matrix now includes `R_train ∪ R_development`, producing
new head directions rather than scaling the former training-only directions.
All final features remain outside the cache supplied to the solver.

The declared grid is tau `[0, .001, .01]`, ridge `[.0001, .01]`, strength
`[.25, .5, 1, 2, 4, 8, 16, 24, 32]` (54 candidates). Both fitting and development
preservation use explicit internal margins `.01` NLL and `.002` KL, hence
internal budgets `.04/.008`. Scientific final limits remain **`.05/.01`**.
The frozen protocol records all parameters before fitting and rejects mismatches.
Internal margins are configurable with `--nll-safety-margin` and
`--kl-safety-margin` on the freeze command, before this protocol is created; pass
the same values to the training command. The launcher uses the declared defaults.
Changing the frozen experiment after inspecting test results is not permitted.

Among development-valid candidates meeting the training probability target
`1e-6`, select the smallest weight-update norm. If none meets that target, select
the lowest worst, then mean, training probability among valid improved candidates.
This predeclared rule avoids escalating edit strength after the training target
is already met. It cannot guarantee performance on official Gen or the final test.
The selected edit must reproduce on the real model, pass development retention,
merge into native weights, and pass reload verification before final evaluation.
The existing repaired projection solver and its regression tests are preserved;
this frozen linear-head path uses the previously implemented FP64 SVD/regression.

## Final evidence

`evaluate_static_overlap_final_retention.py` binds the frozen protocol to one
verified checkpoint. It evaluates the independent retain set and the separate
evaluation bundle's retain/language anchors against the original base using
full-vocabulary KL(base||edit) and per-anchor mean answer-NLL increase. It uses
**zero numerical budget slack**. No final score participates in checkpoint
selection, localization, fitting, or hyperparameter adjustment. An interrupted
measurement can retry the same checkpoint; a different checkpoint is rejected.

The official evaluator then scores the same checkpoint. The launcher writes both
reports even if preservation fails. Its final exit status is zero only if both
final preservation sets pass and the existing unrounded official Eff/Gen checks
pass: each probability percentage below `.005` and each released accuracy zero.
These are display-zero criteria, not an assertion of mathematically zero
probability or universally erased knowledge. `--max-new-tokens 1` limits only the
generation diagnostic, not teacher-forced official answer scoring.

Outputs:

- `outputs/static_overlap_final_protocol_seed1/final_retention_results.json`
- `outputs/static_overlap_development_head_<timestamp>/training_report.json`
- `outputs/static_overlap_development_head_<timestamp>/evaluation_probability_v2.json`

On export failure, saved factors can be recovered using
`export_static_overlap_edit.py --training-run <development-run> --device cuda --local-files-only`.
Recover **before** final scoring; no new solve or test set is needed. The existing
factor recovery supports the explicit development split and head separation.

Suggested disclosure (use the past tense only after final evaluation completes):

> The initial held-out retention audit identified development-context failures.
> We subsequently incorporated those contexts into the development preservation
> set and evaluated final retention on a separately held-out set not used for
> localization, optimization, checkpoint selection, or hyperparameter tuning.

This is the last major protocol adaptation. Failure of the frozen final test is
reported as failure, not remedied by another test split or a looser threshold.
