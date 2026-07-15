# Compare four-scope GA/GD with JSON-LMHead-Zero + Target-True Emb Restore

## Research question

Does direct closed-form token-row surgery outperform gradient-based suppression
when both methods see the same forget/retain records and use the same evaluator?

## Baseline

Use the existing seed 0-4 official results:

```text
outputs/official_eval_lmhead_zero_true_restore150_seed{seed}_spefix.json
```

Also include the matching base-model result for each seed.

## Required comparison

- Validate dataset, seed, sample mode, forget count, and retain count before
  combining results.
- Report per-seed Eff, Gen, Spe, and PPL for:
  - base model;
  - JSON-LMHead-Zero + Target-True Emb Restore 1.50;
  - all four GA/GD settings.
- Report mean and population standard deviation across seeds 0-4.
- Preserve source paths in a machine-readable manifest.
- Treat Eff/Gen as lower-is-better, Spe as higher-is-better, and PPL as
  lower/stable-is-better.

## Acceptance criteria

- Missing results fail by default and can only be skipped explicitly.
- Mismatched run metadata fails with an actionable error.
- Both Markdown and CSV tables are produced.
- A single JSON artifact contains config, missing inputs, per-seed rows, and
  aggregate rows.

## Implementation files

- `scripts/compare_gagd_to_json_lmhead.py`
- `README_GAGD_COMPARISON.md`
- `tests/test_compare_gagd_to_json_lmhead.py`
