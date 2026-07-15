# Implement four-scope GA/GD on the official MCF split

## Research question

How much of the JSON-LMHead-Zero result can ordinary gradient ascent/descent
recover when parameter scope and token scope are controlled independently?

## Scope

Run answer-only GA/GD in four settings:

1. `full_all_tokens`
2. `full_selective_tokens`
3. `emb_lm_all_tokens`
4. `emb_lm_selective_tokens`

Selective-token modes use forget subject, `target_new`, and `target_true` token
IDs. The embedding/LM-head selective mode must mask gradients and restore every
non-selected row after each optimizer step.

## Required implementation

- Use the same official MCF split as JSON-LMHead-Zero and ZeroUnlearn:
  - retain records from the first half;
  - forget records from the second half;
  - deterministic `random.sample` by seed.
- Optimize `-forget_answer_nll + retain_answer_nll` using `target_new`, so the
  trained objective is directly comparable with target-new LM-head suppression.
- Report trainable parameter count, selected-token count, before/after NLL,
  target-new-over-target-true success, and training logs.
- Run official MCF evaluation directly on each in-memory trained model.
- Keep checkpoint saving optional because four 3B checkpoints per seed are not
  needed for metric-only comparisons.

## Acceptance criteria

- All four modes run from one command.
- Training and official evaluation use identical records for a given seed.
- Non-selected embedding/LM-head rows are bitwise restored in the selective-row
  mode.
- Each seed produces one official JSON file per mode.
- A smoke run supports two forget records, four retain records, and one step.

## Implementation files

- `scripts/gagd_compare.py`
- `scripts/mcf_sampling.py`
- `scripts/mcf_zero_unlearn_official_eval.py`
- `scripts/run_gagd_vs_json_lmhead.sh`
