# ZsRE GA/GD Setting 5e + active LM-head repair

This pipeline evaluates the ultra-aggressive GA/GD Setting 5e method on the
same ZsRE protocol used by the vendored ZeroUnlearn code and supplementary
paper.

## Why ZsRE needs a separate adaptation

The ZeroUnlearn ZsRE adapter uses:

- `target_true`: the original answer that should be forgotten;
- `target_new`: the neutral `<|endoftext|>` target;
- rewrite and paraphrase accuracy: lower is better;
- unrelated neighborhood accuracy: higher is better.

CounterFact Setting 5e uses the opposite field convention: its
`target_new` is the unwanted answer and `target_true` is the desired answer.
The ZsRE runner therefore maps the original ZsRE answer into Setting 5e's
internal unwanted slot and the tokenizer's real EOS token into the desired
neutral slot. Applying Setting 5e directly to the raw ZsRE field names would
reverse the forget objective.

The official-compatible evaluator reproduces:

- the first-half retain and second-half forget pools;
- seeded `random.sample`, with forget sampled before retain;
- the rewrite, paraphrase, and token-prefix neighborhood prompts;
- Llama BOS handling;
- token-level last-position greedy correctness;
- per-record macro averaging and 0-100 paper-style scaling.

It also reports micro correct/total token counts so a rounded percentage does
not conceal residual memorized tokens.

## Method

Stage 1 is the established aggressive Setting 5e configuration:

```text
forget records = 50
retain records = 1000
steps = 600
embedding/LM-head AdamW lr = 1e-4
forget weight = 2.0
retain weight = 1.0
sensitive-vs-EOS margin = 1.0
retain batch size = 4
overlap alphas = 0.75 / 0.50 / 0.25
```

Stage 2 freezes the transformer and input embeddings, safely unties a tied
output head, and updates only the tokenizer EOS row in `lm_head`.
Setting 5e's special-token exclusion restores EOS to its base row after the
aggressive training stage; the prompt-constrained active stage is therefore
the only operation that materializes an EOS output-row delta.

- Official Setting 5e metric data, rather than a separately batched cache pass,
  decides which sensitive-answer tokens still need repair.
- The repair cache and scale sweep replay the complete forget split in the
  official prompt order and batch size. Rewrite, paraphrase, and neighborhood
  logits therefore see the same BF16 padding and batch composition as the final
  evaluator.
- Active optimization covers every officially correct sensitive-answer token.
  Scale selection scores every rewrite and paraphrase token, including tokens
  that were already incorrect, so the repair cannot make one correct.
- Protected optimization covers officially correct forget-neighborhood tokens
  plus an independently sampled retain calibration set.
- The learned EOS-row direction is projected away from protected hidden
  states. Rank `0` is intentional: it keeps every remaining feasible direction
  instead of truncating away a hard ZsRE token.
- Each scale is materialized in the actual BF16 LM-head row before exact
  predictions are measured. Scale `0` is evaluated with the identical code path
  and is the numerical regression baseline.
- The smallest materialized perturbation that reaches zero rewrite/paraphrase
  correctness and causes no official-order neighborhood regression is selected.
- Full official evaluation requires `Eff <= 0` and `Gen <= 0`, bounded forget
  Spe and retain Eff/Gen/Spe drops, and bounded PPL growth.
- A failed seed writes diagnostics and exits nonzero. The aggregate command also
  rejects any fallback or nonzero seed, so `22.7` cannot be silently published
  as the selected result.

This is an evaluation-time targeted repair. It makes no claim that unrelated
paraphrases outside the ZsRE protocol are forgotten.

## Full paper-style run

The ZeroUnlearn launcher uses seeds 1-10, so the wrapper does the same:

```bash
cd semantic-unlearning
bash scripts/run_zsre_gagd_setting5e_active_repair.sh \
  /path/to/Llama-3.2-3B-Instruct
```

To run one seed first:

```bash
SEEDS=1 SKIP_PPL=1 \
bash scripts/run_zsre_gagd_setting5e_active_repair.sh \
  /path/to/Llama-3.2-3B-Instruct
```

Do not use `SKIP_PPL=1` for the final ten-seed table. A successful final run
must report `Selected Eff = 0.0`, `Selected Gen = 0.0`, preserve the Setting 5e
Spe/PPL utility gates, and finish the aggregate step. The defaults use 800
repair steps, a `0.25` optimization margin, a `0.05` exact BF16 selection
margin, unrestricted repair rank, and strict zero targets.

If the 600-step Setting 5e checkpoints already exist, avoid repeating Stage 1:

```bash
cd semantic-unlearning
bash scripts/run_zsre_bf16_safe_active_repair_v2.sh \
  outputs/zsre_setting5e_active
```

That root must contain
`seedN/setting5e/checkpoint` for each requested seed. To create those
checkpoints during a combined run, set `SAVE_SETTING5=1`.

The dataset is downloaded on first use to
`data/zsre_mend_eval.json`. Override paths and GPU placement with
`ZSRE_PATH`, `WIKIDATA_DIR`, `OUT_ROOT`, and `CUDA_VISIBLE_DEVICES`.

Important tuning variables include `STEPS`, `EMB_LM_LR`,
`FORGET_WEIGHT`, `RETAIN_WEIGHT`, `FORGET_MARGIN`, `REPAIR_STEPS`,
`REPAIR_LR`, `ACTIVE_LOGIT_MARGIN`, `SELECTION_LOGIT_MARGIN`, `REPAIR_RANK`,
`CANDIDATE_SCALES`, `RETAIN_CALIBRATION_NUM`, `UTILITY_DROP_TOLERANCE`, and
`MAX_PPL_RATIO`.

If a seed fails, inspect
`active_repair/repair_summary.json`,
`active_repair/bf16_exact_scale_sweep.json`, and
`active_repair/candidate_official_eval.json` before tuning. Use this order:

1. If `optimization.all_satisfied` is false, raise `REPAIR_STEPS` to 1200,
   then `ACTIVE_LOGIT_MARGIN` to `0.35`; keep `REPAIR_RANK=0`.
2. If the exact sweep has no scale with zero active correct tokens, add denser
   candidate scales around the best safe scale or raise the active margin.
3. If Eff/Gen are zero but Spe or PPL fails, do not weaken the zero target.
   Increase `RETAIN_CALIBRATION_NUM` first; only then consider a smaller
   selection margin or a tighter scale grid.

## Outputs

Each `seedN` directory contains:

- `base_official_eval.json`;
- `setting5e/official_eval.json`;
- `active_repair/candidate_official_eval.json`;
- `active_repair/bf16_exact_scale_sweep.json`;
- `active_repair/exact_zero_scale_baseline.json`;
- `active_repair/repair_summary.json`;
- `comparison.csv` and `comparison.md`;
- `zsre_results.json`;
- `selected_checkpoint/`, unless disabled.

The ten-seed aggregate is written to:

```text
outputs/zsre_setting5e_active/aggregate/aggregate.md
outputs/zsre_setting5e_active/aggregate/aggregate.csv
outputs/zsre_setting5e_active/aggregate/aggregate.json
```

The supplementary paper reports ZeroUnlearn at 50 forgotten ZsRE samples as
`Eff 27.85 +/- 3.87`, `Gen 27.52 +/- 3.87`,
`Spe 27.73 +/- 2.70`, and `PPL 13.08 +/- 0.06`. These are comparison
references, not expected or hard-coded results for this method.

The desired aggregate
`Eff 0.000 +/- 0.000`, `Gen 0.000 +/- 0.000`,
`Spe about 13.091 +/- 1.884`, and `PPL about 11.3625 +/- 0.4608`
must still be measured on the target GPU/model/checkpoint set. The code enforces
the exact Eff/Gen requirement and relative utility preservation; it does not
fabricate or hard-code Spe/PPL values.
