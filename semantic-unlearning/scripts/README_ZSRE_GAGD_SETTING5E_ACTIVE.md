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
steps = 250
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

- Active constraints cover every still-correct sensitive-answer token in the
  official forget rewrite and paraphrase prompts.
- Protected constraints cover initially correct forget-neighborhood tokens and
  an independently sampled retain calibration set.
- The learned EOS-row direction is projected away from protected hidden
  states.
- A scale sweep selects the smallest-regression candidate.
- Full official evaluation then requires non-regressing forget Eff/Gen,
  bounded forget Spe and retain Eff/Gen/Spe drops, and bounded PPL growth.
- If a strict gate fails, the EOS row is restored exactly to Setting 5e before
  the selected checkpoint is saved.

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

The dataset is downloaded on first use to
`data/zsre_mend_eval.json`. Override paths and GPU placement with
`ZSRE_PATH`, `WIKIDATA_DIR`, `OUT_ROOT`, and `CUDA_VISIBLE_DEVICES`.

Important tuning variables include `STEPS`, `EMB_LM_LR`,
`FORGET_WEIGHT`, `RETAIN_WEIGHT`, `FORGET_MARGIN`, `REPAIR_STEPS`,
`REPAIR_LR`, `ACTIVE_LOGIT_MARGIN`, `REPAIR_RANK`,
`RETAIN_CALIBRATION_NUM`, `UTILITY_DROP_TOLERANCE`, and `MAX_PPL_RATIO`.

## Outputs

Each `seedN` directory contains:

- `base_official_eval.json`;
- `setting5e/official_eval.json`;
- `active_repair/candidate_official_eval.json`;
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
