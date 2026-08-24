# MCF SURE: directional Emb+LM GA -> failure-only full-row repair

This experiment intentionally removes the surrogate-paraphrase and LoRA paths.
It uses the locked direct-only target-true-sensitive MCF protocol.

## Target contract

- `requested_rewrite.target_true`: sensitive / unwanted fact.
- `requested_rewrite.target_new`: non-sensitive reference used for direction construction and the direct margin.
- Fields are never swapped.
- Official paraphrases, neighborhoods, benchmark retain records, and PPL text are held out from training and checkpoint selection.

## Architecture

```text
Base Llama
   |
   | clone / untie input embedding E and LM head W
   | transformer Phi frozen
   v
identify target_true-sensitive vocabulary rows S
   |
   v
row-specific sensitive direction
   d = h_true - h_new
   |
   | if d = 0 (notably the first answer token)
   v
   d = w_true - w_new decoder-discriminant fallback
   |
   v
Stage 1
   selected E[S] delta constrained to span(d)
   selected W[S] delta constrained to span(d)
   GA on target_true sensitive token log-probability
   Base non-sensitive distribution KL guard
   no GD/CE toward target_new
   no LoRA
   |
   v
direct margin gate
   margin = NLL(target_true) - NLL(target_new)
   required margin = 0.05 by default
   |
   +---- pass -------------------------------+
   |
   +---- fail                                |
           |                                 |
           v                                 |
       Stage 2                               |
       select target_true LM-head rows       |
       belonging to failed records only      |
       unrestricted full-row delta           |
       no rank basis / no LoRA                |
       failed-record hinge objective          |
       passing direct records guard regression|
           |                                 |
           +---------------------------------+
                           |
                           v
                         final
```

## Why the direction has a fallback

At the first answer token, the model sees exactly the same prompt prefix whether the future answer is `target_true` or `target_new`. Therefore the two pre-answer hidden states can be identical and `h_true - h_new = 0`.

The fallback

```text
d = w_true - w_new
```

is the hidden-space gradient of the true-vs-reference logit gap for that token pair. It gives a non-zero discriminative direction without opening held-out paraphrases or other benchmark probes.

For later teacher-forced tokens, the true and reference prefixes can differ, so the literal hidden-state contrast is used when non-zero.

## Synthetic paraphrase augmentation (Gen-closing fix)

The direction `d` and the direct-margin gate were originally fit from a
single literal prompt template per record. Because `d` is a fixed direction
and the row-edit's suppression strength is `hidden_state . d`, a direction
fit from one template only suppresses the sensitive token where a held-out
paraphrase's hidden state happens to align with that same direction --
partial, not guaranteed, transfer. This showed up empirically as GFS < 100%
(Gen > 0) even when FS = 100% (Eff = 0).

`scripts/mcf_synthetic_paraphrase_templates.py` hand-authors, for each of
the 34 MultiCounterFact relation ids, 2 alternate cloze templates that are
syntactically distinct from the dataset's own canonical
`requested_rewrite.prompt` template for that relation, plus a small bank of
generic content-free context-prefix sentences. Both axes are authored from
scratch and never derived from, or copied from, any record's real
`paraphrase_prompts` -- the official held-out paraphrase set stays
uncontaminated, so GFS/Gen remains an honest measure of whether this
transfers.

`--synthetic-paraphrases-per-record` (default `3`) controls how many such
templates are generated per record. Stage 1 builds `all_records = records +
synthetic_records` and uses it (not `records` alone) for:

- the row-specific contrast-direction basis (`build_row_specific_contrast_bases`);
- GA training and the base-logit KL cache;
- the direct-margin scale-selection gate, which now requires the margin on
  every synthetic template too (reported per-scale as `direct_only_failures`
  vs `synthetic_failures` in `scale_reports`).

`stage1_config.json` keeps the original `stage1_direct_failures` /
`stage1_failing_positions` / `stage1_minimum_margin` fields exactly as
before (direct-prompt-only, for backward compatibility), and adds
`stage1_synthetic_failures`, `stage1_synthetic_failing_positions`,
`stage1_synthetic_minimum_margin`, `stage1_combined_failures`, and
`stage1_combined_minimum_margin` alongside them. Set
`--synthetic-paraphrases-per-record 0` to fully recover the original
direct-only behavior.

Stage 2 (`mcf_sure_fullrow_failure_repair.py`) is unchanged by this and
still gates only on the literal direct prompt; it was not the source of the
Gen gap here since Stage 1 alone already reaches FS = 100%.

## Stage-1 embedding caveat

After untying, an input-embedding row receives ordinary GA gradient only when that token actually occurs in the teacher-forced input prefix. Consequently, some single-token answer embedding rows may remain unchanged even though their LM-head rows receive GA gradient. The implementation logs `embedding_rows_with_nonzero_current_grad` rather than hiding this causal fact.

## Run on the clean locked seed-1 split

```bash
cd /home/ec2-user/workspace/Unlearning/semantic-unlearning

BASE=/home/ec2-user/models/Llama-3.2-3B-Instruct
SPLIT_ROOT=outputs/mcf_targettrue_clean_seed1/seed1
VISIBLE="$SPLIT_ROOT/protocol/training_visible_mcf_target_true.json"
MANIFEST="$SPLIT_ROOT/protocol/split_manifest.json"

bash scripts/run_mcf_sure_directional_emb_lm_fullrepair.sh \
  "$BASE" "$VISIBLE" "$MANIFEST"
```

Default output:

```text
outputs/mcf_directional_emb_lm_fullrepair_seed1/
  stage1/
    checkpoint/
    stage1_config.json
    train_log.jsonl
  stage2_fullrow_repair/
    checkpoint/
    repair_summary.json
    scale_sweep_direct_only.json
```

## Default optimization

Stage 1:

- 600 steps
- batch size 2
- LR `1e-4`
- GA weight 2
- non-sensitive-distribution KL weight 1
- delta L2 `1e-6`
- per-row direction rank 1
- direct constraint margin `0.05`
- synthetic paraphrase templates per record `3` (hand-authored, see below)

Stage 2:

- 800 steps
- unrestricted sparse LM-head rows
- LR `5e-3`
- delta L2 `1e-6`
- pass regression guard weight 1
- direct constraint margin `0.05`

These are one declared configuration, not a rank sweep.

## Mechanical tests

```bash
pytest -q tests/test_mcf_sure_directional_emb_lm_stage1.py
python -m py_compile \
  scripts/mcf_sure_directional_emb_lm_stage1.py \
  scripts/mcf_sure_fullrow_failure_repair.py
```
