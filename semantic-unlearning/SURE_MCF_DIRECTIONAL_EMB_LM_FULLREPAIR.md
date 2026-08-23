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
