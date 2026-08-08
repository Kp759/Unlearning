# MQuAKE Setting 5e multi-row active repair

This directory contains an experimental extension of the existing MQuAKE
Setting 5e reproducibility baseline. The baseline
`mquake_gagd_setting5e_active_repair.py` is unchanged.

The method first runs the established 600-step Setting 5e configuration. Its
default `instance_balanced` sampler draws a sampled MQuAKE instance uniformly,
then draws one of that instance's `requested_rewrite` atoms uniformly. The
`atomic_epoch` option preserves flat atomic-fact sampling as an ablation.

After Setting 5e, the transformer and input embeddings are frozen. A tied
output head is cloned before repair, preserving its logits exactly. The repair
learns one output-row delta for `Unknown` and one shared delta for each token ID
that remains an active teacher-forced sensitive-token failure. Sensitive rows
without a residual failure are not trainable. Protected retain hidden states
both define the optional orthogonal projection and receive an explicit
modified-row logit-drift penalty.

Every BF16 candidate scale is materialized jointly from immutable Setting-5e
rows. Scale zero exactly restores those rows. A candidate is accepted only if:

- forget Eff is exactly 0.00%;
- retain Eff is no more than 0.10 percentage points below Base;
- selected PPL is at most 1.02 times Base PPL; and
- no protected token regresses relative to the zero-scale Setting-5e baseline.

AtomicGen and the official standard/CoT multi-hop questions are opened only
after `selection_commit.json` records an irrevocable decision. They cannot
affect scale or checkpoint selection. `--require-atomic-gen-zero` is therefore
only a post-selection exit policy.

## Run

```bash
bash scripts/run_mquake_setting5e_multiroot_active_repair.sh MODEL_PATH 0
```

Override `OUT_ROOT`, `MQUAKE_PATH`, `WIKIDATA_DIR`, `DTYPE`, `DEVICE_MAP`, or
the batch-size environment variables when needed. Set
`FORGET_SAMPLING=atomic_epoch` for the sampling ablation.

## Diagnostics

The repair directory contains:

- `active_rows.json`
- `active_tokens_before.jsonl`
- `protected_tokens_before.jsonl`
- `multirow_repair_log.jsonl`
- `bf16_exact_multirow_scale_sweep.json`
- `row_delta_norms.json`
- `repair_summary.json`

The selected output also reports Eff, Eff_micro, Eff_instance_macro,
AtomicGen variants, RetainEff, RetainAtomicGen, PPL, aggregate standard/CoT
MHLeak, and per-hop 2/3/4 MHLeak when those hops occur in the pinned split.
