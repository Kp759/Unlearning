# MCF exact-subject scoped span edit

This is the runnable resolution of the writer-only gating failure in
`scripts/mcf_marker_write_read.py`.

The causal point is that gating only the hidden writer removes neighborhood
state drift but leaves the Stage-2 LM-head reader global. Its `q · h_n` term,
and therefore `L_cross`, remains active. `--writer-mode span_gated` now uses one
input-derived exact-subject router for both interventions:

```text
complete forgotten subject present
  -> per-record residual at --writer-layer
  -> per-record sparse logit reader

no complete forgotten subject
  -> identity writer
  -> identity reader
  -> bit-identical Base logits
```

The scope is deliberately **subject-only**, not subject-and-relation. That lets
the same edit activate on held-out relation paraphrases without reading those
paraphrases during training. It also means this is scoped editing/suppression,
not evidence that the base weights no longer contain the fact, and not a
relation-specific semantic gate.

## Run

From `semantic-unlearning/`:

```bash
python -u scripts/mcf_marker_write_read.py \
  --model-path /path/to/base-model \
  --mcf-path data/multi_counterfact.json \
  --wikidata-dir data/wikidata \
  --output-dir outputs/mcf_scoped_span_seed1 \
  --seed 1 \
  --writer-mode span_gated \
  --writer-layer 8 \
  --gate-criterion scope \
  --gate-pass-frac 1.0 \
  --save-checkpoint
```

Then run the ordinary official evaluator:

```bash
python -u scripts/mcf_zero_unlearn_official_eval.py \
  --model-dir outputs/mcf_scoped_span_seed1/checkpoint \
  --mcf-path data/multi_counterfact.json \
  --wikidata-dir data/wikidata \
  --out outputs/mcf_scoped_span_seed1/official_eval.json \
  --seed 1 \
  --sample-mode official \
  --dtype bfloat16 \
  --device-map auto
```

The checkpoint is a normal Hugging Face directory plus
`scoped_span_edit.pt`. The evaluator discovers that sidecar automatically. Its
router sees only complete subject token sequences in `input_ids`; it does not
receive rewrite/paraphrase/neighborhood group labels.

The current runtime contract is full-sequence inference (`use_cache=False`),
which the official evaluator sets. For autoregressive deployment with a KV
cache, keep `use_cache=False` until sticky per-sequence routing across cached
decode steps is implemented; otherwise the subject is absent from one-token
decode calls and the reader gate closes after the prefill.

Key audit fields are:

- `invariants.scope_closed_bit_identical`
- `stage2.reader_scope`
- `audits.training_safe.edited_input_embedding_rows`
- `audits.training_safe.edited_lm_head_rows`
- `scoped_span_edit.evaluation_group_labels_used_by_router` in official output

`kappa_cross` remains in the Stage-1 report as the pre-registered writer-only
diagnostic. It may stay near the old value; once the reader is scoped, its
out-of-scope contribution is zero rather than `kappa * d`.
