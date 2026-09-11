# Natural-Writer No-Router Unlearning Baseline

This branch adds a native natural-prompt baseline intended to close the main
train/evaluation mismatch in the extended-token experiment.

## Why this exists

The extended-token control can score perfectly on routed prompts while official
MCF evaluation stays poor because the learned private fact token is absent from
natural evaluation prompts.  The natural-writer baseline removes that routing
dependency completely:

```text
ordinary natural prompt
        |
        v
ordinary tokenizer IDs
        |
        v
Llama hidden states
        |
        v
one sensitivity-selected MLP writer layer
        |
        v
512 retain-quiet / forget-sensitive down-projection channels
        |
        v
rank-16 low-rank edit
        |
        v
merge into ordinary model weights
        |
        v
native Hugging Face checkpoint
```

There is no tokenizer extension, fact-ID injection, runtime router, runtime
guard, sidecar, or LM-head edit.

## Data contract

- Forget records: the exact 50-record ZeroUnlearn-compatible official split,
  seed 1.
- Eff/canonical rewrite prompt: training-visible. This is the natural direct
  prompt on which the forgetting request is defined.
- Gen/paraphrase prompts: never accessed by the learner.
- Neighborhood prompts: never accessed by the learner.
- Official retain records: the exact 1,000 official retain records are reserved
  and excluded from fitting.
- Fitting preservation: 300 different first-half MCF records.
- Development forgetting: four independent authored relation-noun prompt
  families per forget fact.
- Development retention: two independent authored relation-noun prompt
  families per fitting retain fact.

The experiment therefore distinguishes two failure modes cleanly:

1. **Eff poor after training**: the internal edit itself failed.
2. **Eff strong but Gen poor**: the edit fit the requested facts but did not
   generalize semantically to unseen paraphrases.

Neither case can be explained by a missing private token.

## Architecture

The runner first selects one candidate MLP layer using the existing
forget-versus-preservation sensitivity audit. It then computes mean
`down_proj.weight` gradients on fitting-only forget and retain examples and
scores each MLP input channel by

```text
forget gradient L2
-------------------------------
retain gradient L2 + stability floor
```

The top 512 channels are exposed to a rank-16 `StaticEditor` update. Every
other Llama parameter stays frozen during fitting. The edit is trained with
forgetting pressure plus retain NLL/KL protection, merged into the selected
native `down_proj`, saved as an ordinary Hugging Face model, reloaded, and
verified again before the checkpoint is accepted.

This is a sparse internal writer intervention, not an oracle-routed prompt
adapter.

## Extended-token V2.1 fixes on this branch

The branch also fixes three issues in the oracle control:

1. Adam moments are cleared when a row crosses from forgetting to constrained
   abstention, so the first abstention proposal no longer contains
   forgetting-phase momentum.
2. Global checkpoint selection is feasibility-aware: before the threshold it
   minimizes worst answer probability; after the threshold it prioritizes
   abstention while keeping forgetting feasible.
3. V2.1 runs five additional globally feasible gates instead of stopping at the
   first moment every answer view falls below `1e-6`.

The exact `1e-6` boundary is also consistently treated as failing when the
registered criterion is strictly below `1e-6`.

## Train

From `semantic-unlearning/`:

```bash
bash scripts/run_static_overlap_natural_writer.sh \
  /path/to/Llama-3.2-3B-Instruct \
  "$PWD/data/multi_counterfact.json"
```

The default output is:

```text
outputs/static_overlap_natural_writer_v1_seed1/
```

A successful run contains:

```text
training_report.json
writer_layer_localization.json
writer_channel_localization.json
checkpoint/
  config.json
  model*.safetensors
  tokenizer*
  training_manifest.json
  static_edit_export.json
```

Do not run official evaluation unless training produced
`verified_natural_writer_checkpoint`.

## Official MCF evaluation

Use the native checkpoint directly; no injection or routing step is needed:

```bash
bash scripts/evaluate_static_overlap_natural_writer_official.sh \
  outputs/static_overlap_natural_writer_v1_seed1 \
  "$PWD/data/multi_counterfact.json" \
  "$PWD/data/wikidata"
```

This uses:

- forget = 50
- retain = 1000
- seed = 1
- official split
- ordinary natural prompts
- BF16 evaluation

and writes:

```text
outputs/static_overlap_natural_writer_v1_seed1/official_mcf_eval.json
```

## What to compare

The important comparison is not just the average training loss. Compare:

| Stage | Metric |
|---|---|
| Train forget | max/mean answer probability and target-met |
| Authored development forget | same metrics on unseen authored templates |
| Official Eff | direct natural rewrite behavior |
| Official Gen | unseen official paraphrase behavior |
| Official Spe | specificity / neighborhood preservation |
| Official retain Eff/Gen/Spe | held-out retain behavior |
| PPL | broad language utility |

If train and authored development are both near zero but official Eff is not,
inspect the exact canonical prompt/tokenization and export parity. If Eff is
near training but Gen is substantially worse, the remaining problem is
semantic transfer and should be addressed by the internal mechanism or
training-visible authored diversity—not by reintroducing a router.

## Relation to the embedding-keyed neuron work

The repository's `mcf_embedding_keyed_neuron_v3_6_2` line is the more
mechanistically ambitious experiment. It uses a sparse embedding writer plus an
internal layer-27 detector/actuator and already contains much stronger
preservation, hard-tail, recovery, and one-shot official-evaluation contracts.

The natural-writer baseline here is intentionally simpler. Its purpose is to
answer the immediate question:

> Does replacing oracle private-token routing with a native internal writer edit
> substantially close the train-to-official-Eff/Gen gap?

If it does, use it as the clean no-router baseline and continue the stronger
keyed-neuron mechanism separately. If it does not, the result gives a much more
useful diagnosis than the extended-token score because train and evaluation now
share the same ordinary input path.
