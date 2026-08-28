# Embedding-Keyed Sparse Neuron Erasure

This branch tests one precise hypothesis:

> Globally shared embedding rows can write a context-composed key, and a small
> set of existing nonlinear MLP neurons can decode that key into a local
> erasure operation without external routing or LM-head damage.

This is an experiment, not a claim that knowledge has already been erased.
The implementation fails closed before held-out evaluation unless its
training-only mechanism, serialization, norm, and causal checks pass.

## Architecture

```text
ordinary subject token IDs
        |
        v
sparse deltas on existing embedding rows (frozen V3 writer)
        |
        v
frozen early Transformer layers compose the complete context
        |
        v
record-owned existing SwiGLU neurons at one MLP layer
  gate_proj/up_proj rows = nonlinear contextual-code detector
  down_proj columns      = erasure actuator
        |
        v
frozen remaining Transformer + exactly unchanged LM head
```

There is no new vocabulary token, tokenizer expansion, subject-string matcher,
router, retrieval cache, sidecar, LoRA, adapter, constant logit bias, or
LM-head update. After training, the edits are materialized into ordinary model
weights.

### Concrete example

Suppose one edited subject is `Gautier de Coincy`. The writer may modify the
existing rows for `Gautier`, `de`, and `Coincy`. The row for `de` is still
globally shared, so `Melchior de Vogüé` also receives its modified lookup.
The desired separation happens later:

```text
Gautier + de + Coincy -> frozen context composition -> code [+, +, +, +]
Melchior + de + Vogüé -> frozen context composition -> code [0, 0, 0, 0]
```

Four disjoint, low-activation SwiGLU neurons are assigned to the first record.
Their gate/up rows learn the multibit detector; their matching down columns
learn the local erasure residual. The shared `de` row alone is trained not to
activate the group. This is the architectural meaning of “the embedding creates
which-neurons-to-activate information, and the MLP neurons perform the edit.”

## What is frozen and what changes

- Reused Stage 1: ordinary sparse subject embedding-row deltas from the V3
  compositional writer. They are frozen during this experiment.
- New detector: selected rows of one existing MLP's `gate_proj` and `up_proj`.
- New actuator: the matching columns of that MLP's `down_proj`.
- Unchanged: every unselected embedding/MLP parameter, all other Transformer
  parameters, and the complete LM head.

The default 50-record experiment assigns four disjoint neurons per record (200
existing neurons total). The summary reports exact edited scalar counts and the
fraction of base-model parameters they represent.

## Data firewall: no evaluation leakage

The learner in `scripts/mcf_embedding_keyed_neuron_erasure.py` deliberately has
no `--mcf-path`, official-evaluation, paraphrase, neighborhood, retain, PPL,
alias, or adversarial argument.
The launchers also remove all known evaluation-path variables from the learner
and fresh-reload environments; the learner refuses to start if one is present.

It may read only:

1. the locked direct-only training view;
2. the exact split manifest that identifies those 50 direct cases;
3. the exact training-safe context manifest used by the frozen Stage-1 writer;
4. the frozen Stage-1 writer state;
5. disjoint Wikipedia sentences used as generic protection text.

The learner checks all relevant SHA-256 bindings. It recursively rejects
evaluation-only fields in the context artifact and requires the manifest's
access receipt to say:

```json
{
  "official_paraphrases_seen": 0,
  "official_neighborhoods_seen": 0,
  "benchmark_retain_seen": 0,
  "official_ppl_seen": false
}
```

The original MCF file is first opened only by a separate evaluation process,
after the checkpoint has passed training acceptance and a fresh-process reload.
Official paraphrases, neighborhoods, retain examples, and PPL therefore cannot
affect gradients, neuron selection, hyperparameters, early stopping, checkpoint
selection, or retries. Alias, description, adversarial, probe, and relearning
sets are also reserved for post-checkpoint evaluation.

The one-time split builder necessarily reads the source dataset to export the
direct-only view. It is a boundary/sanitization process, not the learner: held-
out fields are absent from its training output. This method reuses the exact
already-frozen V3 artifacts rather than rebuilding or inspecting the source.

## Training stages

### 0. Writer-aware neuron selection

For each record, capture layer activations with the frozen embedding writer on
and off. Candidate neurons must lie in the lowest protected-activation fraction.
The default selection score rewards a stable writer-induced displacement and
penalizes ordinary protected activation. Ownership is disjoint across records.

### 1. Nonlinear contextual-code detector

Train only the selected gate/up rows. Training-safe positives must activate
their owned neuron group; shared-subword, subject-fragment, leave-one-component-
out, unrelated, and cross-record contexts must not. Turning off the embedding
writer must also remove the activation code.

### 2. Local erasure actuator

Freeze the detector and train only the matching down columns. The objective
requires the sensitive answer to lose against the direct reference while
preserving reference NLL, training-safe negatives, protected top-k output
distributions, and writer-off behavior. Every gate/up row and down column has a
hard relative norm cap; this prevents the V4 unbounded-nullspace failure.

The LM head is never optimized and is hash-checked before saving.

## Fail-closed acceptance

A checkpoint is written only if all of the following hold on training-safe data:

- zero direct failures at margin 1;
- zero training-safe positive failures;
- reference-NLL regression at most 0.05;
- detector and actuator hard norm caps respected;
- hook/materialized-weight behavior agrees within native-dtype tolerance;
- the LM head is bit-identical;
- removing the embedding writer breaks at least 50% of direct cases;
- removing the neuron decoder breaks at least 50% of direct cases;
- the strict detector certificate passes.

The fresh-process verifier independently checks serialized selected rows,
selected MLP rows/columns, caps, LM-head digest, and all locked margins before an
official evaluator can run.

## Mandatory ablations

The preregistered registry is
`protocols/mcf_embedding_keyed_neuron_ablation_registry_v1.json`.

The primary run always computes the most important four-way causal ablation on
the same learned checkpoint:

| Configuration | Scientific question |
|---|---|
| Full embedding + neuron | Does the complete mechanism work? |
| Embedding only | Can the writer forget without the MLP decoder? |
| Neuron only | Is the MLP edit merely doing ordinary stand-alone editing? |
| Reconstructed base | Does restoration recover the original behavior? |

The registered training ablations isolate:

- writer-aware versus random dormant-neuron selection;
- learned detector versus no gate/up learning;
- full compositional-negative loss versus no negative/cross loss;
- explicit writer-off dependence versus no writer-off constraint;
- 1, 2, 4, or 8 neurons per record;
- MLP layers 4, 8, 12, or 16.

The SLURM array performs these as training-only jobs and never opens official
evaluation data. That prevents using ablation outcomes as an unofficial
evaluation-driven hyperparameter search.

## Why 50 records first

Fifty is the right first experiment because the mechanism can fail for
fundamental geometric reasons. Spending twice the compute does not make that
test more informative. If—and only if—the primary 50-record setting passes, run
a preregistered 100-record confirmatory experiment with a separately frozen
100-record Stage-1 writer and the same primary hyperparameters. Never reuse the
50-record official results to redesign the 100-record method.

## Running the primary experiment

The expected V3 artifacts are:

```bash
export V3_OUTPUT_DIR=outputs/mcf_compositional_marker_v3_seed1_3b
ls "$V3_OUTPUT_DIR/protocol/training_visible_target_aware_direct.json"
ls "$V3_OUTPUT_DIR/protocol/split_manifest.json"
ls "$V3_OUTPUT_DIR/method/context_manifest.json"
ls "$V3_OUTPUT_DIR/method/stage1_writer.pt"
```

Submit:

```bash
bash scripts/submit_mcf_embedding_keyed_neuron_seed1.sh
```

Or run the SLURM file directly with explicit paths:

```bash
PROJECT_DIR="$PWD" \
MODEL_PATH=/path/to/Llama-3.2-3B-Instruct-clean \
WIKIDATA_DIR=/path/to/wikipedia_sure_50020 \
V3_OUTPUT_DIR=outputs/mcf_compositional_marker_v3_seed1_3b \
OUTPUT_DIR=outputs/mcf_embedding_keyed_neuron_seed1_3b \
bash slurm/run_mcf_embedding_keyed_neuron_seed1_3b.slurm
```

Submit the training-only ablation array separately:

```bash
bash scripts/submit_mcf_embedding_keyed_neuron_ablations.sh
```

After the array finishes, aggregate its training-only evidence without opening
the benchmark:

```bash
python scripts/aggregate_mcf_embedding_keyed_neuron_ablations.py \
  --ablation-root outputs/mcf_embedding_keyed_neuron_ablations_seed1_3b \
  --registry protocols/mcf_embedding_keyed_neuron_ablation_registry_v1.json \
  --out-dir outputs/mcf_embedding_keyed_neuron_ablations_seed1_3b/aggregate
```

## Outputs

- `method/training_firewall_receipt.json`: exact inputs, hashes, and prohibited
  evaluation access;
- `method/neuron_selection_report.json`: owned neurons, scores, activation
  displacement, protected RMS, and selection mode;
- `method/detector_gate_report.json`: per-record positive, negative, and
  writer-off code responses;
- `method/causal_component_ablation.json`: full/embedding-only/neuron-only/base
  training-safe margins;
- `method/embedding_keyed_neuron_state.pt`: exact base and edited sparse values;
- `method/embedding_keyed_neuron_summary.json`: architecture, parameter counts,
  losses, caps, causal evidence, and acceptance;
- `method/post_reload_acceptance.json`: independent serialization replay;
- `official_eval.json`: held-out Eff/Gen/Spe/PPL, created only afterward;
- `comparison/component_ppl_attribution.json`: post-hoc four-way PPL attribution;
- `comparison/comparison.md`: preregistered Base-vs-edited decision table.

## Claim boundary

Passing MCF would support a claim of context-selective factual unlearning under
the tested prompt distribution. It would not, by itself, prove that the fact is
absent from every internal representation. A stronger “robust knowledge
removal” claim additionally needs held-out aliases/descriptions, adversarial
elicitation, latent extraction probes, and relearning attacks—all evaluated
after checkpoint freezing and reported whether they succeed or fail.

Likewise, architectural novelty must ultimately be established against the
current literature, not asserted from code structure alone. The falsifiable
contribution tested here is the *causal combination* of an ordinary sparse
embedding key, nonlinear internally routed existing neurons, an unchanged LM
head, and ablations proving that neither half works alone.
