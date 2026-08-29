# Embedding-Keyed Sparse-Neuron Conditional Suppression

This experiment tests a narrow mechanism claim:

> A frozen sparse embedding writer can create context-composed information that
> an equally sparse set of existing nonlinear MLP neurons uses to suppress a
> factual answer more locally than either an embedding-only edit or an
> independently trained sparse-MLP editor.

The primary claim is **context-conditional factual suppression with measured
locality on declared distributions**. The protocol does not call the mechanism
knowledge deletion and does not promote a passing MCF score to “unlearning.”

## Concrete architecture

```text
ordinary subject token IDs
        |
        v
frozen sparse deltas on existing embedding rows
        |
        v
frozen Transformer blocks compose token + relation + context
        |
        v
record-owned existing SwiGLU neurons at final block MLP layer 27
  selected gate_proj/up_proj rows = nonlinear contextual detector
  matching down_proj columns      = suppression actuator
        |
        v
frozen later Transformer blocks + bit-identical LM head
```

There is no tokenizer expansion, string matcher, retrieval cache, runtime
router, sidecar, adapter, LoRA, constant logit bias, or LM-head edit. After
training, all changes are ordinary weight values in existing embedding rows and
existing MLP rows/columns.

The implementation does **not** contain a scalar runtime gate of the form
`g = a^T h - tau`. Its auditable detector statistic is the signed mean
absolute activation of each record-owned neuron group—the continuous quantity
that actually multiplies its edited down-projection columns. It does not
subtract a paired Base activation that would be unavailable at inference. The
training floor and off-context threshold are diagnostics, not a fictional
runtime classifier.

## Training-only v1 rejection and v2 revision

The first layer-8 feasibility run was rejected before its official evaluation: its
detector passed 0/50 records, and the subsequently attempted actuator was
unstable. That run is negative development evidence, not a paper result. V2 is
registered before evaluation of a v2 checkpoint and changes five mechanistically
motivated items: it audits/trains the actual absolute neuron gate, places the
decoder at layer 27 where the writer's late-layer code is available, selects a
quieter 20% neuron pool, permits full-row detector reprogramming, and uses a
lower-learning-rate actuator with writer-off preservation every step. A strict
detector failure now stops before actuator training.

This is not described as a blind preregistration: earlier project portability
and scoped-span results motivated the numerical acceptance bars. The stronger
firewall claim is that official prompts are unavailable to the v2 learner and
cannot select or retry its checkpoint.

## What changed after mechanism review

### 1. The 2x2 table is no longer called a necessity proof

The fixed-checkpoint interventions remain useful:

| Fixed fitted checkpoint | What it answers |
|---|---|
| Full embedding + neuron | Does the fitted mechanism work? |
| Embedding only | Does the frozen writer itself suppress the answer? |
| Neuron only | Does this fitted MLP edit still work when its writer is removed? |
| Reconstructed Base | Does exact sparse restoration recover Base? |

These rows diagnose which components one fitted solution relies on. They do not
establish architecture-level necessity.

The decisive control is now `mlp_only_retrained`: a second optimization run
starting from Base with no embedding delta. It uses:

- the same records and training-safe contexts;
- the same seed, MLP layer, neuron count, dormant fraction, steps, learning
  rates, detector floor/off threshold, KL weight, and relative norm caps;
- a strong base-positive-versus-compositional-negative neuron selector, so the
  control is not handicapped by random neuron choice;
- zero embedding edits, verified again after checkpoint reload.

If this independently trained model reaches the same preregistered forgetting
and locality envelope—including the identical 100k retain-tail test—the report
marks the embedding-key necessity claim as falsified. Turning the joint model's
writer off cannot substitute for this run. The converse is deliberately
weaker: if the control fails, the result supports a keyed advantage under the
registered optimization budget, not a theorem that every MLP-only learner must
fail.

### 2. Frozen-writer portability is measured directly

Before selecting or optimizing any neuron, the learner measures the frozen
writer on every training-safe positive context. It refuses decoder construction
unless at least 95% of prompts globally and 80% for the worst record exceed the
predeclared amplitude 4.5. This is a feasibility check, not held-out evidence.

`audit_mcf_frozen_writer_portability.py` reconstructs Base and writer-only
states and measures the original Stage-1 marker projection
`Q^T(h_writer - h_base)` on every official forget rewrite and paraphrase. The
threshold comes from the already-frozen `stage1_writer_report.json`; it is not
fit to official prompts.

Acceptance requires:

- at least 95% complete marker projections globally; and
- at least 80% for the worst record.

The audit is an empirical diagnostic of the claimed pathway. It is not stated
as a mathematical upper bound on every possible nonlinear decoder.

Official paraphrases stay behind the data firewall until **both** the proposed
checkpoint and the independently trained no-writer checkpoint are frozen. If
portability fails, the run is rejected. Any writer redesign must be a new
preregistered experiment; the failed official prompts may not be fed back into
this checkpoint.

### 3. Locality is a tail criterion, not a single example

`audit_mcf_embedding_keyed_neuron_retain_tail.py` reconstructs Base and the
complete edit for unique official retain prefixes from five groups:

- rewrite;
- paraphrase;
- neighborhood;
- attribute auxiliary; and
- generation auxiliary.

With seed 1 and 9,000 retain records, the current MCF file yields 126,864 unique
prefixes. The audit refuses to run with fewer than 100,000.

For every prefix it reports:

- maximum absolute signed record-owned gate activation;
- whether the next-token argmax changes;
- restricted top-k KL; and
- maximum absolute top-k log-probability change.

The registered empirical bar is 24/13,000, inherited from the earlier exact
scoped-span retain-auxiliary audit. The primary response-event rate and top-1
change rate must not exceed that rate, and the one-sided 95% Wilson upper bound
for the response-event rate must also be below it. The report explicitly notes
that the old exact matcher and the learned neuron response are not identical
random mechanisms; 24/13,000 is a demanding comparison bar, not a theorem.

### 4. Existing-neuron function is profiled before editing

The old fixed slice could fill a 1,024-prompt budget with early training cases
and silently omit broad-corpus prompts. Selection now uses 8,192 prompts with:

- 4,096 corpus prompts reserved first;
- all record groups represented by round-robin allocation; and
- remaining capacity backfilled without duplicates.

For every selected neuron the selection receipt reports baseline RMS, p99,
p999, maximum activation, threshold exceedance rate, original down-column norm,
and a residual-contribution bound. This does not prove that a neuron has no
unseen function. The 100k post-freeze effect tail measures the remaining risk.

### 5. Frequency shrinkage is now a mechanism factorial

The claim is no longer “we tuned a smaller cap for common tokens.” The
registered writer conditions are:

| Writer condition | Row cap | Frequency alpha |
|---|---:|---:|
| `frequency_capped` | 8 | 0.15 |
| `uniform_same_cap` | 8 | 0 |
| `uniform_raised_cap` | 16 | 0 |

All three fix `max_subject_token_frequency=1e9`, so common rows are retained;
the factor changes attenuation, not eligibility.

Each writer is retrained from scratch, saved with `--stage1-only`, and passed
through the same neuron-decoder protocol. Every checkpoint is evaluated with
the decoder on (`full_embedding_plus_neuron`) and off (`embedding_only`).
Forget cases are reported separately for rare, medium, and common selected
subject rows.

The mechanism prediction is predeclared: removing and raising the frequency cap
should increase common-token leakage much more for embedding-only than for the
full contextual decoder, while the raised-cap full model remains inside the
primary retain-Spe and PPL margins. The common-token stratum is always shown;
an aggregate mean cannot hide bimodality.

### 6. The claim is fixed before latent and relearning tests

Two mandatory post-freeze endpoints are reported:

- `audit_mcf_embedding_keyed_neuron_latent_recovery.py` applies a fixed final
  norm + unchanged-LM-head logit lens at every fourth block downstream of the
  edited MLP. It records `fact_recoverable=true` if the final output still
  prefers the sensitive target or an intermediate layer recovers that
  preference substantially above the final output.
- `audit_mcf_embedding_keyed_neuron_relearning.py` gives an
  architecture-aware attacker the sensitive direct facts and allows additive
  updates at every sparse site touched by the method. Official Eff/Gen are
  measured at fixed steps 0, 1, 2, 4, 8, 16, 32, and 64. Fast recovery records
  `fact_recoverable=true`.

A positive result on either test reinforces the conditional-suppression
framing. A negative finite probe or attack does not prove universal absence.
Each endpoint must first recover the fact from reconstructed Base as a positive
control; otherwise the endpoint is marked incomplete. The final report never
automatically licenses a knowledge-deletion claim.

## Data firewall and ordering

The neuron learner has no MCF, official paraphrase, neighborhood, retain, PPL,
alias, adversarial, latent, or relearning input. It reads only:

1. the locked direct-only training view;
2. the exact split manifest;
3. the training-safe context manifest;
4. the frozen Stage-1 writer state; and
5. Wikipedia documents 20 onward for selection/protection.

Documents 0:20 remain reserved for official PPL. Relevant hashes and zero-
access receipts are checked. The main launcher freezes and reload-verifies both
the proposed and no-writer checkpoints before a separate process opens the
original MCF file.

Official results can reject a frozen configuration, but cannot choose a retry,
hyperparameter, neuron, threshold, or checkpoint inside the same registered
run.

## Registered acceptance

The registry is
`protocols/mcf_embedding_keyed_neuron_ablation_registry_v1.json`. The legacy
filename is retained for launcher compatibility; its current schema is 3 and
its protocol is `mcf_embedding_keyed_sparse_neuron_suppression_v2`.
The paper-level report requires all of the following:

- primary Eff = 0 and Gen = 0;
- forget and retain specificity inside the declared 0.2-point margins;
- retain Eff/Gen changes at most 1 point and PPL change at most 5%;
- strict training gate, norm, materialization, LM-head, and fresh-reload checks;
- within-checkpoint writer and neuron dependence;
- a budget-matched independently trained no-writer model;
- the no-writer model does not meet the same forgetting/locality envelope;
- both models have complete 100k retain-tail receipts;
- frozen-writer portability passes;
- the 100k retain response/effect tail passes; and
- every post-freeze receipt is bound to the same seed and official split; and
- latent-recovery and relearning endpoints have valid reconstructed-Base
  positive controls and are present whether favorable or not.

The registry fixes seed 1 as development and seeds 2 and 3 as confirmatory with
unchanged hyperparameters. A 100-record run is confirmatory only after a
separately frozen 100-record writer/context artifact exists.

## Running the complete seed-1 pipeline

Expected frozen V3 inputs:

```bash
export V3_OUTPUT_DIR=outputs/mcf_compositional_marker_v3_seed1_3b
ls "$V3_OUTPUT_DIR/protocol/training_visible_target_aware_direct.json"
ls "$V3_OUTPUT_DIR/protocol/split_manifest.json"
ls "$V3_OUTPUT_DIR/method/context_manifest.json"
ls "$V3_OUTPUT_DIR/method/stage1_writer.pt"
ls "$V3_OUTPUT_DIR/method/stage1_writer_report.json"
```

Submit the proposed model, full retrained control, and all post-freeze audits:

```bash
bash scripts/submit_mcf_embedding_keyed_neuron_seed1.sh
```

For an interactive allocation, use the argument-taking wrapper rather than a
multi-line shell substitution. It refuses stale/missing V3 inputs and existing
output directories:

```bash
mkdir -p slurm_logs
bash scripts/run_mcf_embedding_keyed_neuron_v2_manual.sh \
  /scratch/yl258/kp759/ul-a8864ff/semantic-unlearning/outputs/mcf_compositional_marker_v3_seed1_3b \
  outputs/mcf_embedding_keyed_neuron_v2_seed1_3b \
  2>&1 | tee slurm_logs/mcf_embedding_keyed_neuron_v2_seed1_manual.log
```

The job is intentionally budgeted for two full training runs. The result report
does not silently pass with a missing control or missing audit.

Training-only secondary ablations:

```bash
bash scripts/submit_mcf_embedding_keyed_neuron_ablations.sh

python scripts/aggregate_mcf_embedding_keyed_neuron_ablations.py \
  --ablation-root outputs/mcf_embedding_keyed_neuron_ablations_seed1_3b \
  --registry protocols/mcf_embedding_keyed_neuron_ablation_registry_v1.json \
  --out-dir outputs/mcf_embedding_keyed_neuron_ablations_seed1_3b/aggregate
```

Frequency-cap factorial:

```bash
bash scripts/submit_mcf_context_gating_frequency_factorial.sh

python scripts/aggregate_mcf_context_gating_frequency_factorial.py \
  --root outputs/mcf_context_gating_frequency_factorial_seed1_3b \
  --registry protocols/mcf_embedding_keyed_neuron_ablation_registry_v1.json \
  --out-dir outputs/mcf_context_gating_frequency_factorial_seed1_3b/aggregate
```

## Important outputs

- `method/neuron_selection_report.json`: stratified prompt-bank receipt and
  selected-neuron baseline activation/function tails;
- `method/writer_preflight_report.json`: frozen-writer training-safe coverage
  measured before decoder construction;
- `method/causal_component_ablation.json`: explicitly labeled
  within-checkpoint intervention;
- `method/embedding_keyed_neuron_state.pt`: exact base/edited sparse weights,
  ownership, signs, thresholds, and writer mode;
- sibling `*_mlp_only_retrained/method/...`: independently trained control;
- sibling `*_mlp_only_retrained/comparison/retain_tail_100k.json`: identical
  tail audit for the control;
- `comparison/writer_portability.json`: frozen Stage-1 marker completeness;
- `comparison/retain_tail_100k.json`: response and actual-effect tails;
- `comparison/official_components.json`: official fixed-component metrics and
  frequency strata;
- `comparison/latent_recovery.json`: downstream residual extraction endpoint;
- `comparison/relearning.json`: fixed-budget sparse-site relearning curve; and
- `comparison/comparison.md`: final claim-gated decision report.

## Publication boundary

This repository now contains a coherent, falsifiable architecture and a
review-resistant evaluation protocol. It does not contain the GPU results that
would make the contribution publishable. An ICLR-quality paper still requires
the registered seed-1 result, unchanged confirmatory seeds, honest negative
results, literature comparison, and uncertainty/error analysis. Code structure
alone is not evidence that the mechanism works.
