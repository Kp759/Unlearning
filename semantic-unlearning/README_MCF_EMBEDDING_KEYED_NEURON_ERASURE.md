# Embedding-Keyed Contextual Residual Suppression

This experiment tests a narrow mechanism claim:

> A frozen sparse embedding writer can create context-composed information that
> a sparse downstream contextual branch can use to suppress a factual answer
> without asking highly shared embedding or LM-head rows to carry the erasure.

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
V3.2-initialized layer-27 gate/up features (4 per record)
        |
        v
canonical exact-prompt multi-label targets
        |
        v
100-update equal-record, per-record-tailed detector repair
        |
        v
clipped internal gate maps the registered response gap
  <= 0.200001 -> 0; >= 0.249999 -> 1
        |
        v
separate detector-disjoint Base actuator features
  nested widths 4 / 8 / 16 per record
        |
        v
threshold-gated sparse residual through learned down_delta at cap 1.50
  Base MLP path, detector tensors, and original down columns remain untouched
        |
        v
frozen later Transformer blocks + bit-identical LM head
```

There is no tokenizer expansion, subject-string matcher, retrieval cache,
external router, adapter, LoRA, constant logit bias, or LM-head edit. V3.5
introduced a declared **internal activation gate** and explicit additive
residual branch. It is therefore not described as an ordinary existing-neuron
weight materialization. V3.5 failed its complete cached-context branch-gate
certificate before fitting the residual. V3.5.1 diagnosed its exact lone
collision without constructing an optimizer. V3.5.2 repaired writer-off scope
but exposed one exact prompt carrying mutually contradictory role-relative
labels. V3.5.3 removed that contradiction through canonical multi-label prompt
semantics, but its complete-update global tails collapsed valid positives to a
29/50 owner gate while negative and writer-off certificates stayed clean.
V3.5.4 preserved the corrected labels, removed global tails from optimization,
and reached a 50/50 detector plus an exact all-cell threshold gate. Its
four-feature actuator remained cap-bound: 147/200 columns saturated at 1.50 and
the positive certificate was not reached. V3.5.5 therefore freezes that exact
detector and varies actuator width through nested, detector-disjoint banks of
4, 8, and 16 Base features per record while retaining the 1.50 per-column cap.
Because width also changes the available aggregate norm budget, this is not
described as a pure rank intervention. Every feasibility fit is discarded; the
protocol still cannot save or evaluate a checkpoint.

## Training-only development through V3.5.5

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
firewall claim is that official prompts are unavailable to each registered
learner and cannot select or retry its checkpoint.

The first v2 attempt then stopped at its writer-portability preflight before
neuron selection. It passed globally (548/567 prompts, 96.65%) but only 48/50
records met the 80% record floor. A training-data lineage audit found two
critical confounds: the nominal V3 writer had resumed the V2 state with zero
optimization events, and the newer context manifest appended free-form V7
surrogates that changed the relation or injected unsupported attributes in the
two failed records. No official evaluation prompt had been opened.

V3 therefore preserves v1 and v2 outputs as failed historical diagnostics and
requires a new upstream writer. Its positive contexts are uniformly the direct
prompt, explicit hand-authored relation-ID alternatives, and unrelated-corpus-
prefix variants. Generic relation fallbacks and external free-form surrogates
are forbidden. The writer must start from Base, run 1,200 steps, and bind its
manifest, state, report, and nonempty optimization log by hash. The 4.5, 95%,
and 80% portability thresholds are unchanged. The writer and decoder also bind
the resolved Base-model path, frozen-Transformer fingerprint, and SHA-256 of
the selected Base embedding rows.

The clean V6.2 writer then passed all 346/346 registered training contexts, but
the V3 detector failed closed at 0/50 records. Positive response was the clear
bottleneck: 49/50 records missed the 0.25 floor, with median
`positive_min = 0.0029`; negatives all passed with maximum absolute response
0.0752; and 13/50 writer-off tails exceeded 0.20. Actuator training did not
start and official evaluation remained unopened. The registry preserves this
as a training-only rejection rather than overwriting it.

V3.1 changes only detector optimization. It keeps the V6.2 writer, deterministic
layer-27 neuron selection, four neurons per record, 200 disjoint neurons, norm
caps, 0.25/0.20 gate, 1,000 optimizer updates, actuator, and firewall fixed. It:

- caches the frozen layer-27 MLP input for every writer-on positive, writer-on
  negative, and writer-off positive context;
- accumulates bounded four-record microbatches across all 50 records before one
  clip, Adam step, and relative-norm projection;
- trains on every positive and negative context on every update;
- uses an equal-record prompt mean plus worst-two positive shortfall; and
- penalizes negative, cross-record, and writer-off responses only for their
  mean plus worst-two squared excess above the unchanged 0.20 gate.

The selected MLP input is upstream of the rows being optimized and the selected
down projection is disabled during detector training, so the cache does not
approximate or stale the learned detector computation.

V3.1 improved the unchanged detector certificate from 0/50 to 45/50 using the
exact same 200-neuron ownership hash
`acc3cc05868483f6c40a8909fca064b59c4ec4d000a76cf1ece6c3e818c750d1`.
It remains a rejected training-only run: actuator training did not start and
official evaluation saw zero prompts. Three writer-off values and one negative
value exceeded 0.20 by only `1.79e-8` to `4.77e-8`; case 10472 was the one
substantive miss, with `positive_min = 0.24717252` (0.00282748 below 0.25).
Those observations do not retroactively change V3.1's exact comparison rule.

V3.2 changes only optimization safety margins and numerical comparison policy,
all registered before its run:

- positive optimization target: 0.30; certificate floor: 0.25 unchanged;
- negative/writer-off optimization ceiling: 0.15; certificate ceiling: 0.20
  unchanged;
- certificate absolute comparison tolerance: `1e-7`;
- exact V3/V3.1 primary neuron-ownership hash required;
- full-context gates saved before final Adam, after Adam, after norm projection,
  and in a fresh final replay; and
- all 1,000 detector optimizer-step rows saved to a complete JSON log.

V3.2 passed that unchanged detector certificate for all 50 records, so the
actuator was allowed to start. The actuator then failed decisively after its
registered 2,000 stochastic updates: 41/50 materialized direct contexts and
294/346 positives still failed, minimum margin was -9.8281, and maximum
reference-NLL regression was 0.6484 against a 0.05 tolerance. Turning off the
neuron decoder also produced 41 direct failures, so the learned down columns
did not improve the writer-only direct-failure count. Official evaluation saw
zero prompts and no checkpoint was saved.

V3.3 redesigns only Stage 2 and imports the exact passed V3.2 `gate_delta` and
`up_delta` tensors. It hash-binds the rejected V3.2 source, replays the fresh
50/50 detector certificate with a separately registered `1e-5` cross-process
replay tolerance, refuses to import the failed `down_delta`, and starts the
actuator at exact zero under the unchanged 0.50 relative norm cap. The fresh
scientific detector certificate itself retains its stricter `1e-7` comparison
tolerance.

Before the full objective, V3.3 ran its registered 100-update positive-only
capacity diagnostic. Every update accumulated all 50 records and all 346
positive contexts, using an equal-record prompt mean plus worst-two squared
margin shortfall. It failed broadly: 40/50 direct contexts and 300/346
positives remained below the +1 margin, the minimum margin was -8.75, and at
least one down column reached the 0.50 cap by step 15. Full actuator training
did not start, no checkpoint was saved, and official evaluation saw zero
prompts.

V3.3 also measured `writer_off_nll_abs_max = 4.8125` and maximum reference-NLL
regression 2.375. Stage 2a did not optimize either preservation term, so these
are not full-objective failures. The writer-off value is nevertheless a direct
structural-selectivity warning: it is 96.25 times the eventual 0.05 tolerance
on the very positive prompts whose embedding writer was disabled.

V3.4 is therefore only a training-safe reachability and selectivity diagnostic.
It hash-binds the exact V3.3 rejection and audits the frozen detector before any
actuator fit at four levels: owned signed-group response, individual owned
activations, owned activation-vector norm, and the full selected-neuron-vector
norm. Paired writer-on/off ratios use a registered `1e-8` denominator floor.
The p10 warning level of 100 is explicitly heuristic—not an acceptance gate—
because activation ratios do not mathematically equal downstream NLL gain.

V3.4 then runs all five registered caps independently:

```text
0.50, 0.75, 1.00, 1.50, 2.00
```

Every cap starts from bit-exact zero `down_delta` and a fresh AdamW optimizer,
runs exactly 100 globally balanced positive-only updates over all 346 contexts,
receives one clip and one down-only norm projection per update, and ends with a
fresh full-context behavioral audit. It records every step incrementally,
every per-record minimum margin, all 200 achieved relative norms and saturation
flags, writer-off NLL drift, and the exact writer-on/off layer-output residual
created by `down_delta`. Every fitted tensor is then zeroed and discarded.

The preregistered reachability choice is the smallest cap with zero direct and
zero all-positive failures. Writer-off selectivity at 0.05 is reported as a
separate structural diagnostic and cannot silently alter that selection rule.
If no positive-reachable cap independently stays inside the 0.05 writer-off
band, V3.4 records a mechanism-readiness rejection even when positive
reachability itself succeeds.
V3.4 never runs the preservation objective, saves a checkpoint, or opens
official evaluation—even if a cap is reachable. If no cap through 2.0 passes,
the registered conclusion is that the fixed 200-neuron layer-27 actuator is
insufficient within the tested norm budget.

The completed V3.4 sweep isolated the actual bottleneck. Cap 1.50 was the
smallest positive-reachable value: its loss reached zero with 34/200 columns
saturated, while cap 2.00 reached the same objective without saturation. Thus
four features per record have enough positive-only actuator capacity. However,
the frozen detector's writer-on/off response ratio was only 2.1545 at p10 and
3.2468 at the median. More decisively, writer-off target NLL moved by 0.25 even
with `down_delta` bit-exact zero—five times the 0.05 tolerance. The reason is
structural: V3.4 replaced gate/up activations, and those activations still
flowed through the selected neurons' original Base down columns.

V3.5 fixes that specific mechanism rather than increasing rank, steps, or cap:

1. the entire ordinary Base MLP output stays untouched;
2. the exact frozen V3.2 gate/up tensors are evaluated only as detector
   features;
3. the registered 0.20/0.25 certificate gap becomes an explicit clipped
   internal gate with a `1e-6` guard;
4. only `gate * detector_activation * down_delta` is added as a separate
   residual; and
5. positive-only feasibility is rerun at the already-established smallest cap
   1.50 for exactly 100 globally balanced updates.

Before fitting, V3.5 requires all owner-positive gates to be one and all
writer-off, negative, and cross-record gates to be zero on the complete cached
training context set. It then audits exact-zero identity on both writer-on and
writer-off NLLs. A separate decomposition temporarily zeros the 200 original
Base down columns with Base gate/up and writer off, measures all 346 positives,
and restores the columns bit-exactly. That decomposition is diagnostic only;
it cannot reselect neurons or change acceptance. The fitted residual is also
audited through full forward passes because last-token detector certificates
alone do not constrain gates at every earlier sequence position.

V3.5 remains training-only. Every fitted `down_delta` is discarded; full
preservation training, checkpoint creation, and official evaluation are
prohibited regardless of the outcome. A passing V3.5 result licenses a newly
preregistered successor preservation experiment, not retrospective evaluation;
the observed V3.5 result did not license one.

The observed V3.5 run stopped before any actuator fit. All 346 owner-positive
gates were exactly one, all 16,954 positive non-owner gates and all 23,250
negative gates were exactly zero, and 17,299 of 17,300 writer-off gates were
exactly zero. One writer-off gate was 0.5735465884208679, associated with source
case 10803. The aggregate artifact did not record the offending context index
or detector group, so it cannot establish whether the owner detector or a
different record's detector fired.

V3.5.1 is therefore a read-only forensic replay. It hash-binds the rejected
V3.5 artifacts, imports the exact frozen V3.2 gate/up tensors, retains the
unchanged 0.20/0.25 boundaries, and recomputes every raw signed response before
clipping. It records the source prompt and provenance, source context index,
detector group and case ID, owner/non-owner status, and exact-prompt duplicates
for each violation. It must reproduce exactly one nonzero writer-off gate from
case 10803. Detector and actuator optimizer construction, threshold calibration,
checkpoint creation, and official evaluation are all prohibited.

The completed V3.5.1 replay resolved the ambiguity. The only branch violation
was a non-owner collision:

```text
writer-off source:   case 10803, context 4
firing detector:     case 17353, group 30
owner group:         false
raw signed response: +0.2286771833896637
clipped gate:        0.5735465884208679
```

Thus V3.2's owner-wise writer-off certificate was internally correct but too
narrow for V3.5's branch, which consumes all 50 detector groups. This does not
support a broad cross-talk claim: every positive cross-group cell, every
negative cell, and 17,299/17,300 writer-off cells were already exactly at the
desired gate endpoint.

V3.5.2 fixes only that demonstrated training/certificate scope mismatch. It:

1. imports the exact frozen V3.2 gate/up tensors and verifies their initial
   50/50 owner-wise certificate;
2. hash-binds the exact V3.5.1 source/context/detector diagnosis;
3. runs exactly 100 globally accumulated detector updates at the unchanged
   `0.001` learning rate and `1.0` relative norm cap;
4. retains the 0.30 owner-positive target plus the already-successful global
   positive-cross and negative protection losses;
5. changes writer-off training from one owner column to all 50 group columns
   for every context, with equal source-record mean plus worst-two squared
   excess above the unchanged 0.15 training target; and
6. keeps the scientific 0.25/0.20 certificate and clipped branch thresholds
   unchanged.

The repair is identity-agnostic: case IDs 10803 and 17353 license the revision
through the frozen forensic receipt but never index or weight the loss. The
fresh all-cell gate must have 346/346 owner positives at one and zero violations
across 16,954 positive-cross, 23,250 negative, and 17,300 writer-off cells.
Only then may the already-selected cap 1.50 receive the same 100-update
positive-only feasibility fit. That fitted `down_delta` is always discarded.
Full preservation training, checkpoint creation, threshold calibration, and
official evaluation remain prohibited regardless of outcome.

The completed V3.5.2 run repaired all 17,300 writer-off cells to the stricter
0.15 training ceiling, but finished at 49/50 owner records. Its two remaining
all-cell violations were not independent model errors. They had the identical
exact-prompt SHA-256
`9a4070c81368070d9ee1383958c18109bf7af90ee59042b3132b7a51e9d6ca38`:

```text
positive role: case 10472, context 1, detector 10472 -> response 0.22039907
negative role: case 19763, context 4, detector 10472 -> response 0.21156451
```

The same exact prompt was required to activate detector 10472 in its registered
positive occurrence and to suppress that detector when the prompt appeared as
case 19763's record-relative negative. No deterministic detector can satisfy
both labels on one canonical hidden state. This is a labeling contradiction,
not evidence that the 0.25/0.20 response thresholds need relaxation.

V3.5.3 fixes the label semantics rather than deleting either occurrence or
tuning a threshold:

1. every exact prompt is canonicalized once and all duplicate occurrences
   reuse one bit-identical cached hidden state;
2. its active detector set is the union of all records for which that exact
   prompt is a registered positive;
3. a record-relative negative keeps its source owner inactive, but does not
   erase a valid positive label belonging to another record;
4. a prompt that is both positive and negative for the **same** record remains
   an unrecoverable manifest error and fails closed;
5. every active label trains toward 0.30, every inactive writer-on label and
   every writer-off label trains within 0.15, while the scientific 0.25/0.20
   certificate remains unchanged; and
6. V3.5.3 added complete-update worst-two terms to all components.

That last change was not benign. With the existing component coefficients, the
positive, inactive-cross, source-negative, and writer-off global tails entered
the one clipped update at effective weights 1, 2, 5, and 10. V3.5.3 retained
clean negative and writer-off certificates but only 29/50 owner-positive
records passed—a broad quiet-solution signature rather than a label-semantics
failure.

V3.5.4 changes only the loss geometry implicated by that result. Canonical
hidden reuse and multi-label targets are unchanged. Complete-update tails are
still reported as diagnostics but have optimization weight zero. Equal-record
means and per-record worst-two losses remain active for positives, owner
negatives, inactive cross-labels, and every writer-off detector group. A
non-optimizing step-1 gradient audit records each weighted component norm
before the ordinary total backward pass.

The known prompt and case IDs only hash-bind the V3.5.2 diagnosis. They never
index or upweight the repair loss. The mandatory post-repair certificate now
checks positive source owners, every active prompt label, every inactive
prompt label, source-negative owners, and every writer-off cell before the
discarded cap-1.50 actuator fit may begin.

V3.5.4 then passed the detector certificate for all 50 records and mapped every
active label to gate 1 and every inactive/writer-off label to gate 0. Its
isolated four-feature-per-record actuator nevertheless plateaued at cap 1.50,
with 147 of 200 columns saturated. That result establishes neither a detector
failure nor an intrinsic impossibility: the detector features had also been
forced to serve as the actuator basis.

V3.5.5 makes that separation concrete. The exact V3.5.4 gate/up tensors are
reproduced by deterministic replay and verified by tensor hash. A separate
actuator bank is selected from unmodified Base SwiGLU features, excludes all
200 detector neurons, is globally disjoint, and is nested so width 4 is a
prefix of width 8, which is a prefix of width 16 for every record. Native arms
hold the per-column relative cap at 1.50 and select the smallest width with zero
direct and positive failures. Widths 8 and 16 also receive matched width-4
per-record Frobenius-budget controls; these controls diagnose rank versus
aggregate norm budget and cannot select the width. Every arm starts at exact
zero with fresh optimizer state, runs exactly 100 all-record/all-positive
updates, receives a fresh full-context audit, and is then discarded.

The native width-16 arm was the smallest arm to reach every positive margin,
ending with zero shortfall and 34/800 saturated columns. The matched width-4
aggregate-budget controls at widths 8 and 16 both failed. The licensed V3.6
configuration is therefore the combined width-16, cap-1.50-per-column design;
the V3.5.5 evidence does not attribute the gain to actuator rank alone.

## Development-retain shared-row exposure

A separate audit consumed 9,438 first-half development-retain records after
excluding all 1,000 reserved official-retain IDs. These records must never be
described later as blind evaluation. The audit found:

- 198/234 (84.6%) embedding rows actually edited by V6.2 also occur in the
  development-retain prompts;
- 5,762/9,438 (61.05%) development-retain prompts contain at least one actual
  edited row;
- 199/236 (84.3%) forget-subject token IDs are reused; and
- 49/50 forget subjects contain an actually edited row seen in development
  retain.

This establishes shared-parameter exposure, not causal retain degradation.
It motivates the architectural boundary: the embedding edit may be a small
marker, but token identity cannot itself decide which fact to erase. None of
the development IDs, overlap counts, token frequencies, or edit norms enters
V3.5/V3.5.2/V3.5.3/V3.5.4/V3.5.5 selection, optimization, acceptance, or retry.

The reproducible follow-up also measures whether the largest actual row edits
land on the most reused tokens:

```bash
python scripts/audit_mcf_embedding_writer_shared_row_exposure.py \
  --model-path "$MODEL_PATH" \
  --mcf-path data/multi_counterfact.json \
  --split-manifest "$WRITER/protocol/split_manifest.json" \
  --writer-state "$WRITER/method/stage1_writer.pt" \
  --output "$AUDIT_DIR/development_retain_shared_row_exposure_v2.json"
```

It reports Pearson correlation against raw and `log1p` prompt frequency,
Spearman correlation, per-row edit norms, and the largest
frequency-times-norm exposures. `--historical-lm-head-state` optionally runs
the analogous old LM-head-row analysis, explicitly labeled as a lexical proxy
rather than a behavioral exposure metric.

To add the case-ID binding to the preserved historical V3 rejection without
changing its gate JSON:

```bash
python scripts/report_mcf_detector_gate_cases.py \
  --gate "$DECODER_OUT/method/detector_gate_report.json" \
  --training-visible \
    outputs/mcf_compositional_marker_v6_2_clean_seed1_3b/protocol/training_visible_target_aware_direct.json \
  --out "$DECODER_OUT/method/detector_gate_case_report.tsv"
```

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

The clean V6 writer reached 339/346 (97.98%) globally but failed two
record-level gates at 5/7, so decoder construction and official evaluation were
correctly refused. V6.1 kept the contexts and thresholds fixed, trained on all
positives for each sampled record, and added a worst-two squared-shortfall
term. It reached 340/346 (98.27%), but two different records failed at 3/7 and
5/7. This is consistent with record-local updates redistributing margin, but
the moved failure identities do not prove that cause.

V6.2 therefore accumulates gradients from all 50 records, in 17 bounded
microbatches of at most three records, before each Adam update. The non-KL
objective is normalized as an equal record mean and KL as a global prompt
mean. The top-64 KL is evaluated from those exact frozen LM-head rows without
materializing the full vocabulary logits. All data, markers, loss weights,
learning rate, row caps, 1,200 optimizer updates, and 4.5 / 95% / 80%
acceptance thresholds remain fixed. This is not a
compute-matched comparison: V6.2 has 60,000 record exposures versus V6.1's
3,600, a 16.67x increase. A V6.2 pass would establish that the globally
balanced engineering configuration works; an exposure-matched control or
direct gradient-conflict audit is still required before claiming that
cross-record interference was causally established.

The context manifest also audits the two concrete sharing pathways: selected
embedding rows owned by multiple records and positive prompts that contain
selected rows owned by another record. Incidence supports the possibility of
cross-record coupling; it is not reported as a gradient-conflict measurement.
The Stage-1 report separately measures initial and final pairwise cosines for
per-record write-only and full-objective gradients. A negative cosine is an
observed local conflict, while causal attribution of V6.1's failures remains
conditional on the V6.2 intervention result.

The upstream acceptance receipt means artifact integrity **and** an exact
fresh-Base replay of the same pre-decoder gate. Count-derived fractions may be
serialized in float32 and are checked within `1e-7`; counts, thresholds, and
Boolean decisions are still recomputed. An integrity-only receipt cannot
authorize the neuron run.

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
4. the from-Base, relation-templates-only Stage-1 writer state, report, and
   nonempty hash-bound optimization log; and
5. the hash-bound training-only artifacts from the rejected V3.2 run whose
   detector passed 50/50 records;
6. the hash-bound V3.4 reachability-pass/selectivity-reject artifacts;
7. the hash-bound V3.5 single-cell gate rejection; and
8. the hash-bound V3.5.1 exact non-owner collision forensics;
9. the hash-bound V3.5.2 duplicate-prompt contradiction;
10. the hash-bound V3.5.3 positive-collapse rejection;
11. the hash-bound V3.5.4 detector-pass/actuator-reachability rejection; and
12. the hash-bound V3.5.5 discarded-fit width-16 mechanism-readiness pass; and
13. Wikipedia documents 20 onward for deterministic ownership, actuator-bank
    replay.

The separate 9,438-record development overlap audit reads the original MCF
source, but the learner does not receive its path, records, token frequencies,
or per-row results. Only its already-observed aggregate evidence is recorded in
the public registry as architecture motivation. It does not select the V3.5.1
collision, alter its diagnosis, enter the V3.5.2, V3.5.3, or V3.5.4 repair
loss, select any V3.5.5 actuator feature or width, or enter the V3.6
preservation objective.

Documents 0:20 remain reserved for official PPL. Relevant hashes and zero-
access receipts are checked. V3.6 can freeze one training-only candidate state
only after every registered learner-side gate passes. A separate, future
hash-bound process must reload that candidate before it may open the original
MCF file.

Official results can reject a frozen configuration, but cannot choose a retry,
hyperparameter, neuron, threshold, or checkpoint inside the same registered
run.

## Registered acceptance

The registry is
`protocols/mcf_embedding_keyed_neuron_ablation_registry_v1.json`. The legacy
filename is retained for launcher compatibility; its current schema is 17 and
its protocol is `mcf_embedding_keyed_sparse_neuron_suppression_v3_6`.
V3.5.5 was incapable of reaching paper-level acceptance because it discarded
every fitted actuator arm, saved no checkpoint, and opened no official prompts.
V3.6 freezes the exact detector and selected width-16 bank, trains only on
registered training-safe evidence, and saves a candidate state only if every
forgetting and preservation gate passes. It still cannot open official prompts.
The later official checkpoint protocol requires all of the following:

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

## Running V3.6 width-16 full preservation

First build the clean Stage-1 writer from Base. This opens no official
evaluation probes and refuses any existing output path:

```bash
## Batch submission
bash scripts/submit_mcf_compositional_marker_clean_stage1_seed1.sh

## Or, inside an interactive allocation
mkdir -p slurm_logs
bash scripts/run_mcf_compositional_marker_clean_stage1_manual.sh \
  outputs/mcf_compositional_marker_v6_2_clean_seed1_3b \
  2>&1 | tee slurm_logs/mcf_compositional_marker_v6_2_clean_seed1_manual.log
```

V3.6 requires all preserved V3.5.5 inputs plus the completed V3.5.5
mechanism-readiness output. It reconstructs the exact detector and selected
width-16 actuator bank, starts the actuator from bit-exact zero, retains the
positive-warm-start optimizer state, and runs 200 globally balanced full-
preservation updates. The wrapper rejects an existing output directory:

```bash
bash scripts/run_mcf_embedding_keyed_neuron_v3_6_manual.sh \
  outputs/mcf_compositional_marker_v6_2_clean_seed1_3b \
  outputs/mcf_embedding_keyed_neuron_v3_2_seed1_3b_aws_v6_2 \
  outputs/mcf_embedding_keyed_neuron_v3_4_seed1_3b_aws_v6_2 \
  outputs/mcf_embedding_keyed_neuron_v3_5_seed1_3b_aws_v6_2 \
  outputs/mcf_embedding_keyed_neuron_v3_5_1_seed1_3b_aws_v6_2 \
  outputs/mcf_embedding_keyed_neuron_v3_5_2_seed1_3b_aws_v6_2 \
  outputs/mcf_embedding_keyed_neuron_v3_5_3_seed1_3b_aws_v6_2 \
  outputs/mcf_embedding_keyed_neuron_v3_5_4_seed1_3b_aws_v6_2 \
  outputs/mcf_embedding_keyed_neuron_v3_5_5_seed1_3b_aws_v6_2 \
  outputs/mcf_embedding_keyed_neuron_v3_6_seed1_3b_aws_v6_2 \
  2>&1 | tee slurm_logs/mcf_embedding_keyed_neuron_v3_6_seed1_aws_v6_2.log
```

V3.5.5 and earlier launchers are historical, not the current experiment.
Reproducing V3.5.5 requires commit
`15f8000f0f4bd7599c93d799efcceff8faa6cdcc`.
Reproducing V3.5.4 requires commit
`51b674cd2c666e78045c216a4ea5e6a8ce2b80fd`.
Reproducing V3.5.3 requires commit
`b603c1ac74677e4a43cb506ae779f75f5a41ef11`; reproducing V3.5.2 requires commit
`be48b9bc2319c9ca55a3b9413f7e7e75c952b839`; reproducing V3.5.1 requires commit
`72aae4087490c18bab2d7f727c53c7d3331a01e3`; reproducing V3.5 requires commit
`7cd132a8af22887cdb9e195100a6a261688eef44`.

## Important outputs

- `method/neuron_selection_report.json`: stratified prompt-bank receipt and
  selected-neuron baseline activation/function tails;
- `method/writer_preflight_report.json`: frozen-writer training-safe coverage
  measured before decoder construction;
- `method/detector_hidden_cache_report.json`: all-record/all-context cache
  coverage and the exact cached-computation contract;
- `method/detector_initial_import_gate.json`: fresh confirmation that the
  pre-repair tensors exactly replay the frozen V3.2 certificate;
- `method/detector_training_log.json`: all 100 globally accumulated repair
  updates and the canonical multi-label, equal-record/per-record-tail contract;
- `method/detector_gradient_balance_audit.json`: non-optimizing step-1 raw and
  weighted component-gradient norms proving that the audit did not mutate
  gradients or optimizer state;
- `method/frozen_v3_2_detector_import.json`: hashes and lineage checks proving
  that only the passed V3.2 gate/up detector tensors were imported;
- `method/frozen_v3_4_rejection_import.json`: hashes and exact metric checks
  binding the preserved positive-reachable/selectivity-rejected V3.4 run;
- `method/frozen_v3_5_rejection_import.json`: hashes and exact aggregate checks
  binding the single-cell V3.5 rejection;
- `method/frozen_v3_5_1_forensics_import.json`: hashes and exact coordinates
  binding the diagnosed 10803-context-4 to 17353-group-30 collision;
- `method/frozen_v3_5_2_rejection_import.json`: hashes and exact coordinates
  binding the shared-prompt positive/negative contradiction;
- `method/frozen_v3_5_3_rejection_import.json`: hashes and checks binding the
  29/50 positive-collapse result with clean negative/writer-off certificates;
- `method/frozen_v3_5_4_rejection_import.json`: exact V3.5.4 detector-pass and
  width-4 actuator-reject metrics plus source artifact hashes;
- `method/exact_v3_5_4_detector_replay.json`: bit-exact gate/up tensor-hash
  verification before any actuator bank is constructed;
- `method/multilabel_prompt_manifest.json`: canonical prompt identities,
  active detector label sets, every preserved source-role occurrence, and the
  zero same-record-conflict receipt;
- `method/isolated_threshold_gate_report.json`: every raw response and clipped
  gate distribution, separate response/gate violation counts, and exact
  violating-cell coordinates;
- `method/detector_endpoint_audit.json`: hashes and consistency checks binding
  the initial frozen source certificate and repaired endpoint phases;
- `method/detector_gate_case_report.tsv`: record-index to MCF case-ID binding
  with positive, negative, writer-off, and pass/fail values;
- upstream `method/clean_stage1_acceptance.json`: exact from-Base lineage,
  relation-template policy, artifact hashes, nonempty optimization log, and
  passed training-safe 4.5 / 95% / 80% portability replay;
- `method/isolated_threshold_gate_report.json`: the mandatory fresh all-cell
  post-repair branch certificate;
- `method/actuator_neuron_selection_report.json`: detector-disjoint nested
  4/8/16 ownership, ranking inputs, and Base-feature selection scores;
- `method/actuator_width_*_feasibility.json`: five independent discarded fits
  (three native per-column-cap arms and two matched width-4-budget controls);
- `method/v3_5_5_actuator_width_feasibility.json`: width selection, matched
  controls, artifact hashes, full-context audits, and discard receipts; and
- `method/training_only_v3_5_5_completion.json`: final receipt proving that full
  preservation, checkpoint creation, and official evaluation did not occur;
- `method/frozen_v3_5_5_success_import.json`: source hashes and exact width-16
  success checks licensing V3.6;
- `method/exact_v3_5_5_width16_selection_replay.json`: bit-exact ownership
  replay for the 800 detector-disjoint actuator features;
- `method/v3_6_zero_actuator_identity_audit.json`: proof that the separate
  residual is behaviorally identical at exact-zero down delta;
- `method/v3_6_positive_warm_start.json`: criterion-stopped all-positive warm
  start with a complete incremental log;
- `method/v3_6_full_preservation_training.json`: all-context coverage,
  forgetting/locality gates, protected KL, causal checks, frozen tensor checks,
  and full training-log receipt;
- `method/v3_6_actuator_endpoint_audit.json`: pre-update, post-Adam,
  post-projection, and final-fresh replay binding; and
- `method/training_only_v3_6_completion.json`: fail-closed candidate decision.
  `method/v3_6_candidate_state.pt` exists only when that decision passes.

## Publication boundary

This repository now contains a coherent, falsifiable architecture and a
review-resistant evaluation protocol. It does not contain the GPU results that
would make the contribution publishable. An ICLR-quality paper still requires
the registered seed-1 result, unchanged confirmatory seeds, honest negative
results, literature comparison, and uncertainty/error analysis. Code structure
alone is not evidence that the mechanism works.
