# RWKU experiment

This experiment evaluates on pinned
[RWKU benchmark](https://github.com/jinzhuoran/RWKU) records. It uses the same
local Llama-3.2-3B-Instruct snapshot and Setting 5e hyperparameters so its
method comparison is aligned with this repository's existing tables. It does
not claim numerical comparability with a different base model from the RWKU
paper.

This is a **held-out-clean corpus-assisted portability experiment**, not the
official RWKU target-entity-only training protocol. The method sees a declared
calibration half of level-1/level-2, `positive.json`, and unrelated MCF retain
records. Official RWKU makes only the target entity and original model visible
to the method. Therefore these results must not be labeled official/native
RWKU; the official benchmark registry correctly keeps unchanged `our_method`
at `NEEDS_METHOD_EXTENSION`.

## Experimental unit and seeds

RWKU is a single-target benchmark: one real-world person is unlearned in each
run. Seeds 0–9 therefore identify ten independent targets rather than ten
random resamplings of one batch:

| Seed | RWKU target |
|---:|---|
| 0 | Stephen King |
| 1 | Confucius |
| 2 | Bruce Lee |
| 3 | Warren Buffett |
| 4 | Christina Aguilera |
| 5 | Cindy Crawford |
| 6 | Marie Osmond |
| 7 | Paris Hilton |
| 8 | Justin Bieber |
| 9 | Prince Harry, Duke of Sussex |

The dataset is pinned to commit
`d72f493d481d1b0a9bdc6e869d32baeffad8904f`; the benchmark code is pinned to
`b8a03b3ce34fb4a96001df545a56558d75a078a3`.

For each target, content hashes split official level-1 and level-2 probes 50:50.
Probe-derived objectives may use only the calibration side. The representation
method additionally uses RWKU's pinned `positive.json` as its declared
target-training corpus and as a calibration-only MIA proxy. Its proxy score may
select a checkpoint, but it is never reported as final evidence of success.
Headline direct and paraphrase metrics use held-out level-2 questions. Exact
duplicate records are grouped before splitting, so duplicate content cannot
cross the boundary. Level-3, neighbor, membership-inference, and utility
records are evaluation-only.

## Methods

The aggregate includes six rows:

1. Base model
2. Original ZeroUnlearn
3. Setting 5e without repair
4. Setting 5e + protected LM-head repair
5. Repair-only control
6. Protected representation unlearning v2

Original ZeroUnlearn is the vendored
`ZeroUnlearn.ZeroUnlearn_main.apply_unl_to_model`, with layers 16–18 and the
existing hyperparameter file. The RWKU calibration answer is mapped to
ZeroUnlearn's sensitive `target_true` field and tokenizer EOS to neutral
`target_new`.

Setting 5e uses the established 600-step all-token embedding/LM-head
margin objective and overlap-aware row restoration. RWKU calibration answers
are sensitive `target_new` values; EOS is the desired `target_true`. Unrelated
MCF facts provide 1,000 retain examples. These optimization examples and the
external MCF gate examples described below are sampled as disjoint sets.

The protected repair freezes the transformer and input embeddings, unties the
output head if necessary, and never edits EOT/EOS. It first finds calibration
answer-token positions that still violate the forget margin. Only the
corresponding non-special target-answer output rows are eligible, and any row
that also occurs in an unrelated MCF protected answer is excluded. Thus the
parameter scope matches the sparse active-pair repair used for MCF/TOFU rather
than globally increasing a shared termination row.

Protection uses 128 unrelated MCF facts from the external gate partition, not
from the 1,000 facts used to optimize Setting 5e. In addition to their
answer-token likelihoods, the repair samples prompt contexts across every
protected example. The sparse delta is projected away from the leading
protected hidden-state span and is directly penalized for protected-context
logit drift. A materialized-dtype scale sweep applies three hard gates before
efficacy:

- protected-answer probability ratio at least `0.999`;
- maximum selected-row logit drift at most `0.05`; and
- zero protected-context top-1 changes.

Scale zero is mandatory and wins whenever no effective edit passes all three
gates. The repair-only row applies this exact procedure to a fresh base model.
Every repair report lists the selected rows, protected overlaps, unsupported
active positions, all scale candidates, and the reason for a no-op.

This is deliberately an LM-head ablation, not a claim of representation-level
erasure. The letter-scored multiple-choice control bypasses the edited answer
rows, and the frozen-base-head probe bypasses the repaired head entirely.
Improving either metric requires changing the live hidden representation in
the Setting 5e stage; an output-row repair cannot honestly do so.

Protected representation unlearning v2 is the corpus-assisted
representation-level method. It
starts from a fresh Base checkpoint, rather than inheriting Setting 5e's
embedding/head changes. It freezes the input embeddings, normalization
parameters, and LM head, then trains low-rank adapters on selected projection
matrices in the late transformer blocks. The selected adapter delta is merged
into those projections for evaluation. Consequently, a successful result
cannot be explained by changing EOT/EOS or by editing the target answer's
output row.

The v2 representation configuration uses rank-24 adapters (`alpha=48`)
on `q_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj` in
the final twelve decoder layers for 1,800 steps. Three of every five steps
replay four stratified QA constraints; the other two cover MC neutrality and
two MIA-proxy records. Up to 16 `positive.json` rows supply subject-cloze
tasks. It uses subject-versus-masked-subject anchors at every second selected
residual layer, a final-state concept basis, a worst-token-aware answer
probability target of `1e-6`, balanced four-position MC neutrality,
frozen-Base-head target demotion, and current-distribution
loss/Min-K/zlib/Min-K++ matching. These are fixed pre-evaluation defaults, not
values tuned on held-out outcomes.

At 85% of the fixed training budget, v2 scores every declared calibration QA
constraint without generation and builds a diverse active set of at most 96
remaining violations. The last 15% of steps replays that set while keeping the
same external retain objectives. This calibration-only polish is deterministic;
it never reads level-3, official MIA, neighbor, utility, or held-out answers.

Its probe-derived forget objective is constructed only from the target's
calibration split. It includes answer-token unlikelihood or margin losses over
direct and cloze questions, deterministic paraphrases, conservative aliases,
forced prefixes, truthful reverse-association questions, and generic
adversarial wrappers. RWKU's designated `positive.json` training text supplies
subject-cloze tasks and a bounded representation proxy to broaden coverage
beyond the calibration answers; raw all-token gradient ascent is not used.
This makes the method corpus-assisted rather than RWKU target-only/zero-shot.
Two controls are part of the training objective rather than post-hoc decoder
tricks:

- a frozen copy of the Base LM head scores live final hidden states. The
  correct target row is constrained below every other declared answer row,
  while all non-target rows are neutralized instead of rewarding one false
  answer. A candidate must keep frozen-head accuracy at or below reported
  chance, target probability at or below uniform chance, and mean normalized
  target rank at least `0.90`;
- four-way multiple-choice questions rotate the correct answer through every
  letter position while their A/B/C/D logits are driven toward a uniform
  distribution. The loss does not reward choosing a known-wrong answer.

Retention is anchored to cached Base-model teacher outputs. Retain answer
likelihood, top-token distribution divergence, top-1 agreement, and hidden
state similarity constrain the adapters on unrelated MCF prompts. The MCF
examples used for optimization and those used for checkpoint gating are
disjoint. This prevents selecting a checkpoint merely because it memorized
the retention batch. Additional constructed-MC versions of external MCF facts
protect letter-scored utility, and low/high quantile gates catch outliers that
an average-only retain metric would hide.

Training projects only scored decoder states through the frozen LM head for
QA, MC, and retain objectives instead of materializing logits for every
sequence position. Independent target and retain graphs backpropagate
sequentially within each optimizer step. Four QA constraints or two
positive-likelihood constraints are replayed per relevant phase. This expands
complete task-family coverage while keeping peak accelerator memory bounded.

Checkpoint and scale selection may use the RWKU calibration split, the
declared target/non-target `positive.json` proxy, and the disjoint external
retain gate only. Final held-out level-1/level-2 questions, all level-3
attacks, official membership-inference records, neighboring entities, and
downstream utility sets remain evaluation-only. No reported held-out result is
used to choose a checkpoint, stop training, or reject a run. Adapter snapshots
at steps 250, 500, 750, 1,000, 1,250, 1,500, 1,750, and 1,800 enter a coarse
checkpoint/scale funnel; the best checkpoints and neighboring scales then
receive the full gate suite.
Every candidate is evaluated after its update is materialized into the model
weight dtype, so the retained checkpoint is the exact model that passed the
gates. Selection also performs bounded greedy generation on stratified
calibration prompts; teacher-forced suppression alone is not accepted as
calibration recovery success.

The target and non-target `positive.json` records are content-deduplicated and
split before any subject-cloze or likelihood objective is constructed.
Subject-cloze tasks use optimization records only; gate hashes cannot enter
training through another objective. Gate rows use the same final 512-token
window convention as the final likelihood attack. A candidate must keep
non-target feature drift below `0.01`; it cannot improve proxy AUC merely by
damaging both populations.

## Metrics and controls

Accuracies are percentage points. Probabilities and membership attack
advantage are in `[0,1]`.

Forget metrics (lower is better):

- held-out fill-in-the-blank recovery and answer probability;
- direct held-out target QA recovery and answer probability;
- deterministic paraphrase recovery;
- conservative subject-alias question recovery, with coverage;
- official level-3 adversarial recovery, including per-attack breakdown;
- direction-agnostic membership-inference advantage across loss, zlib,
  Min-K 20%, and Min-K++ 20%;
- target-answer geometric mean token probability.

Retain metrics:

- neighboring-entity recovery;
- MMLU accuracy;
- BBH reasoning exact match;
- TruthfulQA MC1/MC2;
- TriviaQA exact match/F1;
- general utility as the macro-average of MMLU, BBH, TruthfulQA MC1, and
  TriviaQA F1;
- the same local Wikidata perplexity convention used by the MCF evaluator;
- full-retain answer-probability ratio against the exact seed-matched base
  pass.

Alternative-output controls:

- full answer log likelihood and geometric token probability;
- answer-suffix recovery after forcing the first half of a multi-token answer;
- deterministic answer aliases, with coverage;
- a four-choice formulation using target-answer distractors, with all four
  correct-letter rotations as the primary score and the old single ordering
  retained as a diagnostic;
- open-ended generation recovery;
- a frozen-base-head probe: untouched base LM-head answer rows read each
  method's live final hidden states. This detects facts that remain in the
  representation even if a repaired stop-token row suppresses decoding.

## Acceptance targets

Every non-Base result now includes a machine-readable `success_contract`.
Missing required values fail closed, so bounded or `--skip-ppl` runs cannot be
mistaken for complete successes. The fixed held-out targets are:

- direct, cloze, paraphrase, alias, forced-prefix, and open-ended recovery
  exactly `0%`;
- adversarial recovery at most `10%`;
- target, full-answer, forced-prefix, and answer-alias probability below
  `0.01` (`1e-6` remains the stricter calibration optimization target);
- direction-agnostic membership-inference advantage at most `0.05`;
- four-way multiple-choice recovery within five points of `25%` chance; and
- frozen-head recovery at or below its explicitly reported candidate-set
  chance baseline, target probability at or below uniform chance, and
  normalized target rank at least `0.90`.

Retain gates require the full-retain probability ratio in `0.995–1.005`,
neighbor and general utility drops no larger than two points, component utility
drops no larger than three points, and perplexity no more than `2%` above Base.
For the representation method, all internal calibration efficacy and disjoint
non-target protection gates must also pass.

Driving MC below chance can indicate systematic answer inversion, so it is not
automatically stronger evidence than chance-level behavior. These are protocol
targets, not measured claims; a fresh GPU run determines whether v2 meets them.

## Run

Validate all pinned files and the target map:

```bash
python scripts/rwku_data.py \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --manifest data/rwku/manifest.json
```

CPU-only protocol check:

```bash
python scripts/rwku_experiment.py \
  --seed 0 \
  --dry-run \
  --no-download \
  --model-path /path/to/Llama-3.2-3B-Instruct
```

Run the fixed v2 representation pilot for seed 0 (Stephen King) on NJIT after
the pinned files are present locally:

```bash
cd /scratch/yl258/kp759/Unlearning/semantic-unlearning
PYTHON_BIN=/path/to/unlearning/bin/python \
MODEL_PATH=/scratch/yl258/kp759/hf/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95 \
CUDA_VISIBLE_DEVICES=0 \
  scripts/run_rwku_seed0_representation_v2.sh
```

The launcher runs Base plus v2 representation, performs the complete held-out
evaluation, saves the selected checkpoint, forbids network downloads, and
writes under `outputs/rwku_v2` unless `RWKU_OUTPUT_ROOT` is set. Do not tune
its method parameters from the resulting level-3/MIA/neighbor/utility scores;
if the fixed pilot fails, change formulation using calibration diagnostics and
start a newly declared experiment.

Run all six methods for seeds 0–9 on CUDA, then aggregate:

```bash
MODEL_PATH=/path/to/Llama-3.2-3B-Instruct \
  scripts/run_rwku_experiment.sh
```

The runner deliberately fails without CUDA rather than silently switching to
a smaller or different model. Forward extra options to every seed, for
example:

```bash
MODEL_PATH=/path/to/model scripts/run_rwku_experiment.sh \
  --eval-batch-size 8
```

Gradient checkpointing is enabled by default for the transformer-adapter
stage; use `--no-gradient-checkpointing` only when memory headroom has been
verified.

For a bounded plumbing smoke test on a GPU:

```bash
python scripts/rwku_experiment.py \
  --seed 0 \
  --model-path /path/to/model \
  --steps 2 \
  --representation-steps 2 \
  --repair-steps 2 \
  --retain-num 8 \
  --repair-retain-num 4 \
  --forget-eval-limit 2 \
  --adversarial-eval-limit 2 \
  --mia-eval-limit 2 \
  --neighbor-eval-limit 2 \
  --utility-eval-limit 2 \
  --skip-ppl
```

The default repair gates are intentionally strict. They can be made explicit
for a final run:

```bash
MODEL_PATH=/path/to/model scripts/run_rwku_experiment.sh \
  --repair-min-protected-probability-ratio 0.999 \
  --repair-max-protected-logit-drift 0.05 \
  --repair-max-protected-top1-changes 0 \
  --repair-protected-projection-rank 256 \
  --repair-protected-contexts-per-example 8
```

Smoke results are not valid final measurements and strict aggregation will
reject missing seeds or methods.

Each seed writes `config_used.json`, one JSON result per method, repair
diagnostics, and a combined `results.json`. Strict aggregation writes:

- `outputs/rwku/aggregate/rwku_aggregate.md`
- `outputs/rwku/aggregate/rwku_aggregate.csv`
- `outputs/rwku/aggregate/rwku_aggregate.json`

The aggregator requires exactly seeds 0–9, the pinned dataset revision, the
expected target for each seed, and all six method rows. It reports mean ±
population standard deviation.
