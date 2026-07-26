# RWKU experiment

This experiment extends the MCF/ZsRE comparison to the official
[RWKU benchmark](https://github.com/jinzhuoran/RWKU). It uses the same local
Llama-3.2-3B-Instruct snapshot and Setting 5e hyperparameters so its method
comparison is aligned with this repository's existing tables. It does not
claim numerical comparability with a different base model from the RWKU paper.

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
The method may use only the calibration side. Headline direct and paraphrase
metrics use held-out level-2 questions. Exact duplicate records are grouped
before splitting, so duplicate content cannot cross the boundary. Level-3,
neighbor, membership-inference, and utility records are evaluation-only.

## Methods

The aggregate includes five rows:

1. Base model
2. Original ZeroUnlearn
3. Setting 5e without repair
4. Setting 5e + protected LM-head repair
5. Repair-only control

Original ZeroUnlearn is the vendored
`ZeroUnlearn.ZeroUnlearn_main.apply_unl_to_model`, with layers 16–18 and the
existing hyperparameter file. The RWKU calibration answer is mapped to
ZeroUnlearn's sensitive `target_true` field and tokenizer EOS to neutral
`target_new`.

Setting 5e uses the established 600-step all-token embedding/LM-head
margin objective and overlap-aware row restoration. RWKU calibration answers
are sensitive `target_new` values; EOS is the desired `target_true`. Unrelated
MCF facts provide 1,000 retain examples.

The protected repair freezes the transformer and input embeddings, unties the
output head if necessary, and never edits EOT/EOS. It first finds calibration
answer-token positions that still violate the forget margin. Only the
corresponding non-special target-answer output rows are eligible, and any row
that also occurs in an unrelated MCF protected answer is excluded. Thus the
parameter scope matches the sparse active-pair repair used for MCF/TOFU rather
than globally increasing a shared termination row.

Protection uses 128 unrelated MCF facts. In addition to their answer-token
likelihoods, the repair samples prompt contexts across every protected example.
The sparse delta is projected away from the leading protected hidden-state
span and is directly penalized for protected-context logit drift. A
materialized-dtype scale sweep applies three hard gates before efficacy:

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
- a four-choice formulation using target-answer distractors;
- open-ended generation recovery;
- a frozen-base-head probe: untouched base LM-head answer rows read each
  method's live final hidden states. This detects facts that remain in the
  representation even if a repaired stop-token row suppresses decoding.

## Run

Validate all pinned files and the target map:

```bash
python scripts/rwku_data.py \
  --seeds 0,1,2,3,4,5,6,7,8,9 \
  --manifest data/rwku/manifest.json
```

CPU-only protocol check:

```bash
python scripts/rwku_experiment.py --seed 0 --dry-run --no-download
```

Run all five methods for seeds 0–9 on CUDA, then aggregate:

```bash
MODEL_PATH=/path/to/Llama-3.2-3B-Instruct \
  scripts/run_rwku_experiment.sh
```

The runner deliberately fails without CUDA rather than silently switching to
a smaller or different model. Forward extra options to every seed, for
example:

```bash
MODEL_PATH=/path/to/model scripts/run_rwku_experiment.sh \
  --eval-batch-size 8 \
  --gradient-checkpointing
```

For a bounded plumbing smoke test on a GPU:

```bash
python scripts/rwku_experiment.py \
  --seed 0 \
  --model-path /path/to/model \
  --steps 2 \
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
expected target for each seed, and all five method rows. It reports mean ±
population standard deviation.
