# RWKU Setting 5e experiments

This repository keeps three RWKU methods/protocols distinct. Their results are
not interchangeable.

| CLI training source | Human-readable label | Protocol status |
|---|---|---|
| `probe_assisted_entity_fact` | RWKU probe-assisted entity-fact portability | `nonofficial_probe_assisted_entity_fact_portability` |
| `target_only_generated_entity_corpus` | Setting 5e + protected repair with RWKU target-generated entity corpus | `official_protocol_different_model_confirmatory_method_extension` |
| legacy command (no `--training-source`) | legacy independent row-hash experiment | `prompt_held_out_only_legacy_nonofficial` |

The target-only track on Llama-3.2-3B-Instruct is an **official RWKU protocol
evaluation on a different model using an RWKU-specific
target-corpus-generation extension**. It is not an exact reproduction of the
paper's Llama-3-8B results, unchanged Setting 5e, or a native numerical paper
reproduction.

The LoRA implementation in `scripts/rwku_representation.py` remains a
**separate representation-unlearning method**. It is not merged into Setting
5e or sparse LM-head repair.

The single Stephen King SURE head-only feasibility experiment is documented in
[`RWKU_SURE_HEAD_ONLY_W1K.md`](RWKU_SURE_HEAD_ONLY_W1K.md). It reuses the
target-only staged firewall, but it is a separate development method and does
not change the Setting 5e defaults below.

## Preserved Setting 5e implementation

The staged tracks call the existing implementation in
`scripts/gagd_compare.py` with mode
`emb_lm_all_restore_post_training_true`. The established defaults remain:

- all-token input-embedding and LM-head optimization; transformer frozen;
- 600 steps, batch size 1, retain batch size 4;
- AdamW, learning rate `1e-4`, no weight decay, gradient clipping `1.0`;
- the existing `mcf_margin` objective with margin `1.0`;
- forget weight `2.0`, retain weight `1.0`;
- overlap-aware post-training restoration with alphas `0.75/0.50/0.25`;
- 1,000 unrelated MCF optimization records and a disjoint 128-record MCF
  repair-selection partition.

RWKU optimization views compile to `gagd.Example` with this exact direction:

```text
answer      = sensitive answer
target_new  = sensitive answer
target_true = tokenizer.eos_token
source      = fact_id
```

EOS is resolved from the loaded tokenizer at runtime. It is never serialized
as a guessed Llama token. Original ZeroUnlearn intentionally uses the opposite
request convention:

```text
target_true = sensitive answer
target_new  = tokenizer.eos_token
```

The default behavior of MCF, ZsRE, and TOFU is unchanged. Balanced fact-cycle
sampling is activated only for an entity-fact training bundle.

## Entity-fact schema and identity

`config/rwku/entity_fact_schema_v1.json` (`rwku_entity_fact_v1`) is the
authoritative entity-fact schema. Every fact contains:

```json
{
  "schema_version": "rwku_entity_fact_v1",
  "protocol_label": "...",
  "protocol_status": "...",
  "entity_id": "rwku:1_Stephen_King",
  "subject": "Stephen King",
  "subject_aliases": [],
  "fact_id": "SHA256(entity_id, relation_id, normalized answer)",
  "relation_id": "first_published_novel",
  "canonical_sensitive_answer": "Carrie",
  "sensitive_answer_aliases": [],
  "source_records": [],
  "optimization_views": [],
  "held_out_views": [],
  "partition": "calibration_fact | unseen_fact | generated_training_fact",
  "training_allowed": true,
  "source_hashes": {},
  "relation_assignment_provenance": [],
  "manual_override_sha256": "..."
}
```

Fact identity is computed from all three components:

```text
SHA256(entity_id, relation_id, normalized canonical_sensitive_answer)
```

It is never subject-plus-answer or answer-only. Thus `birth_city → Portland`
and `birth_state → Maine` remain distinct, and the same answer under two
different relation IDs also remains distinct.

Every source record retains its source file, row index, full record SHA-256,
level, query type, normalized query hash, original answer, assigned relation,
and assigned fact. SHA-256 is an integrity and identity key, not a learned
feature: it proves which immutable row was assigned, keeps duplicates
indivisible, makes splits reproducible, and detects upstream changes. It does
not hide the data or define semantic similarity.

Relation assignment uses, in order:

1. a committed manual override;
2. the frozen deterministic mapper;
3. an optional pinned/revision-recorded clusterer; or
4. a hard failure.

The seed-0 Stephen King override is
`config/rwku/fact_overrides/seed0.json`. It is derived only from pinned Level 1
and Level 2. Level 3 is not loaded by the builder. Strict mode rejects missing
relations, ambiguous records, conflicting canonical answers, semantically
overloaded relations, unknown override hashes, or aliases crossing facts.

## Probe-assisted split

Level 1 and Level 2 are combined before fact assignment. Exact records and
normalized prompt views are deduplicated. For `N >= 2` facts and requested
fraction `f`:

```text
n_unseen = floor(N * f + 0.5)
n_unseen = max(1, min(N - 1, n_unseen))
split_key = SHA256(f"{split_seed}:fact:{fact_id}")
```

Facts are sorted by `split_key`; the first `n_unseen` are wholly unseen and the
rest are calibration facts. No fact, alias, duplicate, or equivalent view may
cross.

Within each calibration fact:

```text
view_key = SHA256(
  f"{split_seed}:view:{fact_id}:{view_content_sha256}"
)
```

For a multi-view fact, at least one sorted view is held out for
**seen-fact/unseen-prompt generalization** and the remaining views optimize
Setting 5e. A single-view calibration fact is training-only. Every view of an
unseen fact is reserved for **unseen-fact entity transfer**.

The old independent Level-1/Level-2 row-hash split remains only through the
legacy command/`--legacy-row-split`. Its correct status is
`prompt_held_out_only_legacy_nonofficial`; it is not an unseen-fact split.

## Immutable artifacts and permissions

Every JSON artifact carries `artifact_role`, `gradient_allowed`,
`selection_allowed`, `evaluation_only`, `allowed_stages`, and a SHA-256 over
its payload and permission metadata.

| Artifact role | Gradient | Selection | Evaluation-only | Allowed stage |
|---|---:|---:|---:|---|
| `training_bundle` | yes | no | no | train |
| `optimization_protection` | yes | no | no | train |
| `repair_selection_gate` | no | yes | no | train |
| `seen_fact_unseen_prompt_eval` | no | no | yes | evaluate |
| `unseen_fact_eval` | no | no | yes | evaluate |
| `official_locked_eval` | no | no | yes | evaluate |

Probe-assisted preparation writes:

- `fact_catalog.json`;
- `training_bundle.json`;
- `seen_fact_unseen_prompt_eval.json`;
- `unseen_fact_eval.json`;
- `official_locked_eval.json`;
- `split_manifest.json`; and
- `fact_audit.md`.

Target-only corpus generation writes:

- `generated_entity_fact_catalog.json`;
- `generated_training_bundle.json`;
- `generated_raw_corpus.json`; and
- `generator_receipt.json`.

Target-only preparation creates `official_locked_eval.json` from the committed
manifest without opening official rows. It includes all official Level 1,
Level 2, Level 3, MIA, neighbor, utility, and fluency identities. The official
files are opened only after checkpoint receipt verification and the atomic
evaluation-opening transition.

## One-way state and data access

The enforced state machine is:

```text
PREPARED
  -> TRAINING
  -> CHECKPOINT_FROZEN
  -> OFFICIAL_EVALUATION_OPENED
  -> EVALUATION_COMPLETE
```

No backward transition is legal. Confirmatory target-only runs reject
`--stage all`; training and evaluation must be separate processes.

- Probe-assisted `prepare` may read pinned Level 1 and Level 2 only.
- Target-only `prepare/train` receive the target name, pinned model, frozen
  generator configuration, generated bundle, and target-independent
  retain/protection resources. They do not open any official evaluation row.
- `train` can open only method-visible training and optimization-protection
  artifacts. A repair gate is evaluated under `torch.no_grad()` and cannot
  contribute to a backward pass.
- `evaluate` validates every declared hash, verifies
  `CHECKPOINT_FROZEN`, atomically records official evaluation opening, and only
  then opens official data.

After official evaluation opens, that experiment ID cannot update, repair,
rescale, retune, replace, or reject the checkpoint. A changed run needs a new
experiment ID and is not confirmatory with respect to already observed
official results.

The checkpoint receipt records model/tokenizer identities, all checkpoint and
artifact hashes, MCF retain/gate hashes, matched-protection hashes, complete
method configuration, implementation hashes, generator receipt, sampler
provenance, timestamps, and confirmatory status. Any changed checkpoint,
training bundle, implementation, or method configuration fails closed.

## Balanced fact-cycle sampling

Entity-fact training does not sample facts with replacement. Each cycle
deterministically shuffles every training fact, visits each once, and then
starts another shuffled complete cycle. For `S` steps and `K` facts, each fact
receives `floor(S/K)` or `ceil(S/K)` updates; exposure imbalance is at most
one.

The run records exposures by fact, view, prompt style, answer alias, actual
tokenizer token ID, and decoded token piece, plus sampler seed, implementation
SHA-256, and plan SHA-256. MCF/ZsRE/TOFU retain their existing row sampler.

Supported view labels are direct question, cloze, conservative subject alias,
deterministic paraphrase, and forced-prefix. Reverse views are off by default.
`--include-relation-conditioned-reverse-prompts` admits only explicitly
declared, logically unambiguous `relation-conditioned reverse` views and marks
them `boundary_expanding=true`; they are a separate ablation.

The optional export is named **MCF-shaped RWKU training request** and has
format `mcf_shaped_rwku_training_request_v1`, benchmark `rwku`, and
`training_only=true`. It is adapter glue, not an MCF benchmark record. The MCF
evaluator rejects it. Its neutral target is a runtime EOS marker resolved by
the RWKU loader.

## Target-only generated corpus

`scripts/build_rwku_generated_entity_corpus.py` has no RWKU data-root input.
It receives only the target/entity ID, pinned local generator identity, frozen
generation config, seed, and optional independent resources. It does not
search, inspect, hash, or open official RWKU files.

The receipt records generator model/revision, tokenizer, prompt templates and
hash, decoding parameters, seeds, raw corpus hash, extractor implementation
and hash, extraction config, accepted/rejected facts and reasons, duplicate
and alias handling, and final bundle hash. `--dry-run` validates configuration
without importing torch or loading a model. Corpus generation and fact
extraction are part of the RWKU-specific method extension.

## Matched protection and repair

Protection keys may come only from a method-visible training/generated bundle,
an independently generated entity corpus, or the predeclared target-independent
vocabulary. Each key records its normalized value, origin type/path/hash,
source fact, visibility before freeze, and vocabulary revision if applicable.

Keys cannot be discovered from held-out Level 1/2, Level 3, MIA, neighbors,
utility, fluency, evaluation artifacts, evaluation outputs, or post-evaluation
errors. Thus `Maine` is unavailable if it occurs only in an unseen official
fact. Strict mode rejects a key without a valid provenance chain.

`scripts/build_rwku_matched_protection.py` creates content-hash-disjoint
`matched_protection_train.json` and `matched_protection_gate.json`, plus a
coverage report by answer/alias, token ID/piece, relation/source, and
optimization/gate counts. Insufficient coverage warns or fails in strict mode;
it is never silently called safe.

Sparse repair retains the existing hard gates:

- protected-answer probability ratio at least `0.999`;
- selected-row logit drift at most `0.05`;
- protected top-1 changes equal `0`; and
- mandatory scale zero.

EOS, EOT, BOS, PAD, UNK, every tokenizer special row, and rows shared with
protected unrelated answers are ineligible. If no useful row is safe, scale
zero is a documented no-op.

Reports distinguish token-position, view, and fact outcomes. Token protection
classes include `safe_sparse_head_pair`, `shared_protected_answer_pair`,
`special_token_pair`, and `unsupported_pair`. The qualified Setting 5e label is
`calibration_resolved_by_setting5e`, never `resolved_by_setting5e`. Multi-token
answers can be partly supported. View/fact success refers only to calibration;
it never implies official entity removal.

Sparse LM-head repair is decoder suppression, not representation erasure.
Frozen-head, forced-prefix, multiple-choice, Level-3, MIA, and utility controls
remain necessary. A zero direct-recovery result alone cannot establish that an
entity was removed.

## Evaluation output

The evaluator preserves separate sections for calibration recovery,
seen-fact/unseen-prompt recovery, unseen-fact recovery, official Level 1/2/3,
Level-3 attack types, probabilities, forced-prefix, aliases,
multiple-choice/open-ended/frozen-head controls, membership inference,
neighbors, downstream utility, fluency, perplexity, and full-retain ratio.

Every recovery section includes numerator, denominator, percentage, prompt
count, target count, independent fact count when applicable, and a Wilson
interval when meaningful. There is no universal “entity removed” score.
Calibration efficacy, prompt generalization, unseen-fact transfer,
adversarial resistance, decoder suppression, representation recovery, and
utility preservation are separate claims.

## Existing MCF/ZsRE/TOFU repaired-result status

Affected repaired runs now carry:

```text
native_data_and_metrics_but_evaluation_conditioned_repair
```

This applies only to repaired paths where evaluator-derived evidence affected
repair: MCF official paraphrases, ZsRE rewrite/paraphrase/correctness evidence,
or TOFU utility-calibration evidence. Base and unrepaired Setting-5e-only rows
are not automatically assigned this status. Historical metric values are not
rewritten.

## Commands

CPU-only probe-assisted preparation (no download or model load):

```bash
python scripts/rwku_experiment.py \
  --stage prepare \
  --training-source probe_assisted_entity_fact \
  --experiment-id rwku-sk-probe-v1 \
  --seed 0 \
  --model-path /path/to/Llama-3.2-3B-Instruct \
  --output-root outputs/rwku_entity_fact \
  --fact-overrides config/rwku/fact_overrides/seed0.json \
  --fact-holdout-fraction 0.25 \
  --prompt-holdout-per-seen-fact 1 \
  --split-seed 0 \
  --dry-run \
  --no-download
```

Build the matched bank from an independently sourced utility corpus:

```bash
python scripts/build_rwku_matched_protection.py \
  --training-bundle outputs/rwku_entity_fact/rwku-sk-probe-v1/training_bundle.json \
  --source-corpus /path/to/independent_matched_utility.json \
  --protection-vocabulary config/rwku/protection_vocabulary_v1.json \
  --tokenizer-path /path/to/Llama-3.2-3B-Instruct \
  --output-dir outputs/rwku_entity_fact/rwku-sk-probe-v1/protection
```

Probe-assisted training and freezing:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/rwku_experiment.py \
  --stage train \
  --training-source probe_assisted_entity_fact \
  --experiment-id rwku-sk-probe-v1 \
  --seed 0 \
  --model-path /path/to/Llama-3.2-3B-Instruct \
  --model-revision PINNED_REVISION \
  --output-root outputs/rwku_entity_fact \
  --entity-fact-bundle outputs/rwku_entity_fact/rwku-sk-probe-v1/training_bundle.json \
  --matched-protection-train outputs/rwku_entity_fact/rwku-sk-probe-v1/protection/matched_protection_train.json \
  --matched-protection-gate outputs/rwku_entity_fact/rwku-sk-probe-v1/protection/matched_protection_gate.json \
  --steps 600 \
  --no-download
```

Probe-assisted evaluation (irreversibly opens official evaluation):

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/rwku_experiment.py \
  --stage evaluate \
  --training-source probe_assisted_entity_fact \
  --experiment-id rwku-sk-probe-v1 \
  --seed 0 \
  --model-path /path/to/Llama-3.2-3B-Instruct \
  --output-root outputs/rwku_entity_fact \
  --checkpoint-receipt outputs/rwku_entity_fact/rwku-sk-probe-v1/checkpoint_receipt.json \
  --no-download
```

Target-only corpus generation (GPU; not official evaluation):

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/build_rwku_generated_entity_corpus.py \
  --target-entity "Stephen King" \
  --entity-id rwku:1_Stephen_King \
  --generator-model /path/to/Llama-3.2-3B-Instruct \
  --generator-revision PINNED_REVISION \
  --generation-config config/rwku/generation/llama32_3b_target_corpus_v1.json \
  --seed 0 \
  --output-dir outputs/rwku_target_only/corpus/stephen_king
```

Target-only preparation (CPU; official rows remain locked):

```bash
python scripts/rwku_experiment.py \
  --stage prepare \
  --training-source target_only_generated_entity_corpus \
  --experiment-id rwku-sk-target-only-v1 \
  --confirmatory \
  --seed 0 \
  --model-path /path/to/Llama-3.2-3B-Instruct \
  --output-root outputs/rwku_target_only \
  --generated-entity-fact-bundle outputs/rwku_target_only/corpus/stephen_king/generated_training_bundle.json \
  --generator-receipt outputs/rwku_target_only/corpus/stephen_king/generator_receipt.json \
  --no-download
```

Build matched protection using that generated bundle, then target-only
training:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/rwku_experiment.py \
  --stage train \
  --training-source target_only_generated_entity_corpus \
  --experiment-id rwku-sk-target-only-v1 \
  --confirmatory \
  --seed 0 \
  --model-path /path/to/Llama-3.2-3B-Instruct \
  --model-revision PINNED_REVISION \
  --output-root outputs/rwku_target_only \
  --generated-entity-fact-bundle outputs/rwku_target_only/corpus/stephen_king/generated_training_bundle.json \
  --generator-receipt outputs/rwku_target_only/corpus/stephen_king/generator_receipt.json \
  --matched-protection-train outputs/rwku_target_only/rwku-sk-target-only-v1/protection/matched_protection_train.json \
  --matched-protection-gate outputs/rwku_target_only/rwku-sk-target-only-v1/protection/matched_protection_gate.json \
  --steps 600 \
  --no-download
```

Target-only confirmatory evaluation:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/rwku_experiment.py \
  --stage evaluate \
  --training-source target_only_generated_entity_corpus \
  --experiment-id rwku-sk-target-only-v1 \
  --confirmatory \
  --seed 0 \
  --model-path /path/to/Llama-3.2-3B-Instruct \
  --output-root outputs/rwku_target_only \
  --checkpoint-receipt outputs/rwku_target_only/rwku-sk-target-only-v1/checkpoint_receipt.json \
  --no-download
```

Do not tune or rerun the same experiment ID after evaluating. Do not train on
official Level 1/2/3, MIA, neighbor, utility, or fluency records in target-only
mode.
