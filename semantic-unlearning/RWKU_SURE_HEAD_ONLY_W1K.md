# RWKU-H-W1K: Stephen King head-only feasibility

`RWKU-H-W1K` is one development-only experiment. It asks whether SURE can
erase method-visible atomic knowledge about the RWKU seed-0 target, Stephen
King, using sparse LM-head edits while preserving an external 1,000-document
Wikipedia distribution. It is not a sweep and it is not a confirmatory result.

## Locked experiment

The immutable specification is
[`config/rwku/sure_head_only_w1k_seed0.json`](config/rwku/sure_head_only_w1k_seed0.json).
The learner rejects changes to the defining fields:

- target: RWKU seed 0, `Stephen King`, entity ID `rwku:1_Stephen_King`;
- corpus: the v3 `llama32_3b_target_corpus_v3_atomic_facts` bundle and its
  matching complete generator receipt; the receipt's resolved local generator
  snapshot must be the same snapshot supplied as the unlearning base model;
- training views: direct question, cloze, deterministic paraphrase, and the
  suffix-only forced-prefix view when generated;
- replacement: the fixed text `Unknown`;
- editable weights: content-bearing sensitive-answer rows plus neutral-target
  LM-head rows only; punctuation-only rows and the locked functional-token set
  (`a`, `an`, `and`, `at`, `by`, `for`, `in`, `is`, `of`, `on`, `the`, `to`)
  are filtered, while input embeddings and transformer parameters remain
  frozen;
- Stage 1: per-row Rank 4, bounded GA(sensitive), bounded GD(`Unknown`), and
  exact external-Wikipedia KL;
- Stage 2: residual rank ladder 2, 4, 8, used only if Stage 1 leaves atomic-view
  failures;
- utility: exactly 1,000 real-Wikipedia documents, 100,000 requested predictor
  states, at least 90,000 realized states, and no benchmark retain examples;
- checkpoint dtype: BF16 on a single CUDA device.

The utility builder excludes the first 20 documents used by the repository PPL
audit. It also removes every eligible document containing `Stephen King` under
whitespace-normalized, case-insensitive substring matching. The cache records
the exclusion string, rejected-document count, and rejected-index digest. The
learner fails closed if that audit or the prepared-corpus receipt is absent.

## Run the atomic bundle

The corpus directory must already contain the independently generated files
`generated_training_bundle.json` and `generator_receipt.json`. The Wikipedia
directory must be the real external corpus prepared under
`sure_external_wikipedia_corpus_v1`.

```bash
cd semantic-unlearning

bash scripts/run_rwku_sure_head_only_w1k.sh \
  /home/ec2-user/models/Llama-3.2-3B-Instruct \
  outputs/rwku_target_only/corpus/stephen_king_v3_atomic_seed0_run1 \
  /path/to/prepared/real_wikipedia
```

Optional path controls are `RWKU_H_W1K_OUTPUT_ROOT`,
`RWKU_H_W1K_CACHE_ROOT`, `RWKU_H_W1K_UTILITY_CACHE`, `RWKU_DATA_ROOT`, and
`WIKIDATA_DIR`. `RWKU_NO_DOWNLOAD=1` requires all pinned RWKU evaluation files
to be present locally. These controls do not alter the locked training
objective or acceptance thresholds.

The launcher performs three one-way stages:

1. `prepare` records the atomic bundle and generator-receipt identities and
   creates a descriptor of the pinned official files without opening them.
2. `rwku_sure_head_only_w1k.py` sees only the base model, atomic bundle,
   generator receipt, locked configuration, and target-excluded W1K cache. It
   freezes a checkpoint receipt only after the BF16 feasibility gate passes.
3. The generic RWKU staged evaluator verifies every frozen identity, atomically
   opens official evaluation, then measures native Level 1/2/3, adversarial,
   MIA, neighbor, downstream utility, fluency, frozen-head, and PPL outcomes.

The learner deliberately has no CLI argument for an RWKU data root, official
forget probe, neighbor probe, MIA record, or downstream utility record.

## Feasibility decision

A checkpoint is emitted only when all of these are true after physical BF16
materialization:

- direct atomic-view sensitive NLL exceeds neutral NLL on every prompt;
- cloze, deterministic-paraphrase, and forced-prefix sensitive NLL exceeds
  neutral NLL on every prompt;
- every separation is at least `0.01`;
- Wikipedia KL mean, p95, and max are at most `0.01`, `0.05`, and `0.5`;
- sparse total delta norm is at most `1.5`;
- every non-output parameter version remains unchanged and the LM head remains
  untied from the frozen input embeddings.

The principal artifacts are written under
`outputs/rwku_h_w1k/rwku-h-w1k-stephen-king-atomic-seed0-v1/`:

- `sure_head_only_w1k/training_report.json` — atomic gate, utility gate, and
  head-only invariants;
- `sure_head_only_w1k/final_total_delta.pt` — edited LM-head row IDs and delta;
- `sure_head_only_w1k/checkpoint/` — the frozen candidate;
- `checkpoint_receipt.json` — immutable training, code, cache, and checkpoint
  identities;
- `official_evaluation.json` — base and unlearned native RWKU results, created
  only after the receipt boundary opens.

Passing the training gate proves only head-only feasibility on the generated
atomic training distribution. The scientific result is the post-freeze native
RWKU evaluation, especially Level 2/3 forgetting, adversarial recovery, MIA,
neighbor locality, and general utility. Because the intervention edits decoder
rows, it should be described as sparse decoder suppression, not proof that all
internal entity representations were erased.
