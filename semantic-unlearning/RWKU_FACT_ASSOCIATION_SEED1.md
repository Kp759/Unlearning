# RWKU Fact-Association Residual Bank — Seed 1

## Status

This branch transfers the frozen fact-association residual-bank architecture to
the existing **RWKU-Batch-50-v1** split. It is explicitly a
`probe_assisted_cross_benchmark_method_extension`, not unchanged native RWKU
target-only training.

## Frozen protocol

- Batch seed: 1
- Five RWKU people selected by the existing deterministic Batch-50 protocol
- Ten Level-1/Level-2 forget probes per person
- Total forget training/effectiveness rows: 50
- All remaining Level-1/Level-2 content is held out
- Held-out Level-2 deterministic paraphrases are evaluation-only
- Level-3, MIA, neighbor, utility, and PPL data are evaluation-only
- Frozen Llama backbone, input embeddings, and LM head
- Layer 19 intervention at the original formatted RWKU request boundary
- Zero-initialized residual row per selected natural-input factual association
- 30 row updates per association unless explicitly overridden
- LR 0.05, target sensitive-token probability < 1e-6, 12 backtracks
- No replacement/Unknown target

RWKU does not expose symbolic relation IDs in Level-1/Level-2 rows. Therefore
the association unit is:

`(subject, natural query/context, sensitive answer)`

The sensitive answer defines the training target but is never required at
runtime. Runtime routing uses complete subject eligibility plus the nearest
frozen layer-19 contextual key. In the Batch-50 construction each selected
person owns multiple protected probes, so contextual-key disambiguation is
used rather than the unique-subject direct-activation shortcut.

## Commands

```bash
git fetch origin
git checkout fact_association_rwku_seed1
git pull origin fact_association_rwku_seed1

python -m py_compile \
  scripts/rwku_fact_association_embeddings.py \
  scripts/run_rwku_fact_association_embeddings_seed1.py \
  scripts/evaluate_rwku_fact_association_embeddings_seed1.py

pytest -q tests/test_rwku_fact_association_embeddings.py
```

Paths:

```bash
MODEL_PATH="/scratch/yl258/kp759/hf/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95"
RWKU_DATA="$PWD/data/rwku"
SPLIT="$PWD/outputs/rwku_fact_assoc_seed1_locked_split"
PREFLIGHT="$PWD/outputs/rwku_fact_assoc_seed1_preflight"
```

Preflight:

```bash
bash scripts/run_rwku_fact_association_embeddings_seed1.sh \
  "$MODEL_PATH" \
  "$RWKU_DATA" \
  "$SPLIT" \
  "$PREFLIGHT" \
  --preflight-only
```

Do not train unless:

- `correct_row_active_fraction = 1.0`
- `wrong_row_active_fraction = 0.0`
- `failure_count = 0`

Training:

```bash
RUN="$PWD/outputs/rwku_fact_assoc_seed1_train"

bash scripts/run_rwku_fact_association_embeddings_seed1.sh \
  "$MODEL_PATH" \
  "$RWKU_DATA" \
  "$SPLIT" \
  "$RUN"
```

Official Batch-50 extension evaluation:

```bash
bash scripts/evaluate_rwku_fact_association_embeddings_seed1.sh \
  "$RUN" \
  "$RWKU_DATA" \
  "$PWD/data/wikidata" \
  bfloat16
```

Headline outputs include same-50 recovery and sensitive-token top-1 accuracy,
held-out Level-1/Level-2 recovery, held-out paraphrase recovery, Level-3
adversarial recovery, neighbor locality/route activity, and runtime-aligned PPL.

## Interpretation

RWKU natively defines real-world people as unlearning targets and evaluates
forget, adversarial, neighbor, MIA, and utility behavior. This branch does not
claim native target-only training. It uses the repository's already frozen
MCF/zsRE-style Batch-50 probe-assisted extension so the residual-bank method can
be compared under the same 50-forget-example scale used in the other transfer
experiments.
