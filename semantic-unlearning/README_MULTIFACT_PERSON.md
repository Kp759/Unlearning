# Multi-fact person benchmark

Several facts about one real person in ONE sentence; forget one fact, retain
the others (advisor question: forgetting within a multi-fact context).

```
Ernest Hemingway speaks English, was born in Oak Park, and is a citizen of the United States of America.
                                             ^ forget         ^ retained, same sentence
```

```bash
# once, on the login/interactive node (compute nodes may be offline):
curl -L -o data/MQuAKE-CF.json https://raw.githubusercontent.com/princeton-nlp/MQuAKE/main/datasets/MQuAKE-CF.json
sbatch multifact_person_seed1.slurm
cat outputs/multifact_person_v1/seed1/L19/linear_global/official_multifact_eval.md
```

Method: SURE at layer 19, linear classifier fit with L-BFGS, calibrated cutoff
folded into the bias (`--decision-rule calibrated_bias`), regular mode (rows
trained under the classifier's own routing), 30 row updates per fact.

## Data (`scripts/build_multifact_person_dataset.py`, `scripts/multifact_person_data.py`)

- Facts: Wikidata triples of people from MultiCounterFact + MQuAKE-CF-3k-v2 (+ MQuAKE-CF if present), with each source's own direct cloze.
- Knowledge filter (base model only): a fact is kept if the base model gets every object token right (teacher-forced top-1) on BOTH its direct cloze and its `{S} {verb phrase}` prompt.
- One sentence per person: <= 4 facts, distinct objects, one relation per answer group (no two place / language / country facts), one fixed verb phrase per relation (object last; "the"/"a/an" handled; an alternate phrase when the primary equals the fact's own direct prompt).
- Seed 1: 50 forget people (3+-fact sentences first, then 2), one forget fact each, 100 disjoint retain people, no name containing another's.
- Training-visible: the 50 forget facts' direct cloze only (retain-blind, MQuAKE direct protocol). Retain facts, sentences and verb-phrase prompts are evaluation-only.

## Probes and metrics (`scripts/evaluate_multifact_person.py`)

Accuracy = 100 × case-macro teacher-forced object-token top-1 (official MQuAKE convention), base model and SURE on the same probes.

| role | direct | single `{S} {vp}` | multi (fact at position ≥ 1 of the sentence) |
|---|---|---|---|
| forget (lower better) | Eff (training-visible) | held-out phrasing | **held-out, other facts of the person precede it** |
| retain same person (higher) | | | **split by whether the forgotten fact is stated earlier** |
| retain other people (higher) | | | |

Plus routing (fire rate; own-row rate for forget probes), multi by position and by sentence size, and runtime-aligned PPL.

## Pipeline (`scripts/run_multifact_person_seed1.sh`, resumable)

data → `prepare_multifact_association_source.py` → `fit_linear_router.py` → `train_direct_linear_router_rows.py --dataset multifact` → evaluator. Router calibration settings are copied from the frozen MQuAKE linear router when its report exists.
