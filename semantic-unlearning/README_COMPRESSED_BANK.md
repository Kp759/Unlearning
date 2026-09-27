# Compressed bank trained in the loop (MCF, layer 19, linear router, no V2)

Post-hoc compression (`README_RESIDUAL_COMPRESSION.md`) fits a compact form to
rows that were trained per fact. Here the compact form **is** the model: the
unlearning objective optimizes the compact parameters directly, so shared
pieces are learned for forgetting.

```bash
sbatch mcf_compressed_values.slurm   # one router head per fact; compressed values
sbatch mcf_compressed_c1.slurm       # one router head per RELATION; compressed values
# table: outputs/mcf_compressed_v1/L19/compressed_summary.md
```

## Router (keys): `fit_linear_router.py --head-sharing relation`

One logistic head per relation instead of per fact. The subject-token match
(unchanged) picks the entity; the relation head picks which of its facts. The
bank stores R heads + a per-fact head index (`head_index`), so the router grows
with the number of relations, not facts. Two facts with the same subject and
relation would get identical logits (reported as `same_subject_same_relation_pairs`;
none expected on MCF seed 1). Global threshold only.

## Values: `train_mcf_compressed_bank.py --value-mode ...`

| Mode | Row for fact i | Stored per fact | Shared |
|---|---|---|---|
| `full` | P[i] | d | – (joint-trainer reference) |
| `lowrank:K` | codes[i] @ basis | K | K×d |
| `tied_answer` | s_i · D[answer(i)] | 1 | one d-vector per distinct answer |
| `tied_relation` | s_i · D[relation(i)] | 1 | one d-vector per relation |
| `answer_fixed` (A6) | s_i · u(answer(i)) | 1 | nothing |
| `answer_map:r` (A6+) | s_i · (u + uAB) | 1 | 2·d·r |
| `relation_plus_answer` (C1) | a_i · D[relation(i)] + b_i · u(answer(i)) | 2 | one d-vector per relation |

u(answer) = −normalize(output embedding of the fact's first answer token): a
fixed direction from the frozen model (logit lens), so `answer_fixed` learns one
scalar per fact and nothing else.

Training: Adam on the compact parameters (`--lr` for vectors, `--scale-lr` for
scalars/codes), facts in minibatches (`--batch-facts 8`), shipped per-fact
objective (hardest-view answer NLL hinge at 1e-6 + "I don't know" NLL), shipped
checkpoint rule on training-visible train + development views, stop after
feasibility holds for 3 checkpoints or at the 3600 s cap. Rows are routed by
the linear classifier during training (`--training-route oracle` for genie).
Facts the classifier never routes keep an exactly-zero row.

The saved artifact is a normal linear-router artifact whose rows are rebuilt on
CPU from the saved compact state (checked against the trained rows), so the
official evaluator runs unchanged; `compressed_values` holds the compact state
and the storage report.

## Configs

| Job | Router | Value modes |
|---|---|---|
| `mcf_compressed_values.slurm` | per fact (50 heads) | full, lowrank:8, lowrank:32, tied_answer, answer_fixed, answer_map:64 |
| `mcf_compressed_c1.slurm` | per relation | full, tied_relation, relation_plus_answer, answer_fixed, answer_map:64 |

Read in this order:
1. `router_fact/full` vs the shipped reference: the joint trainer alone (no compression).
2. `router_relation/full` vs `router_fact/full`: cost of relation-shared heads.
3. Each value mode vs `full` on the same router: cost of that value compression.
4. `answer_fixed` succeeding would mean one scalar per fact suffices.

At N = 50 the storage numbers show the structure, not the savings; the
`value_floats_at_100K` column is the extrapolation for modes whose shared part
is independent of N. The N-scaling runs are what test it.
