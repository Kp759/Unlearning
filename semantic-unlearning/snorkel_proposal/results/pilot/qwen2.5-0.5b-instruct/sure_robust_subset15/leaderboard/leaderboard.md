# SURE vs SURE-R — Qwen2.5-0.5B, 15 pilot requests (local check)

All scores 0–100, higher is better. E = erasure on held-out forget families, L = locality, U = utility relative to base, R = robustness under attacks (lens, int8/int4, relearning). SCALPEL = harmonic mean of the available E, L, U, R. *Strict* = weights only (runtime components removed). Rows marked † are reference cheats the benchmark must defeat.

## fact · Qwen2.5-0.5B-Instruct · 15 requests

| Method | SCALPEL | E | E_min | E_direct | L | L_min | U | R | Strict | seeds |
|---|---|---|---|---|---|---|---|---|---|---|
| sure_r | 72.2 | 66.2 | 50.0 | 100.0 | 97.0 | 86.7 | 100.0 | 50.0 | 2.9 | 1 |
| sure_r_hardneg | 8.1 | 2.3 | 0.0 | 100.0 | 100.0 | 100.0 | 100.0 | 27.8 | 2.9 | 1 |
| sure | 0.0 | 2.9 | 0.0 | 33.3 | 100.0 | 100.0 | 100.0 | 0.0 | 2.9 | 1 |

### Per family (forget: residual %, lower is better; retain: retained %, higher is better)

| Method | F:context_prefix | F:crosslingual | F:direct | F:multifact | F:multihop | F:multiturn | F:paraphrase | F:question | F:surface | R:disjoint | R:in_context | R:name_collision | R:same_object | R:same_relation | R:same_sentence | R:subject_mentioned |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sure_r | 0 | 50 | 0 | 50 | 100 | 43 | 13 | 40 | 0 | 100 | 87 | 100 | 100 | 100 | 100 | 100 |
| sure_r_hardneg | 0 | 100 | 0 | 50 | 100 | 100 | 33 | 100 | 0 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| sure | 76 | 94 | 67 | 100 | 100 | 86 | 60 | 90 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |

### Legacy view vs SCALPEL (same run)

Legacy view = only the families the official MCF / zsRE metrics use: erasure on the request and its paraphrases (Eff/Gen), retention on same-answer neighbours and unrelated questions (Spe). A shortcut that looks perfect here and collapses under SCALPEL is what the legacy metrics reward.

| Method | legacy erasure | legacy locality | SCALPEL E | SCALPEL L | SCALPEL |
|---|---|---|---|---|---|
| sure_r | 93.3 | 100.0 | 66.2 | 97.0 | 72.2 |
| sure_r_hardneg | 83.3 | 100.0 | 2.3 | 100.0 | 8.1 |
| sure | 36.7 | 100.0 | 2.9 | 100.0 | 0.0 |

### Attacks (residual on forget facts after the attack, relative to the base model after the same attack; lower is better)

| Method | int8 | int4 | relearn (held-out half) | logit lens (upper half, chance-corrected) |
|---|---|---|---|---|
| sure_r | – | – | – | 50.0 |
| sure_r_hardneg | 37.5 | 12.5 | – | 72.2 |
| sure | – | – | – | 100.0 |

### Open-ended behaviour (greedy generation; probes the base model answers in generation)

| Method | forget: leak | forget: refusal | forget: other (wrong, empty, degenerate) | retain: correct | retain: false refusal |
|---|---|---|---|---|---|
| sure_r | 48.0 | 0.0 | 52.0 | 100.0 | 0.0 |
| sure_r_hardneg | 65.0 | 0.0 | 35.0 | 100.0 | 0.0 |
| sure | 86.0 | 0.0 | 14.0 | 100.0 | 0.0 |

### Detect-then-edit decomposition

| Method | route recall (forget) | residual from routing misses | residual when routed | false-fire rate (retain) |
|---|---|---|---|---|
| sure_r | 76.0 | 24.0 | 0.0 | 11.3 |
| sure_r_hardneg | 56.0 | 44.0 | 1.0 | 0.0 |
| sure | 65.0 | 35.0 | 46.0 | 7.0 |

### Cost

| Method | unlearn seconds | inference overhead (× base) |
|---|---|---|
| sure_r | 803.3 | 1.27 |
| sure_r_hardneg | 124.7 | 1.20 |
| sure | 486.0 | 1.22 |



All scores 0–100, higher is better. E = erasure on held-out forget families, L = locality, U = utility relative to base, R = robustness under attacks (lens, int8/int4, relearning). SCALPEL = harmonic mean of the available E, L, U, R. *Strict* = weights only (runtime components removed). Rows marked † are reference cheats the benchmark must defeat.

## fact · Qwen2.5-0.5B-Instruct · 15 requests

| Method | SCALPEL | E | E_min | E_direct | L | L_min | U | R | Strict | seeds |
|---|---|---|---|---|---|---|---|---|---|---|
| sure_r | 63.2 | 42.9 | 25.0 | 60.0 | 100.0 | 100.0 | 100.0 | 50.0 | 2.9 | 1 |
| sure_r_hardneg | 8.0 | 2.3 | 0.0 | 60.0 | 100.0 | 100.0 | 100.0 | 27.8 | 2.9 | 1 |
| sure | 0.0 | 2.9 | 0.0 | 20.0 | 100.0 | 100.0 | 100.0 | 0.0 | 2.9 | 1 |

### Per family (forget: residual %, lower is better; retain: retained %, higher is better)

| Method | F:context_prefix | F:crosslingual | F:direct | F:multifact | F:multihop | F:multiturn | F:paraphrase | F:question | F:surface | R:disjoint | R:in_context | R:name_collision | R:same_object | R:same_relation | R:same_sentence | R:subject_mentioned |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sure_r | 29 | 56 | 40 | 75 | 100 | 71 | 47 | 50 | 20 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| sure_r_hardneg | 33 | 100 | 40 | 75 | 100 | 100 | 60 | 100 | 20 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| sure | 81 | 94 | 80 | 100 | 100 | 86 | 73 | 90 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |

### Legacy view vs SCALPEL (same run)

Legacy view = only the families the official MCF / zsRE metrics use: erasure on the request and its paraphrases (Eff/Gen), retention on same-answer neighbours and unrelated questions (Spe). A shortcut that looks perfect here and collapses under SCALPEL is what the legacy metrics reward.

| Method | legacy erasure | legacy locality | SCALPEL E | SCALPEL L | SCALPEL |
|---|---|---|---|---|---|
| sure_r | 56.7 | 100.0 | 42.9 | 100.0 | 63.2 |
| sure_r_hardneg | 50.0 | 100.0 | 2.3 | 100.0 | 8.0 |
| sure | 23.3 | 100.0 | 2.9 | 100.0 | 0.0 |

### Attacks (residual on forget facts after the attack, relative to the base model after the same attack; lower is better)

| Method | int8 | int4 | relearn (held-out half) | logit lens (upper half, chance-corrected) |
|---|---|---|---|---|
| sure_r | – | – | – | 50.0 |
| sure_r_hardneg | 37.5 | 12.5 | – | 72.2 |
| sure | – | – | – | 100.0 |

### Open-ended behaviour (greedy generation; probes the base model answers in generation)

| Method | forget: leak | forget: refusal | forget: other (wrong, empty, degenerate) | retain: correct | retain: false refusal |
|---|---|---|---|---|---|
| sure_r | 48.0 | 0.0 | 52.0 | 100.0 | 0.0 |
| sure_r_hardneg | 65.0 | 0.0 | 35.0 | 100.0 | 0.0 |
| sure | 86.0 | 0.0 | 14.0 | 100.0 | 0.0 |

### Detect-then-edit decomposition

| Method | route recall (forget) | residual from routing misses | residual when routed | false-fire rate (retain) |
|---|---|---|---|---|
| sure_r | 76.0 | 24.0 | 0.0 | 11.3 |
| sure_r_hardneg | 56.0 | 44.0 | 1.0 | 0.0 |
| sure | 65.0 | 35.0 | 46.0 | 7.0 |

### Cost

| Method | unlearn seconds | inference overhead (× base) |
|---|---|---|
| sure_r | 803.3 | 1.27 |
| sure_r_hardneg | 124.7 | 1.20 |
| sure | 486.0 | 1.22 |

