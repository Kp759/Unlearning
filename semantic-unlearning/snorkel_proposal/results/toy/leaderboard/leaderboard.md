# SCALPEL leaderboard — toy world (pipeline check, not a result)

All scores 0–100, higher is better. E = erasure on held-out forget families, L = locality, U = utility relative to base, R = robustness under attacks (lens, int8/int4, relearning). SCALPEL = harmonic mean of the available E, L, U, R. *Strict* = weights only (runtime components removed). Rows marked † are reference cheats the benchmark must defeat.

## fact · toy-llama · 24 requests

| Method | SCALPEL | E | E_min | E_direct | L | L_min | U | R | Strict | seeds |
|---|---|---|---|---|---|---|---|---|---|---|
| npo | 33.6 | 96.8 | 80.0 | 100.0 | 12.4 | 6.4 | 54.7 | 98.8 | – | 1 |
| rmu | 17.8 | 73.9 | 40.0 | 100.0 | 5.5 | 0.0 | 83.1 | 56.1 | – | 1 |
| gd | 13.9 | 97.3 | 80.0 | 100.0 | 4.1 | 0.0 | 47.9 | 92.7 | – | 1 |
| ga | 5.7 | 100.0 | 100.0 | 100.0 | 1.6 | 0.0 | 28.2 | 100.0 | – | 1 |
| answer_ban † | 0.0 | 8.3 | 0.0 | 100.0 | 6.3 | 0.0 | 72.2 | 0.0 | 2.9 | 1 |
| idk_all † | 0.0 | 100.0 | 100.0 | 100.0 | 1.0 | 0.0 | 11.1 | 0.0 | 2.7 | 1 |
| keyword_router_idk † | 0.0 | 3.1 | 0.0 | 100.0 | 58.7 | 29.2 | 100.0 | 0.0 | 2.9 | 1 |
| noop † | 0.0 | 1.0 | 0.0 | 0.0 | 100.0 | 100.0 | 100.0 | 0.0 | – | 1 |
| prompt_cache_idk † | 0.0 | 1.3 | 0.0 | 100.0 | 75.1 | 45.8 | 100.0 | 0.0 | 2.9 | 1 |
| subject_blocklist_idk † | 0.0 | 8.2 | 0.0 | 100.0 | 1.7 | 0.0 | 88.9 | 0.0 | 2.9 | 1 |

### Per family (forget: residual %, lower is better; retain: retained %, higher is better)

| Method | F:context_prefix | F:crosslingual | F:direct | F:indirect | F:multifact | F:multihop | F:multiturn | F:paraphrase | F:question | F:surface | R:in_context | R:name_collision | R:same_object | R:same_relation | R:same_sentence | R:same_subject | R:subject_mentioned |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| npo | 0 | 0 | 0 | 20 | 0 | 0 | 0 | 0 | 4 | 0 | 17 | 11 | 14 | 20 | 15 | 15 | 6 |
| rmu | 6 | 10 | 0 | 60 | 0 | 40 | 29 | 11 | 12 | 13 | 83 | 22 | 29 | 35 | 0 | 10 | 23 |
| gd | 0 | 0 | 0 | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 11 | 0 | 15 | 11 | 13 | 6 |
| ga | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 8 | 0 | 0 | 0 | 6 | 5 | 0 |
| answer_ban | 0 | 0 | 0 | 0 | 0 | 100 | 0 | 0 | 0 | 0 | 42 | 67 | 0 | 55 | 51 | 48 | 45 |
| idk_all | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| keyword_router_idk | 0 | 99 | 0 | 100 | 30 | 0 | 46 | 86 | 46 | 100 | 29 | 100 | 100 | 100 | 30 | 100 | 87 |
| noop | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| prompt_cache_idk | 0 | 100 | 0 | 100 | 100 | 0 | 100 | 100 | 100 | 100 | 46 | 100 | 100 | 100 | 47 | 100 | 100 |
| subject_blocklist_idk | 0 | 0 | 0 | 60 | 0 | 0 | 0 | 0 | 0 | 100 | 0 | 100 | 100 | 100 | 0 | 0 | 0 |

### Legacy view vs SCALPEL (same run)

Legacy view = only the families the official MCF / zsRE metrics use: erasure on the request and its paraphrases (Eff/Gen), retention on same-answer neighbours and unrelated questions (Spe). A shortcut that looks perfect here and collapses under SCALPEL is what the legacy metrics reward.

| Method | legacy erasure | legacy locality | SCALPEL E | SCALPEL L | SCALPEL |
|---|---|---|---|---|---|
| npo | 100.0 | 14.3 | 96.8 | 12.4 | 33.6 |
| rmu | 94.3 | 28.6 | 73.9 | 5.5 | 17.8 |
| gd | 100.0 | 0.0 | 97.3 | 4.1 | 13.9 |
| ga | 100.0 | 0.0 | 100.0 | 1.6 | 5.7 |
| answer_ban † | 100.0 | 0.0 | 8.3 | 6.3 | 0.0 |
| idk_all † | 100.0 | 0.0 | 100.0 | 1.0 | 0.0 |
| keyword_router_idk † | 57.1 | 100.0 | 3.1 | 58.7 | 0.0 |
| noop † | 0.0 | 100.0 | 1.0 | 100.0 | 0.0 |
| prompt_cache_idk † | 50.0 | 100.0 | 1.3 | 75.1 | 0.0 |
| subject_blocklist_idk † | 100.0 | 100.0 | 8.2 | 1.7 | 0.0 |

### Attacks (residual on forget facts after the attack, relative to the base model after the same attack; lower is better)

| Method | int8 | int4 | relearn (held-out half) | logit lens (upper half, chance-corrected) |
|---|---|---|---|---|
| npo | 1.2 | 0.0 | 0.0 | 0.0 |
| rmu | 8.4 | 9.6 | 43.9 | 30.4 |
| gd | 0.0 | 0.0 | 7.3 | 0.0 |
| ga | 0.0 | 0.0 | 0.0 | 0.0 |
| answer_ban | 0.0 | 0.0 | 0.0 | 100.0 |
| idk_all | 0.0 | 0.0 | 0.0 | 100.0 |
| keyword_router_idk | 49.4 | 49.4 | 51.2 | 100.0 |
| noop | 100.0 | 100.0 | 100.0 | 100.0 |
| prompt_cache_idk | 71.1 | 71.1 | 73.2 | 100.0 |
| subject_blocklist_idk | 0.0 | 0.0 | 0.0 | 100.0 |

### Open-ended behaviour (greedy generation; probes the base model answers in generation)

| Method | forget: leak | forget: refusal | forget: other (wrong, empty, degenerate) | retain: correct | retain: false refusal |
|---|---|---|---|---|---|
| npo | 0.7 | 0.0 | 99.3 | 13.4 | 0.0 |
| rmu | 19.2 | 0.0 | 80.8 | 25.5 | 0.0 |
| gd | 0.3 | 0.0 | 99.7 | 9.7 | 0.0 |
| ga | 0.0 | 0.0 | 100.0 | 4.2 | 0.0 |
| answer_ban | 1.6 | 0.0 | 98.4 | 47.2 | 0.0 |
| idk_all | 0.0 | 100.0 | 0.0 | 0.0 | 100.0 |
| keyword_router_idk | 53.7 | 46.3 | 0.0 | 74.1 | 25.9 |
| noop | 100.0 | 0.0 | 0.0 | 100.0 | 0.0 |
| prompt_cache_idk | 74.9 | 25.1 | 0.0 | 82.4 | 17.6 |
| subject_blocklist_idk | 8.5 | 91.5 | 0.0 | 16.7 | 83.3 |

### Detect-then-edit decomposition

| Method | route recall (forget) | residual from routing misses | residual when routed | false-fire rate (retain) |
|---|---|---|---|---|
| keyword_router_idk | 46.3 | 53.7 | 0.0 | 25.9 |

### Cost

| Method | unlearn seconds | inference overhead (× base) |
|---|---|---|
| npo | 1.5 | 0.38 |
| rmu | 8.5 | 0.45 |
| gd | 1.9 | 0.67 |
| ga | 0.6 | 0.77 |
| answer_ban | 0.0 | 0.57 |
| idk_all | 0.0 | 0.50 |
| keyword_router_idk | 0.0 | 0.66 |
| noop | 0.0 | 0.59 |
| prompt_cache_idk | 0.0 | 1.00 |
| subject_blocklist_idk | 0.0 | 0.67 |



All scores 0–100, higher is better. E = erasure on held-out forget families, L = locality, U = utility relative to base, R = robustness under attacks (lens, int8/int4, relearning). SCALPEL = harmonic mean of the available E, L, U, R. *Strict* = weights only (runtime components removed). Rows marked † are reference cheats the benchmark must defeat.

## fact · toy-llama · 24 requests

| Method | SCALPEL | E | E_min | E_direct | L | L_min | U | R | Strict | seeds |
|---|---|---|---|---|---|---|---|---|---|---|
| rmu | 36.5 | 68.9 | 40.0 | 70.8 | 15.3 | 4.3 | 83.1 | 56.1 | – | 1 |
| npo | 33.6 | 96.8 | 80.0 | 100.0 | 12.4 | 6.4 | 54.7 | 98.8 | – | 1 |
| gd | 13.9 | 97.3 | 80.0 | 100.0 | 4.1 | 0.0 | 47.9 | 92.7 | – | 1 |
| ga | 6.9 | 100.0 | 100.0 | 100.0 | 1.9 | 0.0 | 28.2 | 100.0 | – | 1 |
| answer_ban † | 0.0 | 8.3 | 0.0 | 100.0 | 6.3 | 0.0 | 72.2 | 0.0 | 2.9 | 1 |
| idk_all † | 0.0 | 100.0 | 100.0 | 100.0 | 1.0 | 0.0 | 11.1 | 0.0 | 2.7 | 1 |
| keyword_router_idk † | 0.0 | 3.1 | 0.0 | 100.0 | 58.7 | 29.2 | 100.0 | 0.0 | 2.9 | 1 |
| noop † | 0.0 | 1.0 | 0.0 | 0.0 | 100.0 | 100.0 | 100.0 | 0.0 | – | 1 |
| prompt_cache_idk † | 0.0 | 1.3 | 0.0 | 100.0 | 75.1 | 45.8 | 100.0 | 0.0 | 2.9 | 1 |
| subject_blocklist_idk † | 0.0 | 8.2 | 0.0 | 100.0 | 1.7 | 0.0 | 88.9 | 0.0 | 2.9 | 1 |

### Per family (forget: residual %, lower is better; retain: retained %, higher is better)

| Method | F:context_prefix | F:crosslingual | F:direct | F:indirect | F:multifact | F:multihop | F:multiturn | F:paraphrase | F:question | F:surface | R:in_context | R:name_collision | R:same_object | R:same_relation | R:same_sentence | R:same_subject | R:subject_mentioned |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rmu | 15 | 14 | 29 | 60 | 9 | 40 | 29 | 17 | 17 | 39 | 83 | 22 | 43 | 40 | 4 | 13 | 26 |
| npo | 0 | 0 | 0 | 20 | 0 | 0 | 0 | 0 | 4 | 0 | 17 | 11 | 14 | 20 | 15 | 15 | 6 |
| gd | 0 | 0 | 0 | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 4 | 11 | 0 | 15 | 11 | 13 | 6 |
| ga | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 8 | 0 | 0 | 5 | 6 | 5 | 0 |
| answer_ban | 0 | 0 | 0 | 0 | 0 | 100 | 0 | 0 | 0 | 0 | 42 | 67 | 0 | 55 | 51 | 48 | 45 |
| idk_all | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| keyword_router_idk | 0 | 99 | 0 | 100 | 30 | 0 | 46 | 86 | 46 | 100 | 29 | 100 | 100 | 100 | 30 | 100 | 87 |
| noop | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| prompt_cache_idk | 0 | 100 | 0 | 100 | 100 | 0 | 100 | 100 | 100 | 100 | 46 | 100 | 100 | 100 | 47 | 100 | 100 |
| subject_blocklist_idk | 0 | 0 | 0 | 60 | 0 | 0 | 0 | 0 | 0 | 100 | 0 | 100 | 100 | 100 | 0 | 0 | 0 |

### Legacy view vs SCALPEL (same run)

Legacy view = only the families the official MCF / zsRE metrics use: erasure on the request and its paraphrases (Eff/Gen), retention on same-answer neighbours and unrelated questions (Spe). A shortcut that looks perfect here and collapses under SCALPEL is what the legacy metrics reward.

| Method | legacy erasure | legacy locality | SCALPEL E | SCALPEL L | SCALPEL |
|---|---|---|---|---|---|
| rmu | 76.8 | 42.9 | 68.9 | 15.3 | 36.5 |
| npo | 100.0 | 14.3 | 96.8 | 12.4 | 33.6 |
| gd | 100.0 | 0.0 | 97.3 | 4.1 | 13.9 |
| ga | 100.0 | 0.0 | 100.0 | 1.9 | 6.9 |
| answer_ban † | 100.0 | 0.0 | 8.3 | 6.3 | 0.0 |
| idk_all † | 100.0 | 0.0 | 100.0 | 1.0 | 0.0 |
| keyword_router_idk † | 57.1 | 100.0 | 3.1 | 58.7 | 0.0 |
| noop † | 0.0 | 100.0 | 1.0 | 100.0 | 0.0 |
| prompt_cache_idk † | 50.0 | 100.0 | 1.3 | 75.1 | 0.0 |
| subject_blocklist_idk † | 100.0 | 100.0 | 8.2 | 1.7 | 0.0 |

### Attacks (residual on forget facts after the attack, relative to the base model after the same attack; lower is better)

| Method | int8 | int4 | relearn (held-out half) | logit lens (upper half, chance-corrected) |
|---|---|---|---|---|
| rmu | 8.4 | 9.6 | 43.9 | 30.4 |
| npo | 1.2 | 0.0 | 0.0 | 0.0 |
| gd | 0.0 | 0.0 | 7.3 | 0.0 |
| ga | 0.0 | 0.0 | 0.0 | 0.0 |
| answer_ban | 0.0 | 0.0 | 0.0 | 100.0 |
| idk_all | 0.0 | 0.0 | 0.0 | 100.0 |
| keyword_router_idk | 49.4 | 49.4 | 51.2 | 100.0 |
| noop | 100.0 | 100.0 | 100.0 | 100.0 |
| prompt_cache_idk | 71.1 | 71.1 | 73.2 | 100.0 |
| subject_blocklist_idk | 0.0 | 0.0 | 0.0 | 100.0 |

### Open-ended behaviour (greedy generation; probes the base model answers in generation)

| Method | forget: leak | forget: refusal | forget: other (wrong, empty, degenerate) | retain: correct | retain: false refusal |
|---|---|---|---|---|---|
| rmu | 19.2 | 0.0 | 80.8 | 25.5 | 0.0 |
| npo | 0.7 | 0.0 | 99.3 | 13.4 | 0.0 |
| gd | 0.3 | 0.0 | 99.7 | 9.7 | 0.0 |
| ga | 0.0 | 0.0 | 100.0 | 4.2 | 0.0 |
| answer_ban | 1.6 | 0.0 | 98.4 | 47.2 | 0.0 |
| idk_all | 0.0 | 100.0 | 0.0 | 0.0 | 100.0 |
| keyword_router_idk | 53.7 | 46.3 | 0.0 | 74.1 | 25.9 |
| noop | 100.0 | 0.0 | 0.0 | 100.0 | 0.0 |
| prompt_cache_idk | 74.9 | 25.1 | 0.0 | 82.4 | 17.6 |
| subject_blocklist_idk | 8.5 | 91.5 | 0.0 | 16.7 | 83.3 |

### Detect-then-edit decomposition

| Method | route recall (forget) | residual from routing misses | residual when routed | false-fire rate (retain) |
|---|---|---|---|---|
| keyword_router_idk | 46.3 | 53.7 | 0.0 | 25.9 |

### Cost

| Method | unlearn seconds | inference overhead (× base) |
|---|---|---|
| rmu | 8.5 | 0.45 |
| npo | 1.5 | 0.38 |
| gd | 1.9 | 0.67 |
| ga | 0.6 | 0.77 |
| answer_ban | 0.0 | 0.57 |
| idk_all | 0.0 | 0.50 |
| keyword_router_idk | 0.0 | 0.66 |
| noop | 0.0 | 0.59 |
| prompt_cache_idk | 0.0 | 1.00 |
| subject_blocklist_idk | 0.0 | 0.67 |

