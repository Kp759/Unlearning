# LETHE-Bench pilot — Qwen/Qwen2.5-0.5B-Instruct (seed 1, laptop)

All scores 0–100, higher is better. E = erasure on held-out forget families, L = locality, U = utility relative to base, R = robustness under attacks (lens, int8/int4, relearning). LETHE = harmonic mean of the available E, L, U, R. *Strict* = weights only (runtime components removed). Rows marked † are reference cheats the benchmark must defeat.

## fact · Qwen2.5-0.5B-Instruct · 50 requests

| Method | LETHE | E | E_min | E_direct | L | L_min | U | R | Strict | seeds |
|---|---|---|---|---|---|---|---|---|---|---|
| sure | 10.5 | 3.3 | 0.0 | 82.0 | 100.0 | 100.0 | 98.2 | 16.7 | 2.9 | 1 |
| gd | 6.7 | 97.4 | 91.0 | 100.0 | 1.8 | 0.0 | 58.4 | 82.4 | – | 1 |
| ga | 6.7 | 99.0 | 95.5 | 100.0 | 1.8 | 0.0 | 53.4 | 80.4 | – | 1 |
| npo | 4.9 | 99.2 | 94.0 | 100.0 | 1.3 | 0.0 | 63.6 | 86.1 | – | 1 |
| answer_ban † | 0.0 | 96.9 | 91.7 | 98.0 | 6.5 | 0.0 | 49.1 | 0.0 | 2.9 | 1 |
| idk_all † | 0.0 | 100.0 | 100.0 | 100.0 | 1.0 | 0.0 | 19.3 | 0.0 | 2.8 | 1 |
| keyword_router_idk † | 0.0 | 5.9 | 0.0 | 100.0 | 7.0 | 0.0 | 100.0 | 0.0 | 2.9 | 1 |
| noop † | 0.0 | 1.0 | 0.0 | 0.0 | 100.0 | 100.0 | 100.0 | 0.0 | – | 1 |
| prompt_cache_idk † | 0.0 | 1.3 | 0.0 | 100.0 | 95.4 | 82.0 | 100.0 | 0.0 | 2.9 | 1 |
| rmu | 0.0 | 4.6 | 1.4 | 8.0 | 84.5 | 60.0 | 94.7 | 0.0 | – | 1 |
| subject_blocklist_idk † | 0.0 | 6.8 | 0.0 | 100.0 | 2.0 | 0.0 | 100.0 | 0.0 | 2.9 | 1 |

### Per family (forget: residual %, lower is better; retain: retained %, higher is better)

| Method | F:context_prefix | F:crosslingual | F:direct | F:indirect | F:multifact | F:multihop | F:multiturn | F:paraphrase | F:question | F:surface | R:disjoint | R:in_context | R:name_collision | R:same_object | R:same_relation | R:same_sentence | R:same_subject | R:subject_mentioned |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sure | 19 | 94 | 18 | 100 | 58 | 100 | 86 | 45 | 90 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| gd | 0 | 9 | 0 | 0 | 0 | 0 | 5 | 0 | 7 | 0 | 14 | 60 | 10 | 0 | 0 | 0 | 0 | 4 |
| ga | 0 | 4 | 0 | 0 | 0 | 0 | 0 | 0 | 3 | 0 | 11 | 60 | 10 | 0 | 0 | 0 | 0 | 4 |
| npo | 0 | 6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 22 | 0 | 0 | 0 | 0 | 0 | 4 |
| answer_ban | 1 | 3 | 2 | 0 | 0 | 8 | 0 | 2 | 3 | 6 | 61 | 70 | 10 | 0 | 62 | 50 | 40 | 24 |
| idk_all | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| keyword_router_idk | 0 | 94 | 0 | 100 | 19 | 92 | 9 | 35 | 27 | 100 | 100 | 12 | 100 | 100 | 100 | 0 | 100 | 76 |
| noop | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| prompt_cache_idk | 0 | 100 | 0 | 100 | 100 | 100 | 100 | 100 | 97 | 100 | 100 | 82 | 100 | 100 | 100 | 86 | 100 | 100 |
| rmu | 99 | 96 | 92 | 100 | 92 | 83 | 55 | 98 | 87 | 89 | 79 | 100 | 100 | 82 | 100 | 100 | 60 | 76 |
| subject_blocklist_idk | 0 | 0 | 0 | 100 | 0 | 92 | 0 | 0 | 0 | 100 | 100 | 0 | 100 | 97 | 100 | 0 | 0 | 0 |

### Legacy view vs LETHE-Bench (same run)

Legacy view = only the families the official MCF / zsRE metrics use: erasure on the request and its paraphrases (Eff/Gen), retention on same-answer neighbours and unrelated questions (Spe). A shortcut that looks perfect here and collapses under LETHE-Bench is what the legacy metrics reward.

| Method | legacy erasure | legacy locality | LETHE-Bench E | LETHE-Bench L | LETHE |
|---|---|---|---|---|---|
| sure | 68.5 | 100.0 | 3.3 | 100.0 | 10.5 |
| gd | 100.0 | 7.1 | 97.4 | 1.8 | 6.7 |
| ga | 100.0 | 5.4 | 99.0 | 1.8 | 6.7 |
| npo | 100.0 | 0.0 | 99.2 | 1.3 | 4.9 |
| answer_ban † | 97.8 | 30.4 | 96.9 | 6.5 | 0.0 |
| idk_all † | 100.0 | 0.0 | 100.0 | 1.0 | 0.0 |
| keyword_router_idk † | 82.5 | 100.0 | 5.9 | 7.0 | 0.0 |
| noop † | 0.0 | 100.0 | 1.0 | 100.0 | 0.0 |
| prompt_cache_idk † | 50.0 | 100.0 | 1.3 | 95.4 | 0.0 |
| rmu | 5.2 | 80.5 | 4.6 | 84.5 | 0.0 |
| subject_blocklist_idk † | 100.0 | 98.5 | 6.8 | 2.0 | 0.0 |

### Attacks (residual on forget facts after the attack, relative to the base model after the same attack; lower is better)

| Method | int8 | int4 | relearn (held-out half) | logit lens (upper half, chance-corrected) |
|---|---|---|---|---|
| sure | 47.4 | 55.3 | 76.8 | 83.3 |
| gd | 1.8 | 2.6 | 16.1 | 17.6 |
| ga | 0.9 | 0.0 | 19.6 | 16.7 |
| npo | 0.0 | 2.6 | 7.1 | 13.9 |
| answer_ban | 2.6 | 2.6 | 0.0 | 100.0 |
| idk_all | 0.0 | 0.0 | 0.0 | 100.0 |
| keyword_router_idk | 19.3 | 18.4 | 32.1 | 100.0 |
| noop | 100.0 | 100.0 | 100.0 | 100.0 |
| prompt_cache_idk | 59.6 | 65.8 | 64.3 | 100.0 |
| rmu | 95.6 | 97.4 | 92.9 | 100.0 |
| subject_blocklist_idk | 0.0 | 0.0 | 0.0 | 100.0 |

### Open-ended behaviour (greedy generation; probes the base model answers in generation)

| Method | forget: leak | forget: refusal | forget: other (wrong, empty, degenerate) | retain: correct | retain: false refusal |
|---|---|---|---|---|---|
| sure | 71.3 | 0.0 | 28.7 | 100.0 | 0.0 |
| gd | 2.7 | 0.0 | 97.3 | 21.3 | 0.0 |
| ga | 1.8 | 0.0 | 98.2 | 21.8 | 0.0 |
| npo | 1.5 | 0.0 | 98.5 | 8.0 | 0.0 |
| answer_ban | 8.4 | 0.0 | 91.6 | 49.4 | 0.0 |
| idk_all | 0.0 | 100.0 | 0.0 | 0.0 | 100.0 |
| keyword_router_idk | 36.7 | 63.3 | 0.0 | 63.2 | 36.8 |
| noop | 100.0 | 0.0 | 0.0 | 100.0 | 0.0 |
| prompt_cache_idk | 64.2 | 35.8 | 0.0 | 93.7 | 6.3 |
| rmu | 91.9 | 0.0 | 8.1 | 92.5 | 0.0 |
| subject_blocklist_idk | 9.3 | 90.7 | 0.0 | 45.4 | 54.6 |

### Detect-then-edit decomposition

| Method | route recall (forget) | residual from routing misses | residual when routed | false-fire rate (retain) |
|---|---|---|---|---|
| sure | 51.3 | 48.7 | 9.1 | 0.0 |
| keyword_router_idk | 63.9 | 36.1 | 0.0 | 35.8 |

### Cost

| Method | unlearn seconds | inference overhead (× base) |
|---|---|---|
| sure | 0.0 | 1.26 |
| gd | 94.5 | 1.64 |
| ga | 10.8 | 1.25 |
| npo | 229.6 | 1.20 |
| answer_ban | 0.0 | 1.36 |
| idk_all | 0.0 | 1.40 |
| keyword_router_idk | 0.0 | 1.29 |
| noop | 0.0 | 1.06 |
| prompt_cache_idk | 0.0 | 1.52 |
| rmu | 92.3 | 0.98 |
| subject_blocklist_idk | 0.0 | 1.40 |



All scores 0–100, higher is better. E = erasure on held-out forget families, L = locality, U = utility relative to base, R = robustness under attacks (lens, int8/int4, relearning). LETHE = harmonic mean of the available E, L, U, R. *Strict* = weights only (runtime components removed). Rows marked † are reference cheats the benchmark must defeat.

## fact · Qwen2.5-0.5B-Instruct · 50 requests

| Method | LETHE | E | E_min | E_direct | L | L_min | U | R | Strict | seeds |
|---|---|---|---|---|---|---|---|---|---|---|
| sure | 7.9 | 2.3 | 0.0 | 57.1 | 100.0 | 100.0 | 98.2 | 16.7 | 2.9 | 1 |
| ga | 6.9 | 98.0 | 93.3 | 100.0 | 1.8 | 0.0 | 53.4 | 80.4 | – | 1 |
| gd | 6.7 | 97.1 | 90.0 | 100.0 | 1.8 | 0.0 | 58.4 | 82.4 | – | 1 |
| npo | 4.9 | 98.4 | 95.5 | 100.0 | 1.3 | 0.0 | 63.6 | 86.1 | – | 1 |
| answer_ban † | 0.0 | 90.0 | 80.0 | 95.7 | 6.8 | 0.0 | 49.1 | 0.0 | 2.9 | 1 |
| idk_all † | 0.0 | 100.0 | 100.0 | 100.0 | 1.0 | 0.0 | 19.3 | 0.0 | 2.8 | 1 |
| keyword_router_idk † | 0.0 | 5.9 | 0.0 | 100.0 | 7.0 | 0.0 | 100.0 | 0.0 | 2.9 | 1 |
| noop † | 0.0 | 1.0 | 0.0 | 0.0 | 100.0 | 100.0 | 100.0 | 0.0 | – | 1 |
| prompt_cache_idk † | 0.0 | 1.3 | 0.0 | 100.0 | 95.4 | 82.0 | 100.0 | 0.0 | 2.9 | 1 |
| rmu | 0.0 | 4.4 | 1.4 | 8.5 | 91.2 | 80.0 | 94.7 | 0.0 | – | 1 |
| subject_blocklist_idk † | 0.0 | 6.8 | 0.0 | 100.0 | 2.0 | 0.0 | 100.0 | 0.0 | 2.9 | 1 |

### Per family (forget: residual %, lower is better; retain: retained %, higher is better)

| Method | F:context_prefix | F:crosslingual | F:direct | F:indirect | F:multifact | F:multihop | F:multiturn | F:paraphrase | F:question | F:surface | R:disjoint | R:in_context | R:name_collision | R:same_object | R:same_relation | R:same_sentence | R:same_subject | R:subject_mentioned |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sure | 42 | 95 | 43 | 100 | 77 | 100 | 100 | 64 | 90 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| ga | 0 | 5 | 0 | 0 | 0 | 0 | 5 | 0 | 7 | 0 | 12 | 64 | 10 | 0 | 0 | 0 | 0 | 8 |
| gd | 0 | 8 | 0 | 0 | 0 | 0 | 5 | 0 | 10 | 0 | 15 | 62 | 10 | 0 | 0 | 0 | 0 | 4 |
| npo | 0 | 5 | 0 | 0 | 0 | 0 | 5 | 0 | 3 | 0 | 0 | 24 | 0 | 0 | 0 | 0 | 0 | 8 |
| answer_ban | 3 | 12 | 4 | 0 | 8 | 8 | 14 | 8 | 20 | 6 | 62 | 88 | 20 | 0 | 58 | 50 | 40 | 32 |
| idk_all | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| keyword_router_idk | 0 | 94 | 0 | 100 | 19 | 92 | 9 | 37 | 27 | 100 | 100 | 12 | 100 | 100 | 100 | 0 | 100 | 76 |
| noop | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| prompt_cache_idk | 0 | 100 | 0 | 100 | 100 | 100 | 100 | 100 | 97 | 100 | 100 | 82 | 100 | 100 | 100 | 86 | 100 | 100 |
| rmu | 99 | 95 | 91 | 100 | 92 | 83 | 55 | 97 | 90 | 94 | 81 | 100 | 100 | 88 | 100 | 100 | 80 | 88 |
| subject_blocklist_idk | 0 | 0 | 0 | 100 | 0 | 92 | 0 | 0 | 0 | 100 | 100 | 0 | 100 | 97 | 100 | 0 | 0 | 0 |

### Legacy view vs LETHE-Bench (same run)

Legacy view = only the families the official MCF / zsRE metrics use: erasure on the request and its paraphrases (Eff/Gen), retention on same-answer neighbours and unrelated questions (Spe). A shortcut that looks perfect here and collapses under LETHE-Bench is what the legacy metrics reward.

| Method | legacy erasure | legacy locality | LETHE-Bench E | LETHE-Bench L | LETHE |
|---|---|---|---|---|---|
| sure | 46.5 | 100.0 | 2.3 | 100.0 | 7.9 |
| ga | 100.0 | 5.8 | 98.0 | 1.8 | 6.9 |
| gd | 100.0 | 7.7 | 97.1 | 1.8 | 6.7 |
| npo | 100.0 | 0.0 | 98.4 | 1.3 | 4.9 |
| answer_ban † | 93.9 | 30.8 | 90.0 | 6.8 | 0.0 |
| idk_all † | 100.0 | 0.0 | 100.0 | 1.0 | 0.0 |
| keyword_router_idk † | 81.6 | 100.0 | 5.9 | 7.0 | 0.0 |
| noop † | 0.0 | 100.0 | 1.0 | 100.0 | 0.0 |
| prompt_cache_idk † | 50.0 | 100.0 | 1.3 | 95.4 | 0.0 |
| rmu | 5.6 | 84.1 | 4.4 | 91.2 | 0.0 |
| subject_blocklist_idk † | 100.0 | 98.4 | 6.8 | 2.0 | 0.0 |

### Attacks (residual on forget facts after the attack, relative to the base model after the same attack; lower is better)

| Method | int8 | int4 | relearn (held-out half) | logit lens (upper half, chance-corrected) |
|---|---|---|---|---|
| sure | 47.4 | 55.3 | 76.8 | 83.3 |
| ga | 0.9 | 0.0 | 19.6 | 16.7 |
| gd | 1.8 | 2.6 | 16.1 | 17.6 |
| npo | 0.0 | 2.6 | 7.1 | 13.9 |
| answer_ban | 2.6 | 2.6 | 0.0 | 100.0 |
| idk_all | 0.0 | 0.0 | 0.0 | 100.0 |
| keyword_router_idk | 19.3 | 18.4 | 32.1 | 100.0 |
| noop | 100.0 | 100.0 | 100.0 | 100.0 |
| prompt_cache_idk | 59.6 | 65.8 | 64.3 | 100.0 |
| rmu | 95.6 | 97.4 | 92.9 | 100.0 |
| subject_blocklist_idk | 0.0 | 0.0 | 0.0 | 100.0 |

### Open-ended behaviour (greedy generation; probes the base model answers in generation)

| Method | forget: leak | forget: refusal | forget: other (wrong, empty, degenerate) | retain: correct | retain: false refusal |
|---|---|---|---|---|---|
| sure | 71.3 | 0.0 | 28.7 | 100.0 | 0.0 |
| ga | 1.8 | 0.0 | 98.2 | 21.8 | 0.0 |
| gd | 2.7 | 0.0 | 97.3 | 21.3 | 0.0 |
| npo | 1.5 | 0.0 | 98.5 | 8.0 | 0.0 |
| answer_ban | 8.4 | 0.0 | 91.6 | 49.4 | 0.0 |
| idk_all | 0.0 | 100.0 | 0.0 | 0.0 | 100.0 |
| keyword_router_idk | 36.7 | 63.3 | 0.0 | 63.2 | 36.8 |
| noop | 100.0 | 0.0 | 0.0 | 100.0 | 0.0 |
| prompt_cache_idk | 64.2 | 35.8 | 0.0 | 93.7 | 6.3 |
| rmu | 91.9 | 0.0 | 8.1 | 92.5 | 0.0 |
| subject_blocklist_idk | 9.3 | 90.7 | 0.0 | 45.4 | 54.6 |

### Detect-then-edit decomposition

| Method | route recall (forget) | residual from routing misses | residual when routed | false-fire rate (retain) |
|---|---|---|---|---|
| sure | 51.3 | 48.7 | 9.1 | 0.0 |
| keyword_router_idk | 63.9 | 36.1 | 0.0 | 35.8 |

### Cost

| Method | unlearn seconds | inference overhead (× base) |
|---|---|---|
| sure | 0.0 | 1.26 |
| ga | 10.8 | 1.25 |
| gd | 94.5 | 1.64 |
| npo | 229.6 | 1.20 |
| answer_ban | 0.0 | 1.36 |
| idk_all | 0.0 | 1.40 |
| keyword_router_idk | 0.0 | 1.29 |
| noop | 0.0 | 1.06 |
| prompt_cache_idk | 0.0 | 1.52 |
| rmu | 92.3 | 0.98 |
| subject_blocklist_idk | 0.0 | 1.40 |

