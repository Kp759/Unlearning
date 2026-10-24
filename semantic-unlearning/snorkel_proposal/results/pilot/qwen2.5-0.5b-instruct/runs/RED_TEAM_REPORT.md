# Red-team report: scalpel-fact-dev-s1

## Split integrity

| Check | OK | Detail |
|---|---|---|
| held-out forget probes differ from every request prompt | yes | 0 identical |
| no retain probe text appears in the method's view | yes | 0 found |
| no prompt is both a forget and a retain probe | yes | 0 conflicts |
| no retain probe asks a forgotten (subject, relation) | yes | 0 found |
| every fact request has a known direct probe | yes | 0 missing |
| one request per subject | yes | max 1 |
| every forget probe names an existing request | yes | 0 dangling |
| probe ids are unique | yes | 0 duplicated |
| canary present | yes |  |
| families with < 5 probes are excluded from aggregates (reported) | yes | {"forget/indirect": 2} |
| a family that defeats prompt_cache_idk has >= 5 probes | yes | forget/paraphrase=40, forget/question=30, forget/crosslingual=67 |
| a family that defeats subject_blocklist_idk has >= 5 probes | yes | retain/same_subject=5, retain/same_sentence=14, retain/subject_mentioned=25 |
| a family that defeats keyword_router_idk has >= 5 probes | yes | forget/indirect=2, forget/multihop=12, forget/crosslingual=67, forget/surface=18 |
| a family that defeats answer_ban has >= 5 probes | yes | retain/same_object=34 |
| a family that defeats idk_all has >= 5 probes | yes | retain/disjoint=28, retain/same_relation=13 |
| a family that defeats noop has >= 5 probes | yes | forget/paraphrase=40, forget/question=30 |
| manifest fingerprint matches the split content | yes |  |

## Reference cheats (must score below 35 with AND without the attack axis)

| Cheat | SCALPEL | SCALPEL_core (no attacks) | E | L | U | R | Defeated |
|---|---|---|---|---|---|---|---|
| noop | 0.0 | 2.9 | 1.0 | 100.0 | 100.0 | 0.0 | yes |
| idk_all | 0.0 | 2.8 | 100.0 | 1.0 | 19.3 | 0.0 | yes |
| prompt_cache_idk | 0.0 | 3.7 | 1.3 | 95.4 | 100.0 | 0.0 | yes |
| subject_blocklist_idk | 0.0 | 4.5 | 6.8 | 2.0 | 100.0 | 0.0 | yes |
| keyword_router_idk | 0.0 | 9.3 | 5.9 | 7.0 | 100.0 | 0.0 | yes |
| answer_ban | 0.0 | 16.2 | 96.9 | 6.5 | 49.1 | 0.0 | yes |

**Overall: PASS**
