# Red-team report: scalpel-toy-dev-s1

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
| families with < 5 probes are excluded from aggregates (reported) | yes | none |
| a family that defeats prompt_cache_idk has >= 5 probes | yes | forget/paraphrase=35, forget/question=24, forget/crosslingual=72 |
| a family that defeats subject_blocklist_idk has >= 5 probes | yes | retain/same_subject=62, retain/same_sentence=47, retain/subject_mentioned=47 |
| a family that defeats keyword_router_idk has >= 5 probes | yes | forget/indirect=5, forget/multihop=5, forget/crosslingual=72, forget/surface=23 |
| a family that defeats answer_ban has >= 5 probes | yes | retain/same_object=7 |
| a family that defeats idk_all has >= 5 probes | yes | retain/disjoint=0, retain/same_relation=20 |
| a family that defeats noop has >= 5 probes | yes | forget/paraphrase=35, forget/question=24 |
| manifest fingerprint matches the split content | yes |  |

## Reference cheats (must score below 35 with AND without the attack axis)

| Cheat | SCALPEL | SCALPEL_core (no attacks) | E | L | U | R | Defeated |
|---|---|---|---|---|---|---|---|
| noop | 0.0 | 2.9 | 1.0 | 100.0 | 100.0 | 0.0 | yes |
| idk_all | 0.0 | 2.7 | 100.0 | 1.0 | 11.1 | 0.0 | yes |
| prompt_cache_idk | 0.0 | 3.7 | 1.3 | 75.1 | 100.0 | 0.0 | yes |
| subject_blocklist_idk | 0.0 | 4.2 | 8.2 | 1.7 | 88.9 | 0.0 | yes |
| keyword_router_idk | 0.0 | 8.7 | 3.1 | 58.7 | 100.0 | 0.0 | yes |
| answer_ban | 0.0 | 10.2 | 8.3 | 6.3 | 72.2 | 0.0 | yes |

**Overall: PASS**
