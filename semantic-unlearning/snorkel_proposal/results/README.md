# Results

| Folder | What | Status |
|---|---|---|
| `toy/` | Synthetic world (80 people, 6 countries) and a 2M-parameter Llama trained to know it. **A pipeline check, not a scientific result** | Complete |
| `pilot/qwen2.5-0.5b-instruct/` | Real MultiCounterFact + zsRE + MQuAKE facts, Qwen2.5-0.5B-Instruct, laptop (Apple MPS), seed 1. 11 methods: 6 cheats, GA, GD, NPO, RMU, SURE | Complete; see `pilot/README.md` |
| `cluster/` (created by `scripts/submit_scalpel_compare.sh`, see `scripts/README_CLUSTER.md`) | Llama-3.2-3B / Qwen3-4B / Llama-3.1-8B × 5 seeds: 6 cheats, GA, GD, NPO, RMU, ROME, MEMIT, ZeroUnlearn, SURE | To run (M2) |

Regenerate everything with:

```bash
python -m scalpel toy --out results/toy        # ~10 min, CPU
bash scripts/run_pilot_local.sh                # ~1 h on an M2 laptop
```

## Toy world (`toy/`)

- **Split:** 24 requests and 523 probes, with 17 families of at least 5 probes each (the knowledge filter keeps only what the toy model knows).
- **Red team: PASS.** All six reference cheats are defeated (`toy/runs/RED_TEAM_REPORT.md`):

| Cheat | SCALPEL | SCALPEL_core | Why it fails |
|---|---|---|---|
| noop | 0.0 | 2.9 | E = 1: nothing is forgotten |
| idk_all | 0.0 | 2.7 | L = 1: everything is refused |
| prompt_cache_idk | 0.0 | 3.7 | E = 1.3: every rewording leaks |
| subject_blocklist_idk | 0.0 | 4.2 | L = 1.7: same-subject / same-sentence / subject-mentioned refused; typos and descriptions leak |
| keyword_router_idk | 0.0 | 8.7 | E = 3.1: subject-free descriptions, typos and cross-lingual questions leak |
| answer_ban | 0.0 | 10.2 | L = 6.3: same-answer neighbours and context reading break |

- **Baselines** (`toy/leaderboard/leaderboard.md`): GA, GD and NPO erase almost everything (E 97–100) and destroy locality (L 1.6–12.4). This is the classic over-forgetting pattern, here on a model small enough that every fact shares its parameters with the others.
- **Sequential track** (GD, 4 batches): E stays at 100 while L falls 25 → 14 → 1.9 → 2.0, so locality erodes as edits accumulate (`toy/sequential_gd/sequential.json`).
- **Invariants** (all tested in `tests/`):
  - every probe is known by the base model;
  - noop has zero erasure and full locality;
  - the causality check rejects a peeking wrapper;
  - attacks restore weights exactly;
  - the fast scoring path equals the full forward;
  - the benchmark's greedy decoder equals HF `generate`.
