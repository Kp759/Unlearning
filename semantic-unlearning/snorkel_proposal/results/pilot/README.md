# Real-data pilot: Qwen2.5-0.5B-Instruct on a laptop

**Setup**

| | |
|---|---|
| Machine | Apple M2, 16 GB, MPS, fp32 |
| Code | This folder, `bash scripts/run_pilot_local.sh Qwen/Qwen2.5-0.5B-Instruct` |
| Split | `qwen2.5-0.5b-instruct/split_dev_s1.json`: dev partition, seed 1 |
| Requests / probes | 50 requests, 520 probes |
| Sources | MultiCounterFact + zsRE + MQuAKE (38,086 facts after merging) |

Per-family yields are in `split_summary.md`. A 0.5B model knows 9.8% of the screened facts, and only 2 subject-free (`indirect`) descriptions, so that family is reported but excluded from the aggregates. Release models are ≥ 3B.

This is a **pilot**: one small model, one seed, laptop hyper-parameters (`--relearn-scope mlp_down` for memory). It shows that the benchmark runs on real data and behaves as designed. It is not the paper's result table (that is milestone M2 on the cluster).

## 1. The red-team gate passes on real data

`qwen2.5-0.5b-instruct/runs/RED_TEAM_REPORT.md`: every integrity check passes, and every reference cheat is defeated.

| Cheat | **Legacy view**: erasure / locality | **LETHE-Bench** E / L | LETHE_core | Where it breaks |
|---|---|---|---|---|
| noop | 0.0 / 100.0 | 1.0 / 100.0 | 2.9 | Nothing is forgotten |
| idk_all | 100.0 / 0.0 | 100.0 / 1.0 | 2.8 | Everything is refused |
| prompt_cache_idk | 50.0 / 100.0 | 1.3 / 95.4 | 3.7 | Paraphrase, question, cross-lingual, multi-turn: 97–100% residual |
| **subject_blocklist_idk** | **100.0 / 98.5** | **6.8 / 2.0** | 4.5 | Typos 100% and multi-hop 92% residual. Same-subject, same-sentence, subject-mentioned and in-context are all refused (0%) |
| keyword_router_idk | 82.5 / 100.0 | 5.9 / 7.0 | 9.3 | Subject-free 100%, typos 100%, cross-lingual 94% and multi-hop 92% residual. Same-sentence 0% |
| answer_ban | 97.8 / 30.4 | 96.9 / 6.5 | 16.2 | Same-answer neighbours 0% retained, name collisions 10% |

The *legacy view* scores the same run on only the families that the official MultiCounterFact / zsRE metrics use: the request and its paraphrases for forgetting; same-answer neighbours and unrelated questions for locality. **A subject-name blocklist looks nearly perfect there (100 / 98.5) and scores 0 on LETHE-Bench.** That gap is the case for this benchmark.

## 2. Baselines

All baselines use the defaults ported from `../scripts/paper_baselines.py`: lr 1e-5, MLP parameters, early stop at forget-probability 1e-4. The full tables are in `qwen2.5-0.5b-instruct/leaderboard/leaderboard.md`.

| Method | E | L | U | R | LETHE | What happened |
|---|---|---|---|---|---|---|
| GA | 99.0 | 1.8 | 53.4 | 80.4 | 6.7 | Erases every family. Same-subject, same-object, same-relation and same-sentence all fall to 0% retained. PPL 14.8 → 59.1 |
| GD | 97.4 | 1.8 | 58.4 | 82.4 | 6.7 | Same pattern; the retain loss does not protect *entangled* facts |
| NPO | 99.2 | 1.3 | 63.6 | 86.1 | 4.9 | Same pattern; KL on retain probes 5.8 nats |
| RMU | 4.6 | 84.5 | 94.7 | 0.0 | 0.0 | At default settings it barely erases on this model (direct residual 92%). Per-method sweeps are M2 |

- **Over-forgetting** shows up the same way as in this repository's MultiCounterFact study (GA: 99.8% of Eff removed / 1.8% of Spe kept). LETHE-Bench now attributes it per family. Even `in_context` reading drops to 22–60%: the methods damage the model's ability to *read* a note about the forgotten subject, not only its recall.
- **Robustness is not free.** After relearning on the other half of the requests, GA recovers 19.6% of the held-out facts (relative to the base model under the same attack). The logit lens still ranks 16.7% of answers first in the upper layers, a gap that the output-only metrics hide.
- **Open-ended behaviour.** The gradient methods do not abstain. 97–98% of their forget generations are wrong or degenerate answers (refusals: 0%). A deployer who needs "I don't know" would not get it.
- **The generation scorer catches what teacher forcing misses.** `answer_ban` bans the answer's first token, yet 8.4% of its generations still produce the answer through another tokenization.

## 3. Caveats

- One model (0.5B), one seed, laptop memory limits:
  - the relearning attacker fine-tunes `down_proj` only;
  - RMU runs at its 3B defaults, untuned.
- The `indirect` family has 2 probes on this model and is excluded. Larger models know more descriptions (milestone M2).
- Wall-clock times are for Apple MPS and are not comparable with GPU numbers.

