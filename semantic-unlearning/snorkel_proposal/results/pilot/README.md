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

| Cheat | **Legacy view**: erasure / locality | **SCALPEL** E / L | SCALPEL_core | Where it breaks |
|---|---|---|---|---|
| noop | 0.0 / 100.0 | 1.0 / 100.0 | 2.9 | Nothing is forgotten |
| idk_all | 100.0 / 0.0 | 100.0 / 1.0 | 2.8 | Everything is refused |
| prompt_cache_idk | 50.0 / 100.0 | 1.3 / 95.4 | 3.7 | Paraphrase, question, cross-lingual, multi-turn: 97–100% residual |
| **subject_blocklist_idk** | **100.0 / 98.5** | **6.8 / 2.0** | 4.5 | Typos 100% and multi-hop 92% residual. Same-subject, same-sentence, subject-mentioned and in-context are all refused (0%) |
| keyword_router_idk | 82.5 / 100.0 | 5.9 / 7.0 | 9.3 | Subject-free 100%, typos 100%, cross-lingual 94% and multi-hop 92% residual. Same-sentence 0% |
| answer_ban | 97.8 / 30.4 | 96.9 / 6.5 | 16.2 | Same-answer neighbours 0% retained, name collisions 10% |

The *legacy view* scores the same run on only the families that the official MultiCounterFact / zsRE metrics use: the request and its paraphrases for forgetting; same-answer neighbours and unrelated questions for locality. **A subject-name blocklist looks nearly perfect there (100 / 98.5) and scores 0 on SCALPEL.** That gap is the case for this benchmark.

## 2. Baselines

All baselines use the defaults ported from `../scripts/paper_baselines.py`: lr 1e-5, MLP parameters, early stop at forget-probability 1e-4. The full tables are in `qwen2.5-0.5b-instruct/leaderboard/leaderboard.md`.

| Method | E | L | U | R | SCALPEL | What happened |
|---|---|---|---|---|---|---|
| GA | 99.0 | 1.8 | 53.4 | 80.4 | 6.7 | Erases every family. Same-subject, same-object, same-relation and same-sentence all fall to 0% retained. PPL 14.8 → 59.1 |
| GD | 97.4 | 1.8 | 58.4 | 82.4 | 6.7 | Same pattern; the retain loss does not protect *entangled* facts |
| NPO | 99.2 | 1.3 | 63.6 | 86.1 | 4.9 | Same pattern; KL on retain probes 5.8 nats |
| RMU | 4.6 | 84.5 | 94.7 | 0.0 | 0.0 | At default settings it barely erases on this model (direct residual 92%). Per-method sweeps are M2 |

- **Over-forgetting** shows up the same way as in this repository's MultiCounterFact study (GA: 99.8% of Eff removed / 1.8% of Spe kept). SCALPEL now attributes it per family. Even `in_context` reading drops to 22–60%: the methods damage the model's ability to *read* a note about the forgotten subject, not only its recall.
- **Robustness is not free.** After relearning on the other half of the requests, GA recovers 19.6% of the held-out facts (relative to the base model under the same attack). The logit lens still ranks 16.7% of answers first in the upper layers, a gap that the output-only metrics hide.
- **Open-ended behaviour.** The gradient methods do not abstain. 97–98% of their forget generations are wrong or degenerate answers (refusals: 0%). A deployer who needs "I don't know" would not get it.
- **The generation scorer catches what teacher forcing misses.** `answer_ban` bans the answer's first token, yet 8.4% of its generations still produce the answer through another tokenization.

## 3. SURE (this repository's method) on the same split

SURE ran through `--method sure`: the repository's MultiFactPerson entry points (prep → linear router → Δe rows), trained on **the split's 50 requests only**, at layer 16 (the 3B choice of 19/28 scaled to 24 layers), with no tuning for this model. The run uses transformers 4.57, because the repository scripts need ≥ 4.56. Results: `qwen2.5-0.5b-instruct/runs/sure/results.json`. Command:

```bash
python -m scalpel run --split results/pilot/qwen2.5-0.5b-instruct/split_dev_s1.json --model $QWEN_DIR \
  --method sure --method-arg model_path=$QWEN_DIR --method-arg layer=16 --method-arg device=mps \
  --out results/pilot/qwen2.5-0.5b-instruct/runs/sure --device mps --dtype fp32 --relearn-scope mlp_down
```

| | Value |
|---|---|
| SCALPEL | **10.5**, the best in the pilot (GA/GD 6.7). E 3.3 · L **100.0** · U 98.2 · R 16.7 |
| Locality | **100% on all 8 retain families**, 0% false fires. Same-subject, same-sentence, same-object and subject-mentioned are untouched |
| Forget residual by family (tf) | direct 18 · context prefix 19 · paraphrase 45 · multi-fact 58 · multi-turn 86 · question 90 · cross-lingual 94 · typo 100 · multi-hop 100 |
| Routing decomposition | Route recall 51%. **48.7 points of forget residual come from routing misses, 9.1 from actuation** (the rows fail even when routed) |
| Where routing fails | Typos 0/18 routed (exact subject-token eligibility); multi-hop 0/12 (subject not named); cross-lingual 7/67; multi-turn 3/22; QA-format questions 6/30 |
| Strict tier (weights only) | E 1.0. SURE changes no weights (288k extra parameters in hooks/rows) |
| Attacks | The logit lens still reads 83% (relative) of attacked facts in the upper half. Relearning brings back 77% of the held-out half |

On Llama-3.2-3B with tuned settings, SURE removes 96–100% of base recall on the four legacy benchmarks (`../../EVIDENCE.md`). This untuned 0.5B run is weaker even on the legacy-view families (68.5 erasure / 100 locality: direct 18% and paraphrase 45% residual). The pattern is still clear: the families that legacy benchmarks lack leak far more (question 90%, cross-lingual 94%, typos 100%, multi-hop 100%). The cause is detection, not editing. That is the headroom SCALPEL is built to measure, and the decomposition says where to work.

## 4. Caveats

- SURE ran with its 3B defaults (layer, router calibration, 30 updates per fact), untuned for this model; its QA-format requests were trained as raw text while the evaluator renders questions with the chat template. Both are fixable in the method; the benchmark reports them as they are.
- One model (0.5B), one seed, laptop memory limits:
  - the relearning attacker fine-tunes `down_proj` only;
  - RMU runs at its 3B defaults, untuned.
- The `indirect` family has 2 probes on this model and is excluded. Larger models know more descriptions (milestone M2).
- Wall-clock times are for Apple MPS and are not comparable with GPU numbers.
