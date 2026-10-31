# Roadmap, deliverables and asks

## Status today (v0.1, this folder)

| Component | Status |
|---|---|
| Split builder: 4 sources, knowledge filter, partitions, hidden test, conflict removal, canary | Done, tested |
| 11 forget families + 9 retain families, fact track + RWKU entity track | Done, tested |
| Teacher-forced and open-ended scorers, behaviour labels | Done, tested |
| Utility (MMLU, TriviaQA, PPL), KL on retain probes | Done |
| Attacks: logit lens, int8/int4, relearning | Done, tested (restore exactly) |
| Tiers B / W / W+, causality check, boundary contract | Done, tested |
| Sequential deletion track | Done, tested |
| Detect-then-edit decomposition (routing vs actuation) | Done |
| Baselines in-harness: 6 reference cheats, GA, GD, NPO, SimNPO, trained refusal (IDK), task-vector negation, RMU, ROME, MEMIT, ZeroUnlearn | Done (SimNPO / IDK / TV added 2026-10-28; cluster results pending) |
| External: HF checkpoint loader, plugin loader, SURE adapter | Done. SURE adapter runs end to end on the laptop pilot (Qwen2.5-0.5B, transformers 4.57); cluster run is M2 |
| ROME / MEMIT / ZeroUnlearn in-harness adapters (around the vendored `ZeroUnlearn/` code via `paper_baselines.py`) | Done; request format tested. The vendored code is CUDA-only, so first full run is the Wulver smoke job |
| Red-team gate + split validator | Done, passing on toy and pilot |
| Reports: leaderboard, Pareto plot, CIs, paired tests | Done |
| Toy world (CPU, minutes) and real-data pilot (laptop) | Done |

## Milestones

| | Deliverable | Needs |
|---|---|---|
| **M1** (month 1) | v1.0 freeze: dev splits for the 10-model grid below (5 families, 3B–70B) × 5 seeds; hidden test built and held; zsRE redistribution terms confirmed; HF dataset card | Compute |
| **M2** (months 1–2) | Full baseline tables on the cluster (`scripts/submit_scalpel_compare.sh full all`). Done for Llama-3.2-3B, Qwen3-4B and Llama-3.1-8B: GA, GD, NPO, SimNPO, RMU, ROME, MEMIT, ZeroUnlearn, trained refusal, task-vector negation, SURE, and the 6 cheats, all tiers and attacks, sequential track, 5 seeds each (`EVIDENCE.md` §4). Remaining: the other seven models of the grid; eight more baselines (SCRUB, WHP, ELM, GRACE, WISE, AlphaEdit, ECO, in-context unlearning prompting: the three method classes the current table lacks, lifelong editors, representation-level erasure and prompt-level guardrails), for 19 methods + 6 cheats; scaling curve 50 → 1k → 10k requests; multi-GPU support for the 14–70B models | GPU compute (≈7,000 A100-80GB hours, below) |
| **M3** (months 2–4) | **Expert data**: native-speaker verification of the cross-lingual templates and expansion to 10 languages; expert-written indirect references (descriptions, aliases, coreference chains) for 2,000 subjects; human-labelled grader calibration set (1,000 generations: leak / refusal / hallucination / correct) | Snorkel expert network |
| **M4** (months 3–5) | **Agentic track** (prototype exists: `AGENT_TRACK.md`): the unlearned model runs as an agent with a scratchpad and a mock search tool over a corpus with the target fact removed. Does multi-step reasoning (self-ask, decomposition, tool calls) reconstruct the forgotten fact? Does the agent leak it through tool queries? Long-horizon multi-turn sessions (≥ 10 turns) | Snorkel research collaboration; environment design |
| **M5** (month 6) | Public leaderboard (dev) + hidden-test evaluation service; paper (NeurIPS Datasets & Benchmarks track) | Hosting |

## In-kind asks

1. **Expert data** (M3):
   - linguists or native speakers, for cross-lingual probe verification and expansion;
   - domain annotators, to write unambiguous indirect references that are *not* inferable from retained facts;
   - annotators, to label generations for grader calibration;
   - privacy and legal experts, to design realistic deletion-request scenarios (GDPR Art. 17 style: a person asks for specific facts to be removed; what must remain).
2. **GPU compute: ~7,000 A100-80GB hours** (about 3,500–4,200 H100-hours), on 80 GB GPUs, with multi-GPU nodes for the largest models. Breakdown below.

   > **We request GPU compute as partner compute credits, which the program lists among its in-kind support.** The full grid needs 80 GB GPUs and, for the 24–70B gradient baselines and the relearning attack, multi-GPU nodes. The three-model run so far (3B, 4B, 8B; `EVIDENCE.md` §4) used NJIT's Wulver cluster, which does not cover the grid above 8B at this scale.

   **Model grid** (5 families, 10 models, 3B–70B; three done. Relative cost = GPU-hours relative to Llama-3.2-3B, roughly proportional to parameters, +20% multi-GPU overhead from 24B):

   | Family | Models | Relative cost | Status |
   |---|---|---|---|
   | Llama | Llama-3.2-3B-Instruct, Llama-3.1-8B-Instruct, Llama-3.3-70B-Instruct | 1, 2.7, ~30 | 3B and 8B done |
   | Qwen | Qwen3-4B, Qwen3-14B, Qwen3-32B | 1.4, 4.5, ~12 | 4B done |
   | Gemma | Gemma-3-4B, Gemma-3-27B | 1.4, ~10 | |
   | Mistral | Mistral-Small-24B | ~10 | |
   | OLMo (open training data) | OLMo-2-7B | 2.7 | |
   | **Total** | 10 models | **≈ 76** | |

   The 32B and 70B tiers are what the credits buy that the university cluster cannot: a 70B model in bf16 needs two 80 GB GPUs for inference alone and four to eight for the gradient baselines and the relearning attack.

   **Measured basis.** One (model, seed) on Llama-3.2-3B, 50 requests, every method (9 methods + 6 cheats at the time; 11 methods now, ≈ 2.2 GPU-hours), all tiers and attacks, took **≈ 1.8 GPU-hours** of wall time on Wulver (October 2026, from `sacct`): baselines and cheats ≈ 45 min in total, the original SURE adapter ≈ 14 min, SURE (`ours`) ≈ 35–42 min, sequential track ≈ 8 min, split build ≈ 3 min. Development on the 3B model alone (paper-pipeline reruns, LETHE-Bench reruns, router diagnostics) used ≈ 75 GPU-hours in three days. Measured since on the other two models (`leaderboard.md`, cost section): SURE alone takes ≈ 46 min per seed on Qwen3-4B and ≈ 76 min on Llama-3.1-8B against ≈ 36 min on Llama-3.2-3B, i.e. ×1.3 and ×2.1, inside the parameter-proportional estimate used here (×1.3 and ×2.7).

   | Item | Assumption | A100-80GB hours |
   |---|---|---|
   | Core dev grid | 19 methods + 6 cheats at ≈ 2.2 GPU-h per 3B seed; 10 seeds for models ≤ 8B, 5 seeds for 14–32B, 3 seeds for 70B | ~800 |
   | Hidden test | 5 seeds, all 10 models | ~830 |
   | Scaling curve, 1k requests | 3B and 8B × 3 seeds, one 24B seed; cost ≈ linear in requests | ~930 |
   | Scaling curve, 10k requests | 3B and 8B, one seed each | ~1,630 |
   | Agentic track | up to 10 replies per probe; 2 small models and one 24B, 3 seeds | ~550 |
   | ROME / MEMIT / ZeroUnlearn covariance statistics | 14–70B models | ~100 |
   | Sequential track | all models | ~80 |
   | Representation-level attacks (tuned lens, answer probe; `RED_TEAM.md` §6) added to R | all models, dev + hidden | ~100 |
   | Development, debugging, reruns | at the measured 3B pace, plus the 32B / 70B infrastructure | ~1,500 |
   | **Total** | | **≈ 6,500 → ask ~7,000** (plausible range 5,500–8,500) |

   The least certain items are the 10k-request point and linear scaling in requests: SURE-style methods build rewrites and train rows per fact. One 1k-request seed on 3B will be measured before M2 to replace that assumption.

   **Large-model infrastructure** (the harness has run up to 8B on one GPU):
   - inference-only methods, scoring and attacks (cheats, SURE, logit lens, quantization) fit a 24–27B model on one 80 GB GPU in bf16;
   - GA / GD / NPO / SimNPO / RMU and the relearning attack need multi-GPU sharding (FSDP / DeepSpeed) or a declared parameter-efficient scope from 24B; the scope is reported per model so baselines stay comparable across sizes (at 8B the gradient baselines already train `down_proj` only, recorded in the result);
   - the vendored ROME / MEMIT / ZeroUnlearn code is single-GPU and is tested at 24B before M2; at 70B the covariance statistics are computed sharded;
   - per-model settings (SURE router and write layers, RMU layer, ROME / MEMIT hparams, gradient scope) exist today for `llama32_3b`, `qwen3_4b` and `llama31_8b` (`scripts/slurm/scalpel_common.sh`); SURE's two layers are chosen per model by a measured sweep (`../README_OTHER_MODELS.md`), the other seven models are added the same way.
3. **Red-team review**: Snorkel's benchmark red team examines grader exploits, verifier reliability and contamination before v1.0.
4. **Hosting**: the hidden-test evaluation service and the leaderboard.

## Risks

| Risk | Mitigation |
|---|---|
| Knowledge-filter yield is low for small models in QA/chat formats | Report yields per family. Families below 5 probes are excluded and flagged. Release models start at 3B (Llama-3.2-3B base Eff on MCF is 13.5%, and LETHE-Bench screens until it has enough known facts) |
| `indirect` probes inferable from correlated retained facts | Description facts come from other answer groups; expert-written references (M3) |
| A future method saturates LETHE-Bench | Per-family scores flag saturation. Harder families (agentic, more languages, 10k-request scale) are planned, and the hidden test is refreshed |
| Grader disagreement on open-ended outputs | The teacher-forced scorer is primary; the human-labelled calibration set (M3) measures the generation grader |
| Gradient baselines do not fit one GPU at 24–27B | Multi-GPU sharding, or a declared parameter-efficient scope reported per model; inference-only methods and attacks fit one 80 GB GPU |
| The benchmark favours the authors' own method | SURE is one of 17 rows on the leaderboard (11 methods, 6 reference cheats) on three models, and it is not the top method on any of them (`EVIDENCE.md` §4). The reference cheats include a string version of its design (`keyword_router_idk`), and its expected failure families (indirect, multihop, crosslingual, surface; the strict tier) are reported like any other method's |
