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
| **M1** (month 1) | v1.0 freeze: dev splits for the 8-model grid below (5 families, 3B–27B) × 5 seeds; hidden test built and held; zsRE redistribution terms confirmed; HF dataset card | Compute |
| **M2** (months 1–2) | Full baseline tables on the cluster (`scripts/submit_scalpel_compare.sh full all`): GA, GD, NPO, RMU, ROME, MEMIT, ZeroUnlearn, SURE, and the 6 cheats; all tiers and attacks; scaling curve 50 → 1k → 10k requests; sequential track; multi-GPU support for the 14–27B models | GPU compute (≈5,000 A100-80GB hours, below) |
| **M3** (months 2–4) | **Expert data**: native-speaker verification of the cross-lingual templates and expansion to 10 languages; expert-written indirect references (descriptions, aliases, coreference chains) for 2,000 subjects; human-labelled grader calibration set (1,000 generations: leak / refusal / hallucination / correct) | Snorkel expert network |
| **M4** (months 3–5) | **Agentic track** (prototype exists: `AGENT_TRACK.md`): the unlearned model runs as an agent with a scratchpad and a mock search tool over a corpus with the target fact removed. Does multi-step reasoning (self-ask, decomposition, tool calls) reconstruct the forgotten fact? Does the agent leak it through tool queries? Long-horizon multi-turn sessions (≥ 10 turns) | Snorkel research collaboration; environment design |
| **M5** (month 6) | Public leaderboard (dev) + hidden-test evaluation service; paper (NeurIPS Datasets & Benchmarks track) | Hosting |

## In-kind asks

1. **Expert data** (M3):
   - linguists or native speakers, for cross-lingual probe verification and expansion;
   - domain annotators, to write unambiguous indirect references that are *not* inferable from retained facts;
   - annotators, to label generations for grader calibration;
   - privacy and legal experts, to design realistic deletion-request scenarios (GDPR Art. 17 style: a person asks for specific facts to be removed; what must remain).
2. **GPU compute: ~5,000 A100-80GB hours** (about 2,500–3,000 H100-hours), on 80 GB GPUs, with multi-GPU nodes for the largest models. Breakdown below.

   > **We request GPU compute as partner compute credits, which the program lists among its in-kind support.** The full grid needs 80 GB GPUs and, for the 24–27B gradient baselines, multi-GPU nodes. The 3B run so far used NJIT's Wulver cluster, which does not cover the full grid at this scale.

   **Model grid** (5 families, 8 models; relative cost = GPU-hours relative to Llama-3.2-3B, roughly proportional to parameters, +20% multi-GPU overhead at 24–27B):

   | Family | Models | Relative cost |
   |---|---|---|
   | Llama | Llama-3.2-3B-Instruct, Llama-3.1-8B-Instruct | 1, 2.7 |
   | Qwen | Qwen3-4B, Qwen3-14B | 1.4, 4.5 |
   | Gemma | Gemma-3-4B, Gemma-3-27B | 1.4, ~10 |
   | Mistral | Mistral-Small-24B | ~10 |
   | OLMo (open training data) | OLMo-2-7B | 2.7 |
   | **Total** | 8 models | **≈ 34** |

   **Measured basis.** One (model, seed) on Llama-3.2-3B, 50 requests, every method (9 methods + 6 cheats), all tiers and attacks, took **≈ 1.8 GPU-hours** of wall time on Wulver (October 2026, from `sacct`): baselines and cheats ≈ 45 min in total, the original SURE adapter ≈ 14 min, SURE (`ours`) ≈ 35–42 min, sequential track ≈ 8 min, split build ≈ 3 min. Development on the 3B model alone (paper-pipeline reruns, LETHE-Bench reruns, router diagnostics) used ≈ 75 GPU-hours in three days.

   | Item | Assumption | A100-80GB hours |
   |---|---|---|
   | Core dev grid | 10 seeds for models ≤ 8B, 5 seeds for 14–27B | ~390 |
   | Hidden test | 5 seeds, all 8 models | ~300 |
   | Scaling curve, 1k requests | 3B and 8B × 3 seeds, one 24B seed; cost ≈ linear in requests | ~760 |
   | Scaling curve, 10k requests | 3B and 8B, one seed each | ~1,330 |
   | Agentic track | up to 10 replies per probe; 2 small models and one 24B, 3 seeds | ~550 |
   | ROME / MEMIT covariance statistics | 14–27B models | ~60 |
   | Sequential track | all models | ~50 |
   | Development, debugging, reruns | at the measured 3B pace, plus the large-model infrastructure | ~1,500 |
   | **Total** | | **≈ 4,900 → ask ~5,000** (plausible range 4,000–6,000) |

   The least certain items are the 10k-request point and linear scaling in requests: SURE-style methods build rewrites and train rows per fact. One 1k-request seed on 3B will be measured before M2 to replace that assumption.

   **Large-model infrastructure** (the harness has run up to 8B on one GPU):
   - inference-only methods, scoring and attacks (cheats, SURE, logit lens, quantization) fit a 24–27B model on one 80 GB GPU in bf16;
   - GA / GD / NPO / RMU and the relearning attack need multi-GPU sharding (FSDP / DeepSpeed) or a declared parameter-efficient scope at 24–27B; the scope is reported per model so baselines stay comparable across sizes;
   - the vendored ROME / MEMIT / ZeroUnlearn code is single-GPU and is tested at 24B before M2.
   - per-model settings (SURE router and early-write layers, RMU layer, ROME / MEMIT hparams, gradient scope) exist today for `llama32_3b`, `qwen3_4b` and `llama31_8b` (`scripts/slurm/scalpel_common.sh`); the other five models are added the same way.
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
| The benchmark favours the authors' own method | SURE is one of 14 methods on the leaderboard (8 methods, 6 reference cheats). The reference cheats include a string version of its design (`keyword_router_idk`), and its expected failure families (indirect, multihop, crosslingual, surface; the strict tier) are reported like any other method's |
