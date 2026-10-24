# Red-teaming SCALPEL before anyone else does

Snorkel's red-team service examines benchmarks for six failure classes. For each one, this file lists what could go wrong in an unlearning benchmark, what SCALPEL does about it, and where it is checked automatically. `python -m scalpel redteam --split S --model M --out DIR` runs the automatic part and writes `RED_TEAM_REPORT.md` (pass/fail).

## 1. Reward hacking and grader exploits

| Exploit | Defence | Check |
|---|---|---|
| Refuse everything (`idk_all`) | Retain families → L ≈ 1 | Reference cheat; gate < 35 |
| Refuse when a request prompt appears (`prompt_cache_idk`) | `paraphrase`, `question`, `crosslingual` → E ≈ 1 | Reference cheat |
| Refuse when the subject appears (`subject_blocklist_idk`) | `same_subject`, `same_sentence`, `subject_mentioned` → L ≈ 1; `surface`, `indirect` → E low | Reference cheat |
| Subject **and** relation-word router + refusal (`keyword_router_idk`): the string version of a detect-then-edit method | `indirect` (subject never named), `multihop`, `crosslingual`, `surface` → E low | Reference cheat |
| Ban the forgotten answers' first token everywhere (`answer_ban`) | `same_object` (same answer, other subject) and `in_context` → L ≈ 1 | Reference cheat |
| Averaging hides a collapsed family | Harmonic aggregation over families (E, L) and worst-case attack (R). An arithmetic-mean version let `answer_ban` score 79.6 on the first toy run; the harmonic form gives < 5 | `tests/test_core.py::test_aggregates_harmonic_punishes_a_collapsed_family` |
| **Non-causal wrapper** that lowers the answer when it sees it later in the teacher-forced input | Causality check on every run; > 0.05 logit change marks the run **INVALID** | `tests/test_end_to_end.py::test_causality_check_rejects_peeking`. It also caught a real bug in our own `answer_ban` (a batch-wide `min()`) |
| Obfuscated leaks ("F-r-e-n-c-h", accents, casing) | Normalized whole-word matching plus a letter-spaced variant; teacher-forced scoring cannot be gamed by formatting | `test_classify` |
| Hallucinating a wrong answer instead of forgetting | Reported separately (forget: leak / refusal / other) | Behaviour table |
| Guardrail passed off as unlearning | Tiers B / W / W+: runtime components must be declared, and W+ scores the bare weights | `strict_tier` in every result |

The gate (`redteam.cheat_gate`) requires **max(SCALPEL, SCALPEL_core) < 35** for every reference cheat. The core score is checked too, so the attack axis alone cannot save the benchmark. `validate_split` also requires that, for each cheat, a family that defeats it has at least 5 probes.

## 2. Verifier reliability

- The primary scorer is **teacher-forced exact top-1 on every answer token**: deterministic, with no judge. This is the MQuAKE / MultiFactPerson convention.
- The generation scorer is rule-based and deterministic. An LLM judge is optional and never used for headline numbers. Planned: a human-labelled calibration set of 1,000 generations (leak / refusal / hallucination) to measure the rule grader's agreement. This is an in-kind ask (Snorkel expert data).
- Both scorers are reported side by side, so disagreements between them are visible.

## 3. Task specification and solvability

- Every probe is answered correctly by the base model (the knowledge filter). The ideal score (E = 100 and L = 100) is therefore achievable *in principle*: the genie upper bound in this repository shows it on the legacy families.
- Label conflicts are removed by construction (§4.3 of the spec), and `validate_split` re-checks them on every split.
- `in_context` uses a **counterfactual** note, so "answer from the note" is unambiguous whatever a deployer's policy on reading forgotten facts.
- Known ambiguity: `indirect` descriptions can be inferable from correlated retained facts (born in France → speaks French). The mitigation is that description facts come from different answer groups than the asked relation. Planned: expert-written descriptions (ROADMAP M3).

## 4. Contamination and leakage

- A canary GUID is in every file.
- A hidden test partition (45% of subjects) is used with a private seed and optional **private probe templates** (`--private-templates`), and its probes are never released.
- Splits are model-conditioned and re-sampled per seed. Overfitting to a public instance does not transfer to the test split.
- The method's view is exactly `Split.method_view()`. A validator checks that no retain or held-out text is in it.

## 5. Environment robustness

- Pure PyTorch + 🤗 transformers. Tested on transformers 4.46 locally (CPU and Apple MPS); the repository's cluster environment is newer.
- Scoring shortcuts are tested for exact parity with the slow path: unembedding at the scored positions only (vs the full forward), the bucketed greedy decoder (vs HF `generate`), the cheats' logits-processor generation (vs their own forward), answer-position losses (vs full logits), and the backbone logit lens (vs the full model).
- Results record the split fingerprint, eval config, versions, device, method card and method arguments.
- The base reference is cached by split fingerprint and eval config, so every method is scored against the same base outputs.
- The attacks restore weights exactly (tested). The suite reloads the base model when a method edits weights.

## 6. Difficulty and saturation

- The legacy benchmarks are saturated (`EVIDENCE.md`). SCALPEL adds families that the repository's strongest method is expected to fail (indirect, multihop, crosslingual, surface), and an attack axis no legacy benchmark measures.
- Scores are per family, so saturation of any one family is visible. A family above 98% for every method on the leaderboard is flagged for v-next.
- Difficulty can scale with forget-set size (50 → 1k → 10k), sequential batches, and harder families (expert-written indirect references, more languages).

## Results of the automatic red team

- Toy world: `results/toy/runs/RED_TEAM_REPORT.md`.
- Real-data pilot: `results/pilot/*/redteam/RED_TEAM_REPORT.md`.
