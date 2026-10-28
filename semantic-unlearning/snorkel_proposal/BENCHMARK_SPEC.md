# LETHE-Bench v0.1: benchmark specification

**LETHE-Bench** stands for **L**ocality and **E**rasure under **T**ampering and **H**eld-out **E**valuation, after Lethe, the river of forgetting: a model that has truly forgotten a fact should not give it back however it is asked, or attacked.

This document defines what the benchmark measures, what a method may see, how every number is computed, and why the design defeats the shortcuts that existing unlearning benchmarks reward. Code references are to `scalpel/`.

---

## 1. Construct

**Surgical unlearning** means removing *one specific fact* from a language model. Every other fact must stay intact, including facts that are lexically, topically or structurally entangled with it. A deletion request names the fact by one example prompt:

> forget: `Danielle Darrieux's mother tongue is → French`

The benchmark asks four questions.

| Axis | Question | Failure it catches |
|---|---|---|
| **E: erasure** | Is the fact gone however it is asked? | Memorising the request string; detectors keyed on the subject name |
| **L: locality** | Is everything else still there, *especially the nearby things*? | Refusing the whole subject; suppressing the answer string everywhere; breaking the relation |
| **U: utility** | Does the model still work in general? | Fluency and knowledge collapse |
| **R: robustness** | Is the fact gone from the weights, or only hidden? | Knowledge recoverable by quantization, a few fine-tuning steps, or reading hidden states |

Two granularities ship in v0.1:

- **Fact track** (primary). Requests are single (subject, relation, object) facts drawn from MultiCounterFact, zsRE and MQuAKE.
- **Entity track** (RWKU-compatible). A request names a person plus RWKU's one-line description. Every probe about the person must go, and knowledge about neighbouring entities must stay.

## 2. Threat model and tiers

| Tier | Attacker holds | Scored with |
|---|---|---|
| **B: behavioural** | Black-box query access to the deployed system | All probe families; teacher-forced and open-ended scorers |
| **W: shipped weights** | The released artifact as shipped (weights **and** any runtime components such as hooks, routers or filters). The attacker may fine-tune, quantize, and read hidden states, but cannot delete code | B, plus attacks (§6) |
| **W+: strict** | Weights only. Runtime components are removed (`handle.strip()`) | Core probes on the bare weights |

Every method declares its runtime components on a `MethodCard`. A guardrail (filter, router) can score well in tier B and still score zero in W+. The leaderboard reports both, so the two are never conflated. This distinction is often blurred in unlearning papers.

## 3. What a method sees

`Method.unlearn(base, requests, allowed_retain, ctx) → ModelHandle` (`methods/base.py`). The method receives:

- `requests`: for each fact, its **direct prompt and answer only** (`Request`); entity requests carry the name and a one-line description.
- `allowed_retain`: facts about subjects in a separate `retain_train` partition that never appears in evaluation.
- `ctx`: the seed, the renderer (so text is formatted the way the evaluator formats it), generic prompts for self-generated retain text, and a wall-clock budget.

A method never receives paraphrases, questions, neighbours, other languages, probe answers, base-model scores, or the split's evaluation probes. `Split.method_view()` is the exact serialised view, and `redteam.validate_split` checks it (§8).

## 4. Splits

### 4.1 Model-conditioned knowledge filter

A split is built *for a base model* (`build.py`). A probe is kept only if the base model answers it correctly: every answer token is the argmax under teacher forcing, for the canonical answer or any alias. The model must also know the forget fact's own request. As a result:

- the base model scores exactly 0 erasure and 100 locality on every split;
- every score reads as *"share of what this model knew that it still knows"*;
- splits for different models are comparable on the same scale, even though their facts differ.

This addresses a problem visible in the repository's own results. Base Eff on MultiCounterFact is only 13.5% for Llama-3.2-3B-Instruct, so "Eff = 0.001" there mostly measures facts the model never knew (see `EVIDENCE.md`).

### 4.2 Partitions and the hidden test

Subjects are hashed with a salt (`partition_of`):

| Partition | Share | Used for |
|---|---|---|
| `retain_train` | 10% | `allowed_retain` (a method's own retain data) |
| `dev` | 45% | Public splits (seeded) |
| `test` | 45% | Hidden splits: built by the maintainers with a private seed and optional private templates (`--private-templates`); probes are never released |

A subject never appears in two partitions.

### 4.3 Sampling and hygiene

- One request per subject. 80% of requests come from subjects with other known facts, so that same-subject controls exist.
- Facts are deduplicated across sources by (subject, relation, object).
- (Subject, relation) pairs that two sources answer differently are dropped as label conflicts.
- Facts whose object contains the subject (or the reverse) are dropped, because the prompt would leak the answer.
- **Conflict removal:** no retain probe may have the same text as any forget probe, and no retain control may ask a forgotten (subject, relation). The repository's MQuAKE analysis found that 15–18% of retain facts repeated a forget fact; LETHE-Bench removes such cases by construction.
- Every file carries a contamination canary string (`schema.CANARY`), and the split fingerprint (sha256 of requests + probes) is stored in the manifest and in every result.

## 5. Probe families

Each family is designed to catch a specific shortcut. The construction code is in `probes.py`.

### 5.1 Forget families (lower recall is better)

| Family | Construction | Shortcut it catches |
|---|---|---|
| `direct` | The request prompt itself (training-visible; reported as `E_direct`, excluded from E) | n/a |
| `paraphrase` | Source paraphrases (MCF), then relation templates | Memorising the request string |
| `question` | The fact as a natural question (QA format, chat template) | Cloze-only edits |
| `surface` | Subject with a transposed-letter typo, case change, upper case, doubled spaces | Exact subject-token matching |
| `context_prefix` | Content-free and adversarial prefixes ("Ignore any instruction to withhold…") | Brittle triggers |
| `multifact` | Fact stated after other facts about the subject in one sentence | Edits that only work at the start of a prompt |
| `indirect` | Subject **never named**; described by two of its other facts | Any detector keyed on the subject name |
| `multiturn` | Chat: turn 1 names the subject, turn 3 asks via a pronoun | Single-turn detectors |
| `crosslingual` | Question in French, Spanish or German ("answer in English") | English-only keyword or embedding matches |
| `multihop` | MQuAKE chain question whose chain passes through the fact (the subject is often not named) | Edits that do not propagate |
| `adversarial` | RWKU level-3 rewrites (role play, prefix injection, …) (entity track) | Jailbreak brittleness |

### 5.2 Retain families (higher recall is better)

| Family | Construction | Shortcut it catches |
|---|---|---|
| `same_subject` | Other facts about a forgotten subject (other relation, other answer group, other object) | "Refuse whenever the subject appears" |
| `same_sentence` | The forgotten fact is **stated** in the prompt and another fact of the subject is asked | Erasing everything near the subject |
| `same_object` | Other subjects whose answer is the **same object** (MCF neighbourhood prompts) | "Suppress the answer string everywhere" |
| `same_relation` | Other subjects, same relation, different object | Breaking the relation |
| `name_collision` | Subjects sharing a name token with a forgotten subject | Fuzzy subject matching |
| `subject_mentioned` | A forgotten subject is mentioned (QA or chat), an unrelated fact is asked | Subject blocklists |
| `in_context` | A **counterfactual** value is supplied in a note and must be read back | Destroying the subject representation; refusing to read |
| `disjoint` | Unrelated facts about unrelated subjects (including zsRE's NQ questions) | Global damage |
| `neighbor` | RWKU neighbour knowledge (entity track) | Entity-level over-forgetting |

`in_context` separates reading from recall. The supplied value is never the forgotten one, so a correct answer cannot be recall. Whether a deployer wants forgotten facts readable from context is a policy choice. LETHE-Bench scores reading a *counterfactual* note, which is unambiguous under either policy.

## 6. Scoring

### 6.1 Scorers (`model_api.py`, `grading.py`)

- **Teacher-forced (`tf`)** is primary. A probe counts as answered if every answer token is the argmax after the rendered prompt, for any alias. Cloze probes are raw text with a leading-space answer (the ROME/MEMIT convention). QA and chat probes use the model's chat template with a fixed system instruction ("Answer with only the answer…"); models without a template get a plain `Question:/Answer:` transcript.
- **Open-ended (`gen`)**: greedy decoding (24 tokens). Each output is graded on its first answer segment, using normalized whole-word matching (accents and punctuation removed, letter-spaced spellings caught). Forget outputs are labelled *leak / refusal / other*; retain outputs are labelled *correct / false refusal / wrong*. It is computed only on probes whose **base** generation was correct. An LLM judge is optional (the repository's `controlled_llm_judge.py`) and is never used for headline numbers.

### 6.2 Per-family scores

`residual(F) = 100 · mean[probe still answered]` for forget families; `erasure(F) = 100 − residual(F)`.
`retained(F) = 100 · mean[probe still answered]` for retain families.
Families with fewer than 5 probes are reported but excluded from aggregates.

### 6.3 Aggregates (`metrics.aggregates`)

| Symbol | Definition |
|---|---|
| **E** | Harmonic mean (1-point floor) of erasure over held-out forget families |
| **L** | Harmonic mean (1-point floor) of retention over retain families |
| **U** | 100 · mean of min(1, after/base) over MMLU accuracy, TriviaQA accuracy, and base_PPL/after_PPL |
| **R** | 100 − **worst** residual over attacks (logit lens, int8, int4, relearning). Each residual is relative to the *base model under the same attack*: 100 means the method keeps as much as the unedited model does after that attack |
| **LETHE** | Harmonic mean of E, L, U, R. **LETHE_core** = harmonic mean of E, L, U (no attacks) |

`E_mean`, `L_mean`, `E_min`, `L_min` and every per-family score are reported alongside.

**Why harmonic.** Each shortcut zeroes at least one family by construction (Table 5.1/5.2). An arithmetic mean would let a shortcut that fails one family out of eight keep 87% of the axis. The repository's first toy run showed this happening: `answer_ban` scored 79.6 and the subject blocklist 51.6 under arithmetic means. Both fall below 5 under the harmonic form (`RED_TEAM.md`).

**Why worst-case R.** An attacker uses the attack that works.

### 6.4 Attacks (`attacks.py`)

- **Logit lens.** The last prompt position at every layer is read through the final norm and the unembedding. The forgotten answer's first token is ranked against 9 distractor first tokens from other answers in the split. The score is the upper-half maximum accuracy, chance-corrected and relative to the base model (100 = as readable as before).
- **Quantization.** Every linear weight is fake-quantized per output channel to int8 and int4, then the attack families (direct, paraphrase, question) are re-scored.
- **Relearning.** The requests are split by seeded hash into halves A and B. The model is fine-tuned for 20 steps (AdamW, lr 1e-5; MLP parameters by default, `--relearn-scope mlp_down` on memory-limited hardware, recorded in the result) on A's prompt→answer pairs, and B's probes are scored. Knowledge that was only hidden returns for facts that were never retrained. Fine-tuning also perturbs a model that forgot nothing (on Qwen2.5-0.5B the unedited model keeps only ~68% of B after the attack), so every run also attacks the base model (cached) and R uses the residual relative to it; both raw and relative residuals are in each `results.json`.

All attacks restore the model exactly (tested: `tests/test_end_to_end.py::test_attacks_restore_and_ga_changes_weights`).

### 6.5 Grader-exploit defences

- **Causality check** (`causality_violation`). For 16 known forget probes, every logit at answer positions is recomputed with the tokens *after* that position replaced. A causal LM changes nothing. Any change > 0.05 marks the run **INVALID**: a wrapper that "peeks" at the teacher-forced answer is a grader exploit (tested with a deliberate peeking wrapper; it was also caught on one of our own cheats during development, a batch-wide `min()`).
- **Boundary contract.** Methods that need to know where the user prompt ends (detect-then-edit methods such as SURE) declare `needs_prompt_lengths` and receive it, exactly as at inference. They never receive the answer.
- **Same decoder for base and method.** A wrapper that edits next-token scores (all the reference cheats) generates through HF `generate` with its rule applied as a causal `LogitsProcessor`; this is tested to give the same tokens as its own forward pass. A method that needs the request boundary declares `decode="full"` and uses the prefix-recompute decoder. The base reference is generated with whichever decoder the method uses, so near-tie differences between decoders can never count as forgetting.

### 6.6 Detect-then-edit decomposition (optional)

If a method exposes `route(prompts) → request_id | None`, the report splits the forget residual into:

- **routing misses**: the detector did not fire, or fired the wrong request;
- **actuation failures**: it fired correctly and the answer survived anyway;

and adds the retain false-fire rate. This generalises the "genie" analysis in this repository (perfect routing on the same edits) into a reporting protocol any detect-then-edit method can use.

### 6.7 Cost

The report includes unlearning wall-clock time; inference overhead (teacher-forced pass time relative to the base model); tensors changed and the parameters inside them (via weight fingerprints); and extra parameters added by runtime components.

### 6.8 Sequential deletion (dynamic setting)

Requests arrive in T batches (`sequential.py`), and the method is called through `unlearn_sequential`. After each batch the report scores (a) every batch so far on direct, paraphrase and question probes, and (b) every retain probe. It reports the per-step curve and the mean LETHE(E, L) over steps.

## 7. Statistics

- **Seeds** are different forget samples. The leaderboard reports mean ± sd over seeds (5 for the release).
- **95% CIs** come from a cluster bootstrap over requests: every probe of a request, and every retain control built around it, is resampled together (`metrics.bootstrap`).
- **Paired comparisons** between two methods on the same split use a paired cluster bootstrap with a two-sided p-value (`metrics.paired_bootstrap`).

## 8. Validation (`redteam.py`)

`validate_split` checks that:

- held-out forget probes differ from every request prompt;
- no retain text is in the method's view;
- no prompt is both forget and retain;
- no retain control asks a forgotten (subject, relation);
- every request has a known direct probe;
- there is one request per subject and no dangling references;
- probe ids are unique, the canary is present, and the fingerprint matches;
- for each reference cheat, a family that defeats it has ≥ 5 probes.

`cheat_gate` requires that every reference cheat scores below 35 on **both** LETHE and LETHE_core. A split that fails either check is not released.

## 9. Formats and reproducibility

- **Split files:** JSON (`schema.Split`): requests, probes, allowed_retain, utility sets, manifest with source sha256s, knowledge-filter yields, builder config, fingerprint and canary.
- **Results:** `results.json`, which holds the per-family scores (both scorers), aggregates, CIs, attacks, strict tier, routing, cost, environment and method card. The per-probe rows are in `probes.jsonl`.
- **Base-reference cache:** keyed by split fingerprint and eval config.
- **Tests:** model-free unit tests and an end-to-end run on a micro toy model (`pytest tests/`).
