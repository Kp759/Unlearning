# Official-benchmark-first evaluation (Stage 1)

This framework audits, plans, invokes, and aggregates official benchmark
protocols for the repository's existing final method:

> **Setting 5e + protected/active LM-head repair**

It does not rename plain GA/GD, Original ZeroUnlearn, prompt-conditional
repair, RMU, PERMU, RULE, FIT, or SHRED as our method. It never creates a
custom train/validation/test split. Split design and portability experiments
are intentionally deferred until official results have been frozen.

## What was found in this repository

The current method has three benchmark-specific checkpoint-production paths.
These paths are the callable definition of `our_method`; there is no valid
generic substitute.

| Track | Setting 5e checkpoint production | Protected/active repair | Native evaluation |
| --- | --- | --- | --- |
| MCF | `scripts/gagd_compare.py --mode emb_lm_all_restore_post_training_true` | `scripts/gagd_active_case_repair.py`, called by `scripts/run_gagd_active_case_repair.sh` | `scripts/mcf_zero_unlearn_official_eval.py` through `scripts/run_three_benchmark_experiments.sh` |
| ZsRE | `scripts/zsre_gagd_setting5e_active_repair.py` (Stage 1) | same file (Stage 2), called by `scripts/run_zsre_gagd_setting5e_active_repair.sh` | `scripts/zsre_zero_unlearn_official_eval.py` |
| TOFU | `scripts/tofu_gagd_four_settings_official.py --mode emb_lm_all_tokens`, then `scripts/tofu_gagd_setting5e_restore.py` | `scripts/tofu_gagd_active_forget_repair.py`, called by `scripts/run_tofu_gagd_neighborhood_confidence.sh` | `scripts/tofu_eval.py` |

`scripts/run_three_benchmark_experiments.sh` remains the stable wrapper for all
three. Its interface and output layout are unchanged.

### Immutable mathematical core

The common core is:

1. produce the existing benchmark-native Setting 5e checkpoint;
2. freeze the transformer for the repair stage;
3. freeze input embeddings for the repair stage;
4. edit only selected LM-head row deltas;
5. derive active constraints from still-memorized official forget cases;
6. derive protected constraints from the benchmark's permitted retain/utility
   calibration roles;
7. select a materialized candidate with the existing benchmark-specific hard
   forget and utility gates.

The objective, editable parameters, repair projection, candidate selection,
stopping rules, and information available to the method are immutable. An
adapter may only change representation, formatting, batching, tokenization,
or invocation glue.

### Existing preprocessing and benchmark-specific pieces

- **MCF:** the official second-half forget / first-half retain sampler is in
  `scripts/mcf_sampling.py`. `target_new` is the unwanted answer. Setting 5e
  uses `target_new`/`target_true`/retain token groups and overlap alphas
  `0.75/0.50/0.25`. The repair operates on officially active requested and
  paraphrased cases while protecting native utility cases.
- **ZsRE:** the adapter maps the original answer to the internal unwanted slot
  and the one-token neutral answer `Unknown` to the desired slot. This mapping
  is representation glue required because the MCF field convention is the
  opposite. The repair changes only the `Unknown` output row, uses official
  metric-case identities, and replays official BF16 batching during scale
  selection.
- **TOFU:** complete author QA profiles are retained. The Setting 5e analogue
  groups answer-position rows, fully restores shared/protected rows and the
  complete input matrix, and keeps unique forget output-row updates. Active
  repair enforces the absolute forget-answer probability gate while protecting
  retain95, real-authors, and world-facts. Perturbed prompts are evaluator-only.

Evaluation-only code includes the MCF/ZsRE official evaluators, `tofu_eval.py`,
all UGBench implicit/generalization cases, RWKU probes, WMDP multiple-choice
tests, MUSE retain2/holdout roles, PCH later outcomes/attacks, and Hubble
keep/test/privacy probes. None may be used to select, tune, repair, or stop an
official method run except where the existing native protocol explicitly
defines a permitted calibration role.

The repair settings are not portable constants. MCF margins/ranks, the ZsRE
`Unknown` row and official token identities, and TOFU's answer-probability,
complete-profile row groups, and utility constraints are benchmark-specific.
They cannot be copied to a new input contract without a method extension.

## Taxonomy

The registry contains 15 objects classified as `benchmark`. It separately
contains `evaluation_profile` and `baseline_method` objects.

- **Benchmarks:** MCF, ZsRE, TOFU, MUSE-News, MUSE-Books, RWKU, WMDP-Bio,
  WMDP-Cyber, WMDP-Chem evaluation, UGBench-TOFU, UGBench-Harry-Potter,
  UGBench-ZsRE, PCH continual, Hubble-YAGO, and Hubble-Gutenberg.
- **Evaluation profiles:** RULE-compatible reporting for RWKU/MUSE-Books and
  SHRED-paper-compatible axes for TOFU, MUSE, RWKU, and Hubble.
- **Baseline methods:** Original ZeroUnlearn, RMU, PERMU, RULE, FIT, and SHRED.

RMU is not WMDP data; PERMU is not UGBench data; RULE is not a dataset; FIT is
not PCH; and SHRED is not a dataset. As of the Stage 1 source audit, no
verifiable official SHRED code repository was linked from the paper/authors.
SHRED is therefore paper-specified evaluation metadata only. The framework
does not reimplement it from memory.

## Compatibility matrix

The declared status describes method compatibility. `doctor` reports a
separate effective blocking status when a compatible track lacks a pin, model,
dataset, or evaluator.

| Benchmark ID | Input contract | Declared status | Reason |
| --- | --- | --- | --- |
| `mcf_zerounlearn_official` | QA/fact request | `READY_NATIVE` | Existing Setting 5e producer, repair, sampler, and evaluator |
| `zsre_zerounlearn_official` | QA/fact request | `READY_WITH_DATA_ADAPTER` | Existing semantics-preserving original-answer/`Unknown` adapter |
| `tofu_forget05` | QA/fact request | `READY_WITH_DATA_ADAPTER` | Existing complete-profile TOFU row adapter and repair |
| `muse_news` | sequence/document | `NEEDS_METHOD_EXTENSION` | No passage-level Setting 5e or active-repair semantics |
| `muse_books` | sequence/document | `NEEDS_METHOD_EXTENSION` | No passage-level Setting 5e or active-repair semantics |
| `rwku` | target entity only | `NEEDS_METHOD_EXTENSION` | Method needs explicit facts/answers; probes cannot become a corpus |
| `wmdp_bio` | sequence/document | `NEEDS_METHOD_EXTENSION` | No corpus-sequence repair formulation; MC test is evaluator-only |
| `wmdp_cyber` | sequence/document | `NEEDS_METHOD_EXTENSION` | No corpus-sequence repair formulation; MC test is evaluator-only |
| `wmdp_chem_eval` | evaluation overlay | `EVALUATION_ONLY` | No declared official Chem forget corpus |
| `ugbench_tofu` | evaluation overlay | `EVALUATION_ONLY` | Evaluate only an exact-identity TOFU checkpoint |
| `ugbench_harry_potter` | evaluation overlay | `EVALUATION_ONLY` | No unchanged-method Harry Potter checkpoint producer |
| `ugbench_zsre` | evaluation overlay | `EVALUATION_ONLY` | Evaluate only an exact-identity ZsRE checkpoint |
| `pch_continual` | sequential deletion | `NEEDS_METHOD_EXTENSION` | Current method has no frozen-config sequential state contract |
| `hubble_yago` | QA/fact request | `NEEDS_METHOD_EXTENSION` | Minimal-pair desired/protected repair semantics are not defined |
| `hubble_gutenberg` | sequence/document | `NEEDS_METHOD_EXTENSION` | No passage-level Setting 5e or active-repair semantics |

Thus, the unchanged method is runnable only on MCF, ZsRE, and TOFU after their
artifacts are pinned and present. ZsRE and TOFU already have checked-in data
adapters; no new adapter is claimed for an unsupported contract.

## Official target-model requirements

An official result requires the exact model and tokenizer declared by that
benchmark/reproduction. The role must also be correct.

| Tracks | Required role |
| --- | --- |
| MCF, ZsRE | the declared experiment `Base` target; the generic Llama path is allowed only here |
| TOFU | official Full-TOFU model plus retain95/Target comparison |
| MUSE-News/Books | corpus-specific official Target plus retain model |
| RWKU | official original task model |
| WMDP | official task target; RMU checkpoint only for the optional RMU baseline |
| UGBench | overlay-specific official target/configuration and exact tokenizer |
| PCH | PCH-finetuned starting model plus official retain model |
| Hubble | released perturbed `Full` and standard `Target` minimal pair |

One generic checkpoint must never be substituted across benchmarks. `doctor`
and `run` reject the generic model for TOFU, MUSE, UGBench, PCH, and Hubble.

## Native metrics and directions

- **MCF/ZsRE:** Eff ↓, Gen ↓, Spe ↑, PPL ↓.
- **TOFU:** forget answer probability and forget ROUGE-L ↓; retain answer
  probability/ROUGE-L, real-authors/world-facts normalized probability and
  ROUGE-L, model utility, and retain-only comparison quality ↑. Truth ratio is
  reference/distributional and is not a standalone monotonic score.
- **MUSE:** `verbmem_f` ↓, `knowmem_f` ↓, `knowmem_r` ↑, `privleak` ↓.
- **RWKU:** forget/adversarial/MIA leakage ↓; neighbor, utility, and fluency ↑.

Protocol metadata distinguishes native data/metrics from checkpoint-selection
independence. Affected protected-repair rows for MCF, ZsRE, and TOFU carry
`native_data_and_metrics_but_evaluation_conditioned_repair` when official
paraphrase/correctness or utility-calibration evidence influenced repair.
Base and unrepaired Setting-5e-only rows do not inherit that status. RWKU's
probe-assisted entity-fact track is
`nonofficial_probe_assisted_entity_fact_portability`; its target-generated
corpus extension is
`official_protocol_different_model_confirmatory_method_extension` on the
declared different model. See `RWKU_EXPERIMENT.md` for the locked-stage
protocol.
- **WMDP:** accuracy on the selected forgotten domain ↓ and MMLU utility ↑;
  other WMDP domain accuracies are retained as native reported diagnostics.
- **UGBench:** explicit and implicit/generalization knowledge retention ↓;
  utility ↑.
- **PCH:** Forget Degree ↑ and Retain Utility ↑, with underlying forget
  probability/ROUGE-L/token accuracy ↓ and retain components ↑.
- **Hubble:** unlearn-set memorization and privacy leakage ↓; keep/test utility
  ↑.

Aggregation stores these names and directions as-is. It deliberately computes
no universal “unlearning score.”

## Data-role boundaries

- MCF and ZsRE use their existing official forget/retain sampling. Native
  generalization, neighborhood, and PPL prompts/corpora remain evaluator-only.
- TOFU exposes complete `forget05` and `retain95` profiles. Perturbed questions
  and retain-only logs remain evaluator-only.
- MUSE exposes forget and retain1/calibration passages only. Retain2 and
  holdout/nonmember data are evaluator-only.
- RWKU exposes only target entity and original model. Every forget-level,
  neighbor, MIA, utility, and fluency probe is evaluator-only.
- WMDP exposes only the official Bio or Cyber forget corpus. Multiple-choice
  WMDP questions and MMLU are evaluator-only. Chem has no training role unless
  a pinned official release adds one.
- UGBench-generated paraphrase, subject-replacement, inverse-relation,
  one-hop, and other implicit/generalization records are evaluator-only.
- PCH exposes requests in their official sequence. Future requests, later
  outcomes, relearning, quantization, and utility probes cannot retune earlier
  checkpoints.
- Hubble uses released perturbations and standard/perturbed minimal pairs.
  Keep/test/privacy probes are evaluator-only.

## Environment

The commands below match the training environment. Do not replace these roots
with unrelated scratch paths.

```bash
cd /mnt/train/Unlearning/semantic-unlearning

export PYTHON_BIN=/mnt/train/venvs/unlearning/bin/python
export HF_HOME=/mnt/train/hf
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export GENERIC_MODEL_PATH=/mnt/train/models/Llama-3.2-3B-Instruct
export OFFICIAL_BENCH_ROOT=/mnt/train/official-unlearning-benchmarks
export OFFICIAL_MODEL_ROOT=/mnt/train/models/official-unlearning
export OFFICIAL_DATA_ROOT=/mnt/train/data/official-unlearning
export OFFICIAL_MODELS_CONFIG=$PWD/config/official_benchmarks/models.json
export OFFICIAL_SOURCE_LOCK=$PWD/config/official_benchmarks/source_lock.json

cp config/official_benchmarks/models.example.json "$OFFICIAL_MODELS_CONFIG"
cp config/official_benchmarks/source_lock.example.json "$OFFICIAL_SOURCE_LOCK"
```

The copied examples are intentionally unresolved and cannot produce an
official run. Replace every placeholder with an immutable identity, then run
`doctor`.

## Source-pinning and setup workflow

For every upstream repository:

1. select an immutable 40-hex commit from the official repository;
2. clone under `$OFFICIAL_BENCH_ROOT` and check out that commit detached;
3. write its URL, checkout, and commit to `source_lock.json`;
4. pin each Hugging Face dataset/model to a commit, not `main`;
5. record a dataset fingerprint or content SHA-256;
6. record the evaluator revision independently when it differs;
7. download the official role-specific model to `$OFFICIAL_MODEL_ROOT` and
   record model ID, revision, architecture, tokenizer ID, and tokenizer
   revision in `models.json`;
8. rerun `doctor` and resolve every blocking status.

No clone, dataset download, or model download is performed by this Stage 1
implementation task. Exact source setup templates for every track are emitted
by:

```bash
$PYTHON_BIN scripts/official_benchmarks.py plan \
  --suite all \
  --method our_method \
  --output-dir outputs/official_benchmarks/plans

less outputs/official_benchmarks/plans/setup_commands.sh
```

Shared upstream sources mean some track setup commands are intentionally
identical: MCF/ZsRE share ZeroUnlearn; Bio/Cyber/Chem share WMDP and lm-eval
v0.4.2; the three overlays share UGBench; and YAGO/Gutenberg share Hubble.
Hubble's published evaluation harness revision is recorded separately from the
WMDP v0.4.2 pin.

## CLI

All subcommands except an explicitly executed `run` are CPU-only and avoid
importing torch or initializing CUDA.

```bash
$PYTHON_BIN scripts/official_benchmarks.py inventory

$PYTHON_BIN scripts/official_benchmarks.py doctor \
  --suite all \
  --output-dir outputs/official_benchmarks/audit

$PYTHON_BIN scripts/official_benchmarks.py plan \
  --suite all \
  --method our_method \
  --output-dir outputs/official_benchmarks/plans

# Dry-run is the default. It still refuses an unsupported or unresolved track.
$PYTHON_BIN scripts/official_benchmarks.py run \
  --benchmark mcf_zerounlearn_official \
  --method our_method \
  --output-dir outputs/official_benchmarks/runs/mcf_zerounlearn_official

# GPU work happens only with the explicit flag.
$PYTHON_BIN scripts/official_benchmarks.py run \
  --benchmark mcf_zerounlearn_official \
  --method our_method \
  --output-dir outputs/official_benchmarks/runs/mcf_zerounlearn_official \
  --execute

$PYTHON_BIN scripts/official_benchmarks.py aggregate \
  --runs-root outputs/official_benchmarks/runs \
  --output-dir outputs/official_benchmarks/aggregate
```

`plan` writes `plan.json`, `setup_commands.sh`, `run_commands.sh`, and one
reserved `run_manifest.json` per track. `doctor` writes `doctor.json` and
`doctor.md`. `aggregate` writes a long-form native-metric CSV plus JSON and
Markdown without inventing a cross-benchmark score.

## Exact official commands for the ready tracks

After `doctor` reports the track as official-ready, the following wrappers
preserve the existing protocol and output layout:

```bash
# MCF: official sampler, 50/1000, seeds 0-9, BF16.
env PYTHON_BIN="$PYTHON_BIN" \
  OUTPUT_ROOT=outputs/official_benchmarks/runs/mcf_zerounlearn_official \
  MCF_FORGET_NUM=50 MCF_RETAIN_NUM=1000 \
  MCF_SEEDS="0 1 2 3 4 5 6 7 8 9" DTYPE=bf16 \
  bash scripts/run_three_benchmark_experiments.sh mcf "$GENERIC_MODEL_PATH"

# ZsRE: official sampler, 50/1000, seeds 1-10, BF16, neutral Unknown.
env PYTHON_BIN="$PYTHON_BIN" \
  OUTPUT_ROOT=outputs/official_benchmarks/runs/zsre_zerounlearn_official \
  ZSRE_SEEDS="1 2 3 4 5 6 7 8 9 10" DTYPE=bf16 \
  bash scripts/run_three_benchmark_experiments.sh zsre "$GENERIC_MODEL_PATH"

# TOFU: official Full checkpoint, forget05/retain95, seed 42.
env PYTHON_BIN="$PYTHON_BIN" \
  OUTPUT_ROOT=outputs/official_benchmarks/runs/tofu_forget05 TOFU_SEED=42 \
  bash scripts/run_three_benchmark_experiments.sh tofu "$TOFU_FULL_MODEL_PATH"
```

The unified `run --execute` command invokes these wrappers; it does not
duplicate their optimization or evaluators.

## Provenance and fail-closed official labeling

Every plan reserves and every run writes `run_manifest.json`. It records the
repository commit/dirty state, hashes of all current method implementation
files, complete command, source and evaluator revisions, dataset identity and
fingerprint, model/tokenizer role and identity, complete method
hyperparameters, seeds, dtype/device, dependencies, checkpoint hash when
available, native metric schema, status, and failure reason.

`run` refuses before subprocess launch when any required revision, artifact,
model role, dataset identity, or evaluator is unresolved. A generic checkpoint
is never used as fallback. A manifest is labeled `official_protocol: true`
only after a resolved native command succeeds. Base/Full and unlearned models
must be evaluated on identical examples, and test-driven hyperparameter search
is prohibited.

## Result labels

Keep these three claims distinct:

1. **Official benchmark reproduction:** exact official model roles, data roles,
   evaluator, metrics, and locked revisions.
2. **Official protocol on a different model:** native roles/evaluator are kept,
   but the model is explicitly different; this is not benchmark reproduction.
3. **Non-official portability experiment:** data/model/evaluator or method
   semantics differ. It must not be labeled official.

Stage 1 supports the first claim only when every doctor check passes. It does
not conceal unsupported tracks by manufacturing results or training data.
