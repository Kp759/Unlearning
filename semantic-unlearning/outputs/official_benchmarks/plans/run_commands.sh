#!/usr/bin/env bash
set -euo pipefail

cd /Users/kaustubh/Downloads/Unlearning/semantic-unlearning

# mcf_zerounlearn_official
env PYTHON_BIN="${PYTHON_BIN:-python}" OUTPUT_ROOT=outputs/official_benchmarks/runs/mcf_zerounlearn_official MCF_FORGET_NUM=50 MCF_RETAIN_NUM=1000 MCF_SEEDS="0 1 2 3 4 5 6 7 8 9" DTYPE=bf16 bash scripts/run_three_benchmark_experiments.sh mcf ${GENERIC_MODEL_PATH:?Set GENERIC_MODEL_PATH to the declared MCF/ZsRE target}

# zsre_zerounlearn_official
env PYTHON_BIN="${PYTHON_BIN:-python}" OUTPUT_ROOT=outputs/official_benchmarks/runs/zsre_zerounlearn_official ZSRE_SEEDS="1 2 3 4 5 6 7 8 9 10" DTYPE=bf16 bash scripts/run_three_benchmark_experiments.sh zsre ${GENERIC_MODEL_PATH:?Set GENERIC_MODEL_PATH to the declared MCF/ZsRE target}

# tofu_forget05
env PYTHON_BIN="${PYTHON_BIN:-python}" OUTPUT_ROOT=outputs/official_benchmarks/runs/tofu_forget05 TOFU_SEED=42 bash scripts/run_three_benchmark_experiments.sh tofu ${TOFU_FULL_MODEL_PATH:?Set TOFU_FULL_MODEL_PATH to the pinned official target model}

# muse_news: NEEDS_METHOD_EXTENSION
# The current Setting 5e producers and active repair require answer-token/fact-request structure; passages have no unchanged answer-row or desired-target repair semantics. Inventing QA pairs would change the information and record meaning.

# muse_books: NEEDS_METHOD_EXTENSION
# The current method has no document-sequence Setting 5e selection or protected active-repair definition. Turning Harry Potter passages into answer rows would alter the objective and record meaning.

# rwku: NEEDS_METHOD_EXTENSION
# Our method requires explicit forget facts/answers and protected repair records; RWKU provides only a target entity. Generating a corpus would add information and define a new method formulation.

# wmdp_bio: NEEDS_METHOD_EXTENSION
# The current answer-row Setting 5e and active repair do not define a sequence-corpus objective or repair target. The MC test cannot supply those targets.

# wmdp_cyber: NEEDS_METHOD_EXTENSION
# The method has no sequence-corpus Setting 5e or repair definition; adapting the MC test into training questions is explicitly invalid.

# wmdp_chem_eval: EVALUATION_ONLY
# No official Chem forget corpus is declared. WMDP-Chem may evaluate a Bio/Cyber checkpoint but cannot train our method.

# ugbench_tofu: EVALUATION_ONLY
# UGBench cases are an evaluation overlay. A matching official TOFU checkpoint may be reused, but generated cases cannot enter Setting 5e or repair.

# ugbench_harry_potter: EVALUATION_ONLY
# UGBench is evaluator-only and our method has no official Harry Potter sequence checkpoint producer; evaluation becomes possible only after a separate compatible method extension produces one.

# ugbench_zsre: EVALUATION_ONLY
# UGBench cases are evaluation-only. The existing ZsRE checkpoint can be reused only if its model/tokenizer/source identities exactly match UGBench.

# pch_continual: NEEDS_METHOD_EXTENSION
# The current method is one-shot and benchmark-tuned; it has no stateful sequential request/checkpoint contract. Applying later outcomes to retune selection would change its stopping and selection criteria.

# hubble_yago: NEEDS_METHOD_EXTENSION
# Although YAGO is factual, the current method assumes MCF/ZsRE-specific unwanted/desired target and repair gates. Hubble's minimal-pair target semantics have no validated unchanged adapter yet; defining them changes repair/selection semantics.

# hubble_gutenberg: NEEDS_METHOD_EXTENSION
# Gutenberg passages require sequence/document Setting 5e and repair semantics absent from the current answer-row method. Converting passages into invented QA changes record meaning.
