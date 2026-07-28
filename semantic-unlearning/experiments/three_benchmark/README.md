# TOFU, MCF, and ZsRE experiment setup

This directory defines a reproducible three-benchmark evaluation built on the
existing dataset-native scripts. It deliberately does **not** force TOFU into
the MCF/ZsRE `Eff`, `Gen`, `Spe`, `PPL` schema because the reviewed TOFU
evaluator reports answer probability, truth ratio, normalized answer
probability, and ROUGE-L.

## Reviewed coverage

| Method | MCF | TOFU | ZsRE |
| --- | :---: | :---: | :---: |
| Base | yes | yes | yes |
| Original ZeroUnlearn | yes | yes | not implemented in the current repository |
| Full all tokens | yes | yes | not implemented in the current repository |
| Full selective tokens | yes | yes | not implemented in the current repository |
| Emb/LM all tokens | yes | yes | not implemented in the current repository |
| Emb/LM selective tokens | yes | yes | not implemented in the current repository |
| Setting 5e | yes | yes | yes |
| Setting 5e + protected/active LM-head repair | yes | yes | yes |

The launcher preserves the repository's benchmark-specific protocols:

- **MCF:** official split, 50 forget, 1,000 retain, seeds `0-9`, BF16.
- **TOFU:** `forget05`/`retain95`, 200 forget, 1,000 retain, fixed seed `42`.
  The existing Original ZeroUnlearn runner intentionally rejects other seeds.
- **ZsRE:** 50 forget, 1,000 retain, seeds `1-10`, BF16, `Unknown` as the
  neutral target.

## Run

From `semantic-unlearning/`:

```bash
# Print every command without launching GPU work.
DRY_RUN=1 bash scripts/run_three_benchmark_experiments.sh all /path/to/model

# Run one benchmark.
bash scripts/run_three_benchmark_experiments.sh mcf /path/to/model
bash scripts/run_three_benchmark_experiments.sh tofu /path/to/model
bash scripts/run_three_benchmark_experiments.sh zsre /path/to/model
```

Common overrides:

```bash
OUTPUT_ROOT=outputs/three_benchmark \
CUDA_VISIBLE_DEVICES=0 \
SKIP_EXISTING=1 \
bash scripts/run_three_benchmark_experiments.sh all /path/to/model
```

The complete run is expensive. `all` means the union of the three existing
pipelines, not a shared optimizer or a shared metric definition.

## Outputs

### MCF

The launcher writes:

```text
outputs/three_benchmark/mcf/
├── zerounlearn/seed*/
├── gagd/seed*/
├── repair/seed*/
└── aggregate/
    ├── per_seed.csv
    ├── aggregate.csv
    ├── aggregate.json
    └── aggregate.md
```

`aggregate_mcf_multimethod_results.py` refuses missing files, duplicate seeds,
seed mismatches, non-MCF payloads, and non-finite metrics. Standard deviation is
the population standard deviation, matching the existing multiseed scripts.

### TOFU

The existing TOFU chain writes native ECO-style summaries under:

```text
outputs/three_benchmark/tofu/pipeline/evaluation/
```

The Original ZeroUnlearn TOFU comparison is then written under:

```text
outputs/three_benchmark/tofu/zerounlearn/
```

Use TOFU's native fields when comparing methods:
`forget_answer_prob`, `retain_answer_prob`, `forget_truth_ratio`,
`retain_truth_ratio`, `forget_rougeL_recall`, and `retain_rougeL_recall`.

### ZsRE

The existing ZsRE launcher writes per-seed `zsre_results.json` files and the
validated aggregate under:

```text
outputs/three_benchmark/zsre/aggregate/
```

The aggregate refuses a seed whose selected active-repair candidate misses the
configured zero `Eff`/`Gen` target.

## Reported MCF result table

The following values were supplied with this experiment request. They are stored
verbatim in `mcf_reported_results.csv` and `mcf_reported_results.json`; rerunning
the launcher produces a separate computed aggregate.

| Method | Eff ↓ | Gen ↓ | Spe ↑ | PPL ↓ |
| --- | ---: | ---: | ---: | ---: |
| Base | 13.000 ± 4.837 | 13.900 ± 3.300 | 11.533 ± 1.933 | 11.0625 ± 0.0000 |
| Original ZeroUnlearn | 7.400 ± 3.470 | 10.700 ± 2.722 | 9.709 ± 1.718 | 11.2875 ± 0.1125 |
| Full all tokens | 10.000 ± 3.578 | 12.300 ± 2.795 | 12.528 ± 1.720 | 11.4062 ± 0.2359 |
| Full selective tokens | 10.000 ± 3.578 | 12.300 ± 3.100 | 12.887 ± 1.835 | 10.9313 ± 0.1884 |
| Emb/LM all tokens(s3) | 1.400 ± 1.800 | 7.300 ± 2.410 | 35.517 ± 2.336 | 31.6750 ± 1.7304 |
| Emb/LM selective tokens | 1.600 ± 1.497 | 5.600 ± 1.960 | 21.724 ± 1.031 | 15.7063 ± 3.4865 |
| Setting 5e (used setting 3) | 1.200 ± 0.980 | 5.200 ± 1.720 | 12.726 ± 1.651 | 11.3625 ± 0.4608 |
| 5e + protected LM-head repair | **0.000 ± 0.000** | **0.000 ± 0.000** | **13.091 ± 1.884** | **11.3625 ± 0.4608** |

## Interpretation

The protected LM-head repair is the strongest reported MCF point: it reaches
zero measured efficacy and generalization success while restoring specificity
to roughly the Base/Setting-5e range and keeping PPL unchanged from Setting 5e.
The unrepaired Emb/LM methods forget aggressively but cause severe PPL drift,
which is why they should be treated as intermediate ablations rather than final
models.
