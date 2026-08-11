# Llama-3.2-3B-Instruct / MCF — current best locked-probe result

Canonical record: `config/best_runs/mcf/zerounlearn_locked_forget_only_rank2_seeds1_10_20260810.json`

This model-centric snapshot marks the **ZeroUnlearn-style forget-only locked-probe** run as the preferred 3B MCF comparison.

## Protocol

- seeds: `1–10`
- training/unlearning access: `50 forget + 0 MCF retain`
- Stage 1/2 prompt access: `requested_rewrite` only
- official paraphrases: locked until final evaluation
- neighborhood prompts: locked until final evaluation
- final evaluation: same 50 forget facts + 1000 evaluation-only retain records
- Stage-2 repair rank: `2`
- retain KL/calibration/projection during repair: disabled
- Wulver job: `1171704`

## Aggregate

| Metric | Mean ± population SD |
|---|---:|
| Eff ↓ | **0.0000 ± 0.0000** |
| Gen ↓ | **4.0000 ± 3.6332** |
| Spe ↑ | **27.7110 ± 3.6742** |
| Spe_success ↑ | **96.3000 ± 1.8639** |
| PPL ↓ / stable | **11.5500 ± 0.6771** |

Prompt-level totals:
- rewrite prompts: `500`, failures: `0`
- held-out paraphrase prompts: `1000`, failures: `40`
- held-out paraphrase pass rate: `96.0%`

## Why this is the preferred MCF record

The older `official_protected_v2_seeds0_9` run achieved `Eff=0` and `Gen=0`, but its repair path was evaluator-conditioned. In this record, official paraphrases and neighborhood prompts do not enter Stage 1, Stage 2, repair selection, or repair optimization. Therefore `Gen=4.0` is the cleaner held-out-prompt result and should be used for the fair ZeroUnlearn-style comparison.

The prior record remains preserved under the append-only best-run policy.

## Caveats

- Seed 10 has `Gen=13.0` and is retained in the aggregate.
- A few held-out paraphrases have strongly negative margins; do not tune or repair against these final-evaluation probes.
- The 1000 MCF retain records are evaluation-only, not training protection.
- The full 10-seed base-vs-unlearned retain delta is still pending.
- Absolute PPL should not be claimed as directly better than the ZeroUnlearn paper until the base/evaluator PPL reproduction is aligned.
