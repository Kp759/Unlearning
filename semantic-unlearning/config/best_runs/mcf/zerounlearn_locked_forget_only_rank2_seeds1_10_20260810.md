# MCF best run — ZeroUnlearn-style forget-only locked probes (Llama-3.2-3B-Instruct)

**Record:** `mcf-zerounlearn-locked-forget-only-rank2-seeds1-10-20260810`  
**Status:** `BEST_ACCEPTED`  
**Wulver job:** `1171704` (array seeds 1–10, all completed with exit code `0:0`)  
**Output root:** `outputs/mcf_zerounlearn_forget_only_locked_3b`

This is the preferred fair ZeroUnlearn-style MCF comparison for the 3B model. It replaces the older evaluator-conditioned MCF run as the current best comparison record, while preserving that older record under the append-only registry policy.

## Locked data-access protocol

During Stage 1 and Stage 2, SURE-LM receives only the **50 sampled forget records** and only their `requested_rewrite` prompts.

| Data / probe | Stage 1 | Stage 2 | Final evaluation |
|---|---:|---:|---:|
| 50 forget `requested_rewrite` prompts | yes | yes | yes |
| MCF retain records | 0 | 0 | 1000 |
| Forget paraphrases | no | no | yes |
| Forget neighborhood prompts | no | no | yes |
| Retain KL / calibration | no | no | n/a |
| Retain hidden projection | no | no | n/a |

The same 50 underlying forget facts are evaluated after unlearning. `Gen` therefore measures **same facts, unseen paraphrase formulations**, not unseen facts. The 1000 MCF retain records are evaluation-only.

## Method configuration

### Stage 1 — forget-only Setting 5e
- steps: `600`
- batch size: `1`
- embedding / LM-head LR: `1e-4`
- forget weight: `2.0`
- forget margin: `1.0`
- optimizer: `AdamW`
- post-training interpolation alphas: `0.75 / 0.50 / 0.25`

### Stage 2 — sparse LM-head repair
- repair mode: `minimal_optimize`
- active margin: `0.25`
- max repair steps: `100`
- repair LR: `0.005`
- optimizer: `AdamW`
- hinge weight: `2.0`
- delta L2: `1e-4`
- repair rank: `2`
- retain KL: `0`
- retain calibration records: `0`
- project away retain hidden states: `false`
- trainable repair scope: selected active LM-head rows only

## 10-seed aggregate

Primary standard deviations below use NumPy population SD (`np.std`, `ddof=0`), matching the terminal aggregation used for this record.

| Metric | Mean ± population SD | Direction |
|---|---:|---|
| Eff | **0.0000 ± 0.0000** | ↓ |
| Gen | **4.0000 ± 3.6332** | ↓ |
| Spe | **27.7110 ± 3.6742** | ↑ |
| Spe_success | **96.3000 ± 1.8639** | ↑ |
| PPL | **11.5500 ± 0.6771** | ↓ / stable |
| Minimum rewrite/paraphrase margin | **-2.2422 ± 2.0570** | diagnostic |

Across all seeds:
- direct rewrite prompts: **500 / 500 pass** (`Eff = 0` on every seed)
- held-out paraphrase prompts: **960 / 1000 pass**
- total paraphrase failures: **40 / 1000**
- held-out paraphrase pass rate: **96.0%**

## Per-seed results

| Seed | Eff | Gen | Spe | PPL | Rewrite failures | Paraphrase failures | Min margin |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.00 | 2.00 | 26.20 | 12.750 | 0 | 2 | -0.5625 |
| 2 | 0.00 | 1.00 | 24.50 | 11.062 | 0 | 1 | -2.7500 |
| 3 | 0.00 | 3.00 | 27.08 | 12.375 | 0 | 3 | -0.84375 |
| 4 | 0.00 | 5.00 | 32.02 | 11.062 | 0 | 5 | -3.03125 |
| 5 | 0.00 | 1.00 | 34.56 | 11.062 | 0 | 1 | -0.2500 |
| 6 | 0.00 | 0.00 | 31.01 | 11.062 | 0 | 0 | 0.6875 |
| 7 | 0.00 | 3.00 | 24.47 | 12.562 | 0 | 3 | -1.34375 |
| 8 | 0.00 | 5.00 | 29.73 | 11.062 | 0 | 5 | -6.390625 |
| 9 | 0.00 | 7.00 | 23.73 | 11.062 | 0 | 7 | -3.71875 |
| 10 | 0.00 | 13.00 | 23.81 | 11.438 | 0 | 13 | -4.21875 |

Seed 10 is the largest held-out Gen tail case and is intentionally retained in the aggregate.

## Seed-1 base-model utility check

This is an auxiliary **seed-1-only** comparison on the exact same 50-forget / 1000-retain split; it is not a 10-seed utility aggregate.

| Metric | Base | Unlearned | Delta |
|---|---:|---:|---:|
| Retain Eff | 12.50 | 13.10 | +0.60 |
| Retain Gen | 14.35 | 14.60 | +0.25 |
| Retain Spe | 11.84 | 14.46 | +2.62 |
| Retain Spe_success | 84.82 | 83.72 | -1.10 |
| PPL | 11.0625 | 12.75 | +1.6875 |

Forget-side seed-1 change:
- Eff: `16.00 → 0.00`
- Gen: `14.00 → 2.00`
- Spe: `12.57 → 26.20`

A full 10-seed base-vs-unlearned retain comparison still needs to be aggregated before making a final retain-utility claim.

## Scientific interpretation

- `Eff = 0` on all ten seeds is a clean direct-prompt forgetting result.
- `Gen = 4.0` is a genuine held-out-prompt result because official paraphrases never enter Stage 1, Stage 2, selection, or repair.
- The negative minimum-margin tail shows that a small number of unseen paraphrases can still strongly retrieve the association; these cases must **not** be repaired using held-out probes.
- The 1000 MCF retain examples are evaluation-only and do not protect the model during training.
- Absolute PPL should not yet be claimed as directly better than the ZeroUnlearn paper until base/evaluator PPL reproduction is aligned.

## Provenance still to capture

- copy the raw per-seed `official_eval_locked.json` artifacts into persistent release metadata or a compact Git-tracked aggregate
- capture checkpoint SHA-256 hashes
- aggregate the base model on all ten corresponding 1000-retain splits
- capture Stage-2 complexity across seeds (selected LM-head row count, actual rank, repair steps, and delta norm)
