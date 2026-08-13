# Llama-3.2-3B-Instruct / MCF — technical specification

Companion to `zerounlearn_locked_forget_only_rank2_seeds1_10_20260810.json`. The JSON technical spec in this folder is the machine-readable source of the dimensional and protocol details below.

## Protocol and exact ZeroUnlearn split

The source is `data/multi_counterfact.json` with 21,919 records. `sample_official_mcf_records` performs:

- `half = len(data)//2 = 10,959`;
- retain pool = first half, indices `[0,10959)`, 10,959 records;
- forget pool = second half, indices `[10959,21919)`, 10,960 records;
- one `random.Random(seed)`;
- sample forget **first**, then retain;
- seeds `1..10`;
- 50 forget records/seed;
- Stage 1/2 retain records = **0**;
- final-eval retain records = **1,000**.

Training and final evaluation get the exact same 50 forget facts because forget is the first RNG draw. Stage 1/2 request retain=0; final evaluation requests retain=1000, but that later retain draw cannot change the already-completed forget draw.

`build_mcf_zerounlearn_locked_split.py` creates a repair-visible copy with record order and `requested_rewrite` preserved while emptying `paraphrase_prompts`, `neighborhood_prompts`, and `generation_prompts`. It explicitly checks that source and sanitized copies select identical record indices. This is a **prompt-level holdout**: same forget facts, but paraphrase/neighborhood formulations are unseen until final evaluation.

## Stage 1 — Embedding + LM-head matrix

For Llama-3.2-3B-Instruct:

| Quantity | Value |
|---|---:|
| vocab size | 128,256 |
| hidden size | 3,072 |
| embedding matrix | `128256 x 3072` |
| LM-head matrix | `128256 x 3072` |
| tied in Stage 1 | yes |
| unique trainable matrix entries | **394,002,432** |
| observed total model params | 3,212,749,824 |
| trainable fraction | **12.263713%** |
| raw BF16 matrix size | 788,004,864 bytes = **751.5 MiB** |

Input embeddings and LM head share one storage tensor, so Stage 1 has **one** trainable `128256 x 3072` matrix, not two. All transformer parameters are frozen.

### Stage-1 loss and optimization

The accepted forget-only code uses `mcf_margin_forget_loss`:

`L_i = softplus(NLL(target_true) - NLL(target_new) + 1.0)`

and minimizes `2.0 * mean(L_i)`. Thus it pushes the margin `NLL(target_new)-NLL(target_true)` positive so the factual `target_true` is preferred to the counterfactual `target_new`.

Settings: 600 steps, batch 1, AdamW, LR `1e-4`, weight decay 0, grad clip 1.0, BF16. With 50 forget records, the epoch sampler gives exactly 12 full passes. No benchmark-retain CE or KL is used. The historical code is grouped under Setting-5e/GA-GD, but this accepted locked implementation's code-exact objective is the target-new-vs-target-true margin above, not a 1,000-record retain-GD term.

## Vocabulary restoration after Stage 1

The base tied matrix is snapshotted. After training, the whole matrix is restored to base and only allowed answer-token rows are reapplied:

- unique `target_new`: keep full trained row;
- unique `target_true`: `1.25 * base`;
- `target_new ∩ target_true`: `base + 0.75*(trained-base)`;
- unrelated rows: base;
- retain-overlap alpha branches (`0.50`, `0.25`) are configured but inert because the locked run passes an empty retain set to the token-group builder.

## Stage 2 — sparse active LM-head repair

Stage 2 unties the LM head by cloning the tied output matrix, then freezes the transformer and input embeddings. The full output head remains `128256 x 3072`, but only selected rows may change.

The sanitized dataset yields exactly 50 direct rewrite prompt instances. Margin is:

`m = NLL(target_new)-NLL(target_true)`.

An instance is active when `m < 0.25`. Selected rows are the union of non-special tokenizer IDs occurring in `target_new` and `target_true` for initially active direct rewrites. The selected row set is then fixed.

Repair objective:

`2.0 * sum ReLU(required_margin-m)^2 + 1e-4 * ||DeltaW||_F^2`

with AdamW, LR `0.005`, max 100 steps, early stopping when all 50 direct rewrite constraints are satisfied. Retain KL=0, retain calibration=0, retain hidden projection disabled, and no PPL/rephrase/neighborhood data participates.

## What rank=2 actually means

`repair_rank=2` is a maximum **hidden-direction SVD rank**, not two LM-head rows and not PEFT LoRA. Active answer-position hidden vectors are concatenated and SVD supplies a fixed basis `B`. The trainable update is:

`DeltaW_selected = C @ B_fixed`

where `C` has shape `[selected_rows, actual_rank]`, `B_fixed` has shape `[actual_rank,3072]`, and only `C` is trained.

## Observed repair dimensions across 10 seeds

| Seeds | Selected rows | Requested rank | Actual rank | Trainable C | Fixed basis B | Materialized selected delta |
|---|---:|---:|---:|---:|---:|---:|
| 1,2,5,8,10 | 2 | 2 | **1** | `2x1` = **2 params** | `1x3072` | `2x3072` = **6,144 entries** |
| 3,4,6,7,9 | 0 | 2 | 0 | 0 | none | 0 |

Therefore, on every repaired seed:

- conceptual full sparse repair matrix = `128256 x 3072`;
- nonzero rows = **2**;
- rank <= **1**;
- materialized changed positions = 6,144 versus 394,002,432 in a dense full LM head (**64,128x fewer positions**);
- trainable Stage-2 coefficients = 2 versus 394,002,432 Stage-1 trainables (**197,001,216x fewer trainable values**).

These per-seed row/rank facts are also preserved by the later fixed-SVD-vs-sparse-LoRA ablation, which reused the exact Stage-1 checkpoints.

## Final evaluator

Only after Stage 2 is frozen does `mcf_zero_unlearn_official_eval.py` reopen the original unsanitized MCF source. Per seed it evaluates 50 forget + 1,000 retain records. Each forget record supplies 1 rewrite, 2 paraphrases, and 10 neighborhood prompts.

- Eff: direct rewrites still preferring `target_new`; lower is better.
- Gen: held-out paraphrases still preferring `target_new`; lower is better.
- Spe: neighborhood probability-difference specificity; higher is better.
- Spe_success: neighborhood prompts preferring `target_true`; higher is better.
- margin: `NLL_new-NLL_true`; negative means a forget failure.

PPL is one ZeroUnlearn-compatible Wikidata scalar: the evaluator joins the first 20 Wikidata train text rows, truncates tokenization to 100, computes PPL once, and stores the same scalar under `forget_PPL` and `retain_PPL`. PPL is never used for training or selection.

## Accepted 10-seed result

Population SD:

| Metric | Mean ± SD |
|---|---:|
| Eff ↓ | **0.0000 ± 0.0000** |
| Gen ↓ | **4.0000 ± 3.6332** |
| Spe ↑ | **27.7110 ± 3.6742** |
| Spe_success ↑ | **96.3000 ± 1.8639** |
| PPL ↓/stable | **11.5500 ± 0.6771** |

Totals: 500/500 direct rewrites pass; 960/1000 held-out paraphrases pass. Seed 10 (`Gen=13`) is retained as the largest tail case.

## Exact code provenance

- `scripts/mcf_sampling.py`
- `scripts/build_mcf_zerounlearn_locked_split.py`
- `scripts/mcf_forget_only_setting5e.py`
- `scripts/gagd_compare.py`
- `scripts/mcf_forget_only_active_repair.py`
- `scripts/gagd_active_case_repair.py`
- `scripts/mcf_zero_unlearn_official_eval.py`
- `scripts/run_mcf_zerounlearn_locked_our_method.sh`
- `slurm/run_mcf_zerounlearn_locked_3b.slurm`
- Wulver job `1171704`, output `outputs/mcf_zerounlearn_forget_only_locked_3b`.

## Details still only in local Wulver artifacts

The existing Git best-run snapshot does **not** contain exact per-seed active counts, selected token IDs/tokens, delta norms, actual early-stop step counts, Stage-1 row-group counts, or checkpoint hashes. These are not guessed here. Read them from each seed's `repair_summary.json`, `token_group_report.json`, and `post_training_row_policy.json` under the output root if a deeper archival capture is required.

Implementation caveat: the generic active-repair module has generic paraphrase-enabled metadata, but the locked runner supplies a sanitized file with empty paraphrase lists and then patches provenance to `repair_uses_official_paraphrases=false` / `repair_prompt_scope=requested_rewrite_only`. The actual run is rewrite-only; moving that metadata derivation into the core repair module would make future crash-time provenance cleaner.
