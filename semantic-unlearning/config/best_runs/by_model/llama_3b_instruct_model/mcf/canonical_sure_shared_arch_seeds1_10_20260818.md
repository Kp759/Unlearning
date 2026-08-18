# MCF — Canonical SURE-LM shared architecture, seeds 1–10

**Model:** `meta-llama/Llama-3.2-3B-Instruct`  
**Date:** 2026-08-18  
**Status:** canonical cross-benchmark architecture run  
**Dataset:** MultiCounterFact (MCF)  
**Seeds:** 1–10  
**Main-table metrics only:** `Eff. ↓`, `Gen. ↓`, `Spe. ↑`, `PPL ↓`

This record archives the MCF result produced by the shared canonical SURE-LM implementation used for both MCF and ZsRE. It supersedes the earlier benchmark-specific implementation **for the purpose of architecture-aligned cross-benchmark reporting**, but older MCF snapshots remain in this folder for provenance and may be numerically stronger on some metrics.

## 1. Final 10-seed result

All values are **mean ± sample standard deviation** across seeds 1–10.

| Method | Eff. ↓ | Gen. ↓ | Spe. ↑ | PPL ↓ |
|---|---:|---:|---:|---:|
| Base | **13.6000 ± 4.5019** | **14.8000 ± 2.3476** | **11.4620 ± 2.0742** | **11.0625 ± 0.0000** |
| Canonical SURE-LM | **0.0000 ± 0.0000** | **4.1000 ± 4.3830** | **9.6170 ± 2.3648** | **11.1937 ± 0.1779** |

Metric directions:

- `Eff. ↓`: lower is better forgetting on the direct rewrite prompts.
- `Gen. ↓`: lower is better forgetting on held-out paraphrases of the same facts.
- `Spe. ↑`: higher is better neighborhood specificity/preservation.
- `PPL ↓`: lower/stable is better language-model fluency.

Only these four metrics are intended for the main paper table. Retain-set diagnostics, prompt margins, repair ranks, row counts, and other debugging statistics remain useful for reproducibility but are not main-table columns.

## 2. Locked data protocol

Each seed uses the ZeroUnlearn-compatible split convention:

1. split the MCF pool into first-half and second-half pools;
2. sample `50` forget records from the second half;
3. reserve `1000` retain records from the first half for final evaluation;
4. materialize a seed-specific direct-only training artifact;
5. expose only `requested_rewrite` information during Stage 1 and Stage 2;
6. keep paraphrases, neighborhood prompts, retain evaluation, and PPL hidden until the checkpoint is frozen.

For canonical MCF:

- forget train/repair records per seed: `50`
- benchmark retain examples used during optimization: `0`
- held-out paraphrases used during optimization: `0`
- neighborhood prompts used during optimization: `0`
- PPL text used during optimization: `0`
- sensitive answer field: `target_new`

The MCF direct success constraint is

```text
NLL(target_new) - NLL(target_true) >= 0.25
```

Because lower NLL means higher model preference, a positive margin means the unwanted `target_new` continuation is sufficiently disfavored relative to the factual `target_true` continuation.

## 3. Model matrices and dimensions

For Llama-3.2-3B-Instruct in this implementation:

```text
vocabulary size V = 128,256
hidden size     d = 3,072
```

The input embedding and LM head begin tied, so there is one shared vocabulary-facing weight matrix

```text
W0 ∈ R^(V × d)
   = R^(128256 × 3072)
```

with

```text
128256 × 3072 = 394,002,432 parameters.
```

The observed full model parameter count is `3,212,749,824`, so the tied vocabulary matrix is approximately `12.263713%` of the full model.

All transformer blocks are frozen in Stage 1 and Stage 2.

### Why the matrix has this shape

Every vocabulary token needs one `d=3072`-dimensional vector. Therefore there are `V=128256` rows and `d=3072` columns.

- row index `i`: vocabulary token ID `i`
- column index `j`: hidden/embedding coordinate `j`

The same matrix is used initially for input embeddings and output logits because the model uses tied word embeddings.

## 4. Stage 1 — common GA/GD vocabulary edit

Entrypoint:

```text
scripts/sure_stage1_gagd.py
```

Stage 1 is identical in optimization mechanics for MCF and ZsRE. The benchmark adapter only tells the implementation what answer is sensitive. For MCF, the sensitive answer is `target_new`.

### 4.1 Teacher-forced sensitive PredictionCases

Each of the 50 direct MCF requests has a sensitive answer string. Tokenize that answer using the model tokenizer. If the answer contains `T_i` evaluated tokens, it creates `T_i` teacher-forced next-token PredictionCases.

For seed `s`, let

```text
N_s = sum_i T_i
```

be the number of Stage-1 PredictionCases. This is not a manually chosen hyperparameter; it is determined by the tokenizer and the 50 sampled sensitive answers.

For each case, the model receives the direct prompt plus the already-known sensitive answer prefix and predicts the next sensitive token.

### 4.2 Base-logit teacher cache

Before any training, the base model is evaluated on the same direct PredictionCases and the final-position logits are cached.

The conceptual cache matrix is

```text
L_base ∈ R^(N_s × V)
       = R^(N_s × 128256).
```

The implementation caches it in FP32 on CPU. `N_s` is data/tokenization-dependent; `V=128256` is fixed by the model.

### 4.3 Gradient-ascent forgetting term

For a sensitive token `y`, Stage 1 minimizes

```text
L_GA = mean(log p_theta(y | x, y_<t)).
```

Because `log p_theta(y)` is non-positive, minimizing it drives it downward and suppresses the sensitive token probability.

For MCF, `y` is a token from `target_new`.

### 4.4 Distribution-preservation term

To avoid indiscriminately changing the rest of the vocabulary distribution, Stage 1 also computes

```text
L_GD = KL(p_base(-y) || p_theta(-y)).
```

The sensitive token `y` is removed from both distributions, the remaining `V-1` entries are renormalized, and the current distribution is matched to the frozen base distribution.

Per PredictionCase, the KL therefore acts conceptually on vectors of length

```text
V - 1 = 128255.
```

### 4.5 Stage-1 objective and hyperparameters

```text
L_stage1 = 2.0 * L_GA + 1.0 * L_GD
```

Canonical values:

| Hyperparameter | Value |
|---|---:|
| Steps | `600` |
| PredictionCase batch size | `1` |
| Base-cache batch size | `8` |
| Learning rate | `1e-4` |
| Optimizer | `AdamW` |
| Weight decay | `0` |
| Gradient clipping | `1.0` |
| GA weight | `2.0` |
| GD weight | `1.0` |
| Training dtype | `BF16` |
| Trainable transformer parameters | `0` |
| Trainable tied vocabulary parameters | `394,002,432` |

### 4.6 Stage-1 raw change matrix

Let the matrix after gradient optimization but before restoration be `W_train`.

Define

```text
DeltaW_train = W_train - W0
```

with full shape

```text
DeltaW_train ∈ R^(128256 × 3072).
```

During optimization, gradients can flow through the entire tied vocabulary matrix, so `DeltaW_train` can in principle contain changes in many rows.

## 5. Stage-1 restoration and the retained change matrix

The canonical method does **not** keep all Stage-1 changes.

Let `S` be the set of unique token IDs appearing in the sensitive MCF `target_new` answers for the 50 direct forget records.

The method:

1. snapshots `W0` before Stage 1;
2. finishes GA/GD training;
3. saves the trained rows indexed by `S`;
4. resets the complete matrix to `W0`;
5. writes back only the trained rows in `S`.

The retained Stage-1 change is therefore

```text
DeltaW_stage1 ∈ R^(128256 × 3072)
```

with

```text
DeltaW_stage1[i, :] = W_train[i, :] - W0[i, :]   if i ∈ S
DeltaW_stage1[i, :] = 0                          otherwise.
```

Thus the full matrix is large, but its **row support is sparse**. The number of changed rows is

```text
|S| = number of unique sensitive target_new token IDs
```

and is determined from the tokenizer output for that seed; it is not hard-coded.

The earlier benchmark-specific MCF factual-row multiplier (`1.25 × base`) is not used in this canonical architecture.

After restoration,

```text
W_stage1 = W0 + DeltaW_stage1.
```

At this point input embeddings and the LM head are still tied.

## 6. Stage 2 — common sparse residual repair

Entrypoint:

```text
scripts/sure_stage2_sparse_repair.py
```

Stage 2 is used only for direct residual failures left after Stage 1.

### 6.1 Untying

The LM head is cloned from the restored Stage-1 tied matrix, then untied.

After this operation:

```text
E_stage2_start     = W0 + DeltaW_stage1
Wout_stage2_start  = W0 + DeltaW_stage1
```

but they are now independent tensors.

The transformer and input embedding are frozen. Only a sparse parameterization of selected output rows is optimized.

### 6.2 Active direct failures

For each direct MCF record compute

```text
m = NLL(target_new) - NLL(target_true).
```

A record is active if

```text
m < 0.25.
```

Only active direct records are used to determine Stage-2 selected rows and hidden directions.

### 6.3 Selected rows

Let `R` denote the set of sensitive `target_new` token IDs appearing in active residual records.

The Stage-2 delta is allowed to modify only these rows.

If there are `|R| = R_s` selected rows, the row-restricted change has shape

```text
DeltaW_R ∈ R^(R_s × 3072).
```

The corresponding full vocabulary change matrix is conceptually

```text
DeltaW_stage2 ∈ R^(128256 × 3072),
```

but all rows outside `R` are exactly zero.

### 6.4 Hidden matrix and SVD basis

The direct active cases produce final-layer hidden vectors of width `3072`. Because MCF's constraint compares `target_new` and `target_true`, the repair basis is built from hidden states associated with both sides of the active direct margin computation.

Collect these vectors into

```text
H ∈ R^(M × 3072),
```

where `M` is determined by the number of active direct token positions.

Compute

```text
H = U Sigma V^T.
```

For requested rank `r`, use the first `r` numerically valid right-singular directions:

```text
B_r ∈ R^(r_actual × 3072).
```

where

```text
r_actual = min(requested rank, numerical rank(H)).
```

This is why the actual rank can be smaller than the requested rank.

### 6.5 Low-rank change matrix

For a positive candidate rank, the trainable coefficient matrix is

```text
C ∈ R^(R_s × r_actual).
```

The selected-row change is

```text
DeltaW_R = C B_r
```

with shape

```text
(R_s × r_actual) @ (r_actual × 3072)
    -> R_s × 3072.
```

Therefore the number of trainable repair parameters is only

```text
R_s * r_actual,
```

not `R_s * 3072`.

Each individual change-matrix entry is

```text
DeltaW_R[i,j] = sum_k C[i,k] * B_r[k,j].
```

The basis `B_r` is fixed; gradient descent learns only `C`.

### 6.6 Unrestricted candidate

Candidate rank `0` means unrestricted selected-row repair. In that case the trainable parameter is directly

```text
D ∈ R^(R_s × 3072),
```

so each selected output-row coordinate can change independently.

The full matrix is still sparse by row because only `R_s` rows can be nonzero.

### 6.7 Candidate ranks and optimization hyperparameters

The shared candidate order is

```text
2 -> 8 -> 0
```

where `0` means unrestricted.

Canonical Stage-2 hyperparameters:

| Hyperparameter | Value |
|---|---:|
| Candidate ranks | `2, 8, 0` |
| Maximum steps per candidate | `800` |
| Learning rate | `0.005` |
| Optimizer | `AdamW` |
| Weight decay | `0` |
| L2 coefficient | `1e-6` |
| Batch size | `8` |
| Direct check interval | `25` steps |
| MCF direct margin | `0.25` |

The repair loss is a direct-margin hinge objective plus the small L2 penalty on the selected-row delta.

Rank selection is direct-only:

1. test rank 2;
2. if direct failures remain, test rank 8;
3. if failures remain, test unrestricted rank 0;
4. use the first ordered candidate with zero direct failures;
5. if no candidate reaches zero, minimize `(direct_failures, candidate_order, delta_norm)`.

No paraphrases, neighborhood probes, retain examples, or PPL values participate in rank selection.

## 7. Direct-only scale sweep

After choosing the repair candidate, its magnitude is scaled by a shared candidate list:

```text
1,
0.875,
0.75,
0.625,
0.5,
0.375,
0.25,
0.1875,
0.125,
0.09375,
0.0625,
0.046875,
0.03125,
0.015625,
0.0078125,
0
```

Let the selected scale be `alpha`.

The method chooses the **smallest scale preserving zero direct failures**. If no scale yields zero failures, it minimizes `(direct_failures, scale)`.

Again, scale selection uses direct training-visible constraints only.

## 8. Final matrices and the full change from base

The final input embedding is

```text
E_final = W0 + DeltaW_stage1.
```

The final output head is

```text
Wout_final = W0 + DeltaW_stage1 + alpha * DeltaW_stage2.
```

Both `E_final` and `Wout_final` have shape

```text
128256 × 3072.
```

After Stage 2 they are no longer tied.

Therefore the final changes relative to the base model are:

```text
DeltaE_final    = DeltaW_stage1
DeltaWout_final = DeltaW_stage1 + alpha * DeltaW_stage2.
```

For a positive rank candidate, a selected Stage-2 row satisfies

```text
DeltaW_stage2[i,j]
    = sum_k C[i,k] B_r[k,j]
```

before multiplication by `alpha`.

For an unrestricted candidate,

```text
DeltaW_stage2[i,j] = D[i,j]
```

on selected rows.

Every non-selected Stage-2 row is exactly zero.

## 9. How every matrix size is determined

| Matrix | Shape | How its size is determined |
|---|---|---|
| Base tied vocabulary matrix `W0` | `128256 × 3072` | model vocabulary size × model hidden size |
| Base-logit cache `L_base` | `N_s × 128256` | teacher-forced sensitive-token cases × vocabulary size |
| Stage-1 full change `DeltaW_stage1` | `128256 × 3072` | same shape as vocabulary matrix; only sensitive rows remain nonzero after restoration |
| Active hidden matrix `H` | `M × 3072` | number of active direct hidden vectors × hidden size |
| SVD basis `B_r` | `r_actual × 3072` | numerical rank limited by requested candidate rank |
| Coefficients `C` | `R_s × r_actual` | selected sensitive output rows × actual repair rank |
| Low-rank selected-row delta | `R_s × 3072` | `C @ B_r` |
| Unrestricted selected-row delta `D` | `R_s × 3072` | selected sensitive rows × hidden size |
| Full Stage-2 change `DeltaW_stage2` | `128256 × 3072` | row-scattered selected-row delta; zero elsewhere |

`N_s`, `M`, `R_s`, and `r_actual` vary by seed because they are determined by tokenization and which direct constraints remain active. `V=128256` and `d=3072` are fixed by the model.

## 10. PPL protocol

Both MCF and ZsRE canonical runs use the same tracked PPL fixture:

```text
data/wikidata
```

The evaluator:

1. loads the dataset with `datasets.load_from_disk`;
2. joins `train['text'][:20]` with spaces;
3. tokenizes the joined text;
4. truncates to 100 input tokens;
5. computes autoregressive perplexity.

The identical base-model PPL across MCF and ZsRE (`11.0625`) is the cross-benchmark sanity check that the same base model and PPL fixture are being used.

## 11. Interpretation of this canonical MCF result

The canonical shared method achieves complete direct forgetting on average:

```text
Eff: 13.6000 -> 0.0000
```

and substantially reduces held-out paraphrase success of the unwanted answer:

```text
Gen: 14.8000 -> 4.1000.
```

PPL changes only modestly:

```text
11.0625 -> 11.1937.
```

However, MCF specificity is lower in this architecture-aligned run:

```text
Spe: 11.4620 -> 9.6170.
```

This should be reported as-is. The purpose of this record is to archive the **shared MCF/ZsRE architecture**, not to overwrite the fact that an older benchmark-specific MCF variant achieved a stronger MCF specificity score.

## 12. Reproducibility pointers

Canonical implementation files:

```text
scripts/build_mcf_sure_canonical_split.py
scripts/sure_canonical_core.py
scripts/sure_stage1_gagd.py
scripts/sure_stage2_sparse_repair.py
scripts/run_mcf_sure_canonical.sh
scripts/mcf_zero_unlearn_official_eval.py
scripts/annotate_ppl_provenance.py
scripts/aggregate_sure_canonical.py
```

Run roots used for this registered aggregate:

```text
outputs/mcf_base_canonical_final_seeds1_10_20260818
outputs/mcf_sure_canonical_final_seeds1_10_20260818
```

The supplied terminal aggregate is authoritative for the four values registered in this file. Exact per-seed row counts/ranks are intentionally not invented here; they are recoverable from each seed's `stage1_gagd/vocabulary_restoration.json` and `stage2_sparse_row/repair_summary.json` artifacts.
