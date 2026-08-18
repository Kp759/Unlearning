# ZsRE — Canonical SURE-LM shared architecture, seeds 1–10

**Model:** `meta-llama/Llama-3.2-3B-Instruct`  
**Date:** 2026-08-18  
**Status:** canonical cross-benchmark architecture run  
**Dataset:** ZsRE  
**Seeds:** 1–10  
**Main-table metrics only:** `Eff. ↓`, `Gen. ↓`, `Spe. ↑`, `PPL ↓`

This record archives the ZsRE result produced by the same canonical SURE-LM implementation used for MCF. The optimization scope, Stage-1 objective, restoration operator, Stage-2 sparse-row parameterization, candidate-rank policy, scale sweep, PPL fixture, and aggregation convention are shared across the two datasets. The benchmark adapter changes only the meaning of the sensitive answer and the direct success condition.

## 1. Final 10-seed result

All values are **mean ± sample standard deviation** across seeds 1–10.

| Method | Eff. ↓ | Gen. ↓ | Spe. ↑ | PPL ↓ |
|---|---:|---:|---:|---:|
| Base | **33.0896 ± 4.4061** | **32.1979 ± 4.4302** | **28.1210 ± 2.7368** | **11.0625 ± 0.0000** |
| Canonical SURE-LM | **0.1317 ± 0.2342** | **0.6617 ± 0.5272** | **26.1068 ± 2.1947** | **11.9750 ± 1.4482** |

Metric directions:

- `Eff. ↓`: lower means the original sensitive answer is less often produced on the direct rewrite prompts.
- `Gen. ↓`: lower means the same sensitive answer is less often produced on held-out rephrases.
- `Spe. ↑`: higher means better preservation on locality/neighborhood decisions.
- `PPL ↓`: lower/stable indicates better general fluency.

Only these four metrics are intended for the main paper table.

## 2. Locked no-neutral data protocol

Each seed follows the ZeroUnlearn-compatible pool split:

1. use the second half of ZsRE as the forget pool;
2. use the first half as the retain pool;
3. sample `50` forget records and reserve `1000` retain records;
4. materialize only the 50 direct forget requests into `training_visible_forget.json`;
5. remove neutral/`Unknown` targets from the training-visible artifact;
6. keep rephrases, locality/neighborhood prompts, retain evaluation, and PPL unavailable to Stage 1 and Stage 2.

Canonical ZsRE therefore uses:

- forget train/repair records per seed: `50`
- benchmark retain examples used during optimization: `0`
- held-out rephrases used during optimization: `0`
- locality prompts used during optimization: `0`
- PPL text used during optimization: `0`
- neutral target: **not used**
- sensitive answer field: original `target_true`

For each sensitive next-token PredictionCase, the Stage-2 direct constraint is

```text
z_sensitive <= max(z_non_sensitive) - 0.05.
```

Equivalently, the sensitive token must lose the top-1 decision with a `0.05` logit margin.

## 3. Model matrices and dimensions

The canonical Llama-3.2-3B-Instruct model has

```text
vocabulary size V = 128,256
hidden size     d = 3,072.
```

At initialization, input embeddings and the LM head are tied through one shared vocabulary matrix

```text
W0 ∈ R^(128256 × 3072).
```

Its parameter count is

```text
128256 × 3072 = 394,002,432.
```

The observed total model parameter count is `3,212,749,824`, so Stage 1 exposes approximately `12.263713%` of total model parameters while keeping all transformer blocks frozen.

### Interpretation of rows and columns

- each of the `128256` rows corresponds to one vocabulary token ID;
- each row contains `3072` embedding/output coordinates;
- the same row initially represents the token in both the input embedding and output classifier because the model is tied.

## 4. Stage 1 — shared GA/GD vocabulary editing

Entrypoint:

```text
scripts/sure_stage1_gagd.py
```

Stage 1 uses the same algorithm as canonical MCF. For ZsRE, the sensitive sequence is the original `target_true` answer rather than MCF's `target_new`.

### 4.1 Teacher-forced PredictionCases

For each of the 50 direct forget requests, tokenize the original sensitive answer. Every evaluated answer token becomes a teacher-forced next-token case.

If record `i` contributes `T_i` sensitive tokens, then for seed `s`

```text
N_s = sum_i T_i.
```

`N_s` is data dependent. It is determined by the sampled records and tokenizer; it is not selected manually.

The evaluated prompt for token `t` includes the direct question/rewrite prompt and the teacher-forced sensitive prefix `y_<t`.

### 4.2 Base teacher cache

Before optimization, the frozen base model computes final-position logits for all `N_s` direct sensitive cases.

Conceptually,

```text
L_base ∈ R^(N_s × 128256).
```

The implementation stores these logits in FP32 on CPU. They are the teacher distribution for the GD preservation term.

### 4.3 GA term

For each sensitive token `y`, compute

```text
L_GA = mean(log p_theta(y | x, y_<t)).
```

The training loop minimizes this quantity. Because the log probability is non-positive, minimization makes the sensitive token less probable.

In ZsRE, `y` is a token from the original `target_true` answer.

### 4.4 GD term

The model should change the sensitive token while preserving the rest of the local output distribution. The sensitive token index is removed from both base and current vocab distributions, the remaining values are renormalized, and the KL divergence is computed:

```text
L_GD = KL(p_base(-y) || p_theta(-y)).
```

Each renormalized vector has conceptual length

```text
V - 1 = 128255.
```

This teacher uses the same direct training-visible prompt; no retain, rephrase, locality, or held-out example is needed.

### 4.5 Stage-1 objective

```text
L_stage1 = 2.0 * L_GA + 1.0 * L_GD.
```

Canonical hyperparameters:

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
| Transformer trainable | `no` |
| Tied vocabulary matrix trainable | `yes` |
| Trainable vocabulary parameters | `394,002,432` |

### 4.6 Raw Stage-1 matrix update

Let `W_train` be the vocabulary matrix after the 600 optimization steps but before restoration.

Define

```text
DeltaW_train = W_train - W0.
```

It has full shape

```text
DeltaW_train ∈ R^(128256 × 3072).
```

Because the tied matrix participates in every output logit, optimization may temporarily move many rows. Canonical restoration removes those broad changes.

## 5. Sensitive-row-only restoration

Let `S` be the union of unique token IDs appearing in the 50 direct sensitive `target_true` answers.

After Stage 1:

1. preserve the trained rows indexed by `S`;
2. reset the entire matrix to the exact base snapshot `W0`;
3. reinsert only the preserved sensitive rows.

The retained Stage-1 change matrix is

```text
DeltaW_stage1 ∈ R^(128256 × 3072)
```

with

```text
DeltaW_stage1[i,:] = W_train[i,:] - W0[i,:]  if i ∈ S
DeltaW_stage1[i,:] = 0                         otherwise.
```

Therefore all non-sensitive rows are exactly restored to the base model.

The number of Stage-1 changed rows is

```text
|S| = number of unique target_true token IDs
```

for that seed. It is tokenizer/data dependent rather than a fixed hyperparameter.

After restoration,

```text
W_stage1 = W0 + DeltaW_stage1.
```

The input embedding and LM head remain tied at the end of Stage 1.

## 6. Stage 2 — sparse sensitive output-row repair

Entrypoint:

```text
scripts/sure_stage2_sparse_repair.py
```

### 6.1 Untying the output head

The restored Stage-1 LM head is cloned into a separate output matrix. The transformer and input embeddings are frozen.

At the start of Stage 2:

```text
E_stage2_start    = W0 + DeltaW_stage1
Wout_stage2_start = W0 + DeltaW_stage1.
```

They begin numerically equal but no longer share storage.

### 6.2 Residual direct failures

Expand the 50 direct sensitive answers into the same type of token-level PredictionCases used by the official evaluator.

For each case, let

```text
z_s = sensitive-token logit
z_o = largest non-sensitive logit.
```

The desired condition is

```text
z_s <= z_o - 0.05.
```

A residual case that does not satisfy the required direct condition becomes active for Stage 2.

### 6.3 Selected output rows

Take the union of sensitive token IDs among the active residual PredictionCases. Call this set `R` and its size `R_s`.

Only these output-head rows may change.

The selected-row delta has shape

```text
DeltaW_R ∈ R^(R_s × 3072).
```

The equivalent full change matrix has shape

```text
DeltaW_stage2 ∈ R^(128256 × 3072)
```

but rows outside `R` are exactly zero.

### 6.4 Active hidden matrix

For each active direct sensitive PredictionCase, collect the final-layer hidden representation at the decision position.

If there are `M` active cases,

```text
H ∈ R^(M × 3072).
```

Unlike MCF's sequence-margin adapter, ZsRE's active hidden matrix consists directly of the sensitive token decision states because the official direct constraint is token-level.

### 6.5 SVD basis

Compute

```text
H = U Sigma V^T.
```

For requested repair rank `r`, the implementation keeps the first numerically valid right-singular vectors:

```text
B_r ∈ R^(r_actual × 3072),
```

where

```text
r_actual = min(requested rank, numerical rank(H)).
```

Consequently a requested rank `8` may have an actual rank less than `8` if the active hidden states span fewer independent directions.

### 6.6 Low-rank repair matrix

For a positive candidate rank, learn coefficients

```text
C ∈ R^(R_s × r_actual).
```

The selected-row change is

```text
DeltaW_R = C B_r.
```

Shapes:

```text
(R_s × r_actual) @ (r_actual × 3072)
     -> R_s × 3072.
```

Only `C` is trainable; `B_r` stays fixed.

The number of trainable Stage-2 parameters is therefore

```text
R_s * r_actual.
```

For any selected row `i` and hidden coordinate `j`, the matrix entry is

```text
DeltaW_R[i,j] = sum_k C[i,k] * B_r[k,j].
```

This gives an explicit interpretation of every element in the Stage-2 change matrix.

### 6.7 Unrestricted rank-0 candidate

Candidate rank `0` is the fallback full selected-row parameterization:

```text
D ∈ R^(R_s × 3072).
```

Here every coordinate of every selected row can be learned independently.

This is unrestricted **within selected rows**, not unrestricted over the full vocabulary matrix.

### 6.8 Stage-2 loss

For a batch of active sensitive cases, define

```text
L_margin = mean(ReLU(z_sensitive - z_best_other + 0.05)).
```

The regularized repair objective is

```text
L_repair = L_margin + 1e-6 * mean(DeltaW_R^2).
```

The margin term pushes the sensitive token below the best alternative with the required safety margin. The L2 term discourages unnecessary row movement.

### 6.9 Repair hyperparameters

| Hyperparameter | Value |
|---|---:|
| Candidate ranks | `2, 8, 0` |
| Rank `0` meaning | unrestricted selected rows |
| Maximum steps per candidate | `800` |
| Learning rate | `0.005` |
| Optimizer | `AdamW` |
| Weight decay | `0` |
| L2 coefficient | `1e-6` |
| Batch size | `8` |
| Check interval | `25` steps |
| ZsRE direct logit margin | `0.05` |

Candidate ranks are evaluated in increasing model complexity:

```text
2 -> 8 -> unrestricted.
```

The first candidate reaching zero direct failures is sufficient. If none reaches zero, selection minimizes

```text
(direct_failures, candidate_order, delta_norm).
```

No held-out rephrase, locality, retain, or PPL metric enters this decision.

## 7. Direct-only scale sweep

After choosing the repair parameterization, evaluate the following multiplicative scales:

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
0.
```

Let the selected value be `alpha`.

Selection rule:

```text
choose the smallest alpha with zero direct failures;
if no alpha reaches zero, minimize (direct_failures, alpha).
```

This is still a training-visible direct-only decision.

## 8. Full final change matrices

The final input embedding is

```text
E_final = W0 + DeltaW_stage1.
```

The final output matrix is

```text
Wout_final = W0 + DeltaW_stage1 + alpha * DeltaW_stage2.
```

Both matrices have shape

```text
128256 × 3072.
```

They are no longer tied after Stage 2.

Relative to the base model:

```text
DeltaE_final    = DeltaW_stage1
DeltaWout_final = DeltaW_stage1 + alpha * DeltaW_stage2.
```

For a low-rank candidate on a selected row,

```text
DeltaWout_final[i,j]
 = DeltaW_stage1[i,j]
 + alpha * sum_k C[i,k] B_r[k,j].
```

For an unrestricted candidate,

```text
DeltaWout_final[i,j]
 = DeltaW_stage1[i,j]
 + alpha * D[i,j].
```

For a non-selected Stage-2 row, the Stage-2 contribution is zero.

## 9. How each matrix size/value is determined

| Matrix | Shape | Determination |
|---|---|---|
| Base tied matrix `W0` | `128256 × 3072` | model vocabulary size × hidden size |
| Base-logit cache `L_base` | `N_s × 128256` | teacher-forced sensitive cases × vocab size |
| Stage-1 sparse full change `DeltaW_stage1` | `128256 × 3072` | same model shape; nonzero only on unique sensitive target_true rows |
| Active hidden matrix `H` | `M × 3072` | number of residual sensitive token cases × hidden size |
| SVD basis `B_r` | `r_actual × 3072` | SVD numerical rank, capped by requested rank |
| Coefficient matrix `C` | `R_s × r_actual` | selected sensitive rows × actual rank |
| Low-rank selected-row change | `R_s × 3072` | matrix product `C B_r` |
| Unrestricted selected-row change `D` | `R_s × 3072` | selected sensitive rows × hidden size |
| Full Stage-2 change `DeltaW_stage2` | `128256 × 3072` | scatter selected-row delta into vocabulary rows, zero elsewhere |

The varying quantities are:

- `N_s`: total sensitive teacher-forced token decisions;
- `M`: residual active token decisions after Stage 1;
- `R_s`: number of distinct sensitive token rows among those residuals;
- `r_actual`: numerical rank of the active hidden-state matrix, capped by requested rank.

Those values are determined independently for each seed from the locked direct data and the model state. They should be read from each run artifact rather than replaced by a single hard-coded number.

## 10. PPL protocol shared with MCF

Canonical ZsRE uses exactly

```text
data/wikidata
```

for PPL, matching canonical MCF.

The evaluator:

1. loads the tracked dataset using `datasets.load_from_disk`;
2. joins `train['text'][:20]` with spaces;
3. tokenizes the result;
4. truncates to 100 input tokens;
5. computes autoregressive PPL.

The base PPL is

```text
MCF Base PPL  = 11.0625
ZsRE Base PPL = 11.0625,
```

which verifies that the same base checkpoint and canonical PPL fixture are being compared across the two datasets.

Historical ZsRE values from `data/wikidata_aws_diag` are not part of this canonical comparison.

## 11. Interpretation of the canonical ZsRE result

The shared method produces very strong forgetting:

```text
Eff: 33.0896 -> 0.1317
Gen: 32.1979 -> 0.6617.
```

Specificity decreases moderately:

```text
Spe: 28.1210 -> 26.1068.
```

PPL increases from

```text
11.0625 -> 11.9750.
```

Thus the canonical ZsRE result demonstrates that the shared GA/GD + restoration + sparse residual repair architecture can strongly suppress both direct and rephrased sensitive-answer behavior, with measurable but bounded specificity/fluency cost in this 10-seed aggregate.

## 12. Reproducibility pointers

Canonical implementation files:

```text
scripts/build_zsre_zerounlearn_locked_no_neutral_split.py
scripts/sure_canonical_core.py
scripts/sure_stage1_gagd.py
scripts/sure_stage2_sparse_repair.py
scripts/run_zsre_sure_canonical.sh
scripts/zsre_zero_unlearn_official_eval.py
scripts/annotate_ppl_provenance.py
scripts/aggregate_sure_canonical.py
```

Registered output roots:

```text
outputs/zsre_base_canonical_final_seeds1_10_20260818
outputs/zsre_sure_canonical_final_seeds1_10_20260818
```

The supplied terminal aggregate is authoritative for the four registered paper metrics. Exact per-seed sensitive-row counts, active-case counts, actual ranks, selected scales, and delta norms are intentionally not fabricated in this summary; they are available from the seed-specific `vocabulary_restoration.json` and `repair_summary.json` artifacts.
