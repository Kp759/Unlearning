# ZsRE — current ZeroUnlearn-matched no-neutral aggressive GA/GD result

**Model:** `meta-llama/Llama-3.2-3B-Instruct`  
**Seeds:** 1–10  
**Run:** AWS 10-seed aggressive no-neutral GA/GD + sparse sensitive-row LM-head repair  
**Status:** current best AWS 10-seed ZsRE record; Wulver replication was still running when archived  
**Machine-readable record:** `zerounlearn_locked_no_neutral_gagd_aggressive_seeds1_10_20260813.json`

This file describes **only this current run**. It does not merge hyperparameters or metrics from the older ZsRE Setting-5e active-repair record.

## 1. Headline result

All values are mean ± **sample standard deviation (`n-1`)** over seeds 1–10.

| Stage | F-Eff ↓ | F-Gen ↓ | F-Spe ↑ | R-Eff ↑ | R-Gen ↑ | R-Spe ↑ | PPL ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Stage 1 post-hoc | 1.769 ± 1.606 | 2.159 ± 1.638 | 26.155 ± 2.365 | 27.861 ± 1.670 | 27.073 ± 1.501 | 26.664 ± 1.104 | 18.650 ± 2.204 |
| **Stage 2 / Final** | **0.025 ± 0.079** | **0.627 ± 0.510** | **26.158 ± 2.367** | **27.740 ± 1.795** | **26.949 ± 1.638** | **26.643 ± 1.111** | **18.650 ± 2.204** |

Exact final `F-Eff=0` occurred in **9/10 seeds**. Therefore the correct manuscript statement is the aggregate `0.025 ± 0.079`, not “10/10 seeds have zero Eff.”

Stage 2 reduces the mean residual forget efficacy by about **98.59%** (`1.769 → 0.025`) and held-out generalization accuracy by about **70.96%** (`2.159 → 0.627`) while leaving mean PPL exactly unchanged (`18.650 → 18.650`). Mean changes in retain metrics are small: `R-Eff -0.121`, `R-Gen -0.124`, and `R-Spe -0.021`.

## 2. Exact ZeroUnlearn data split

Source dataset: `data/zsre_mend_eval.json`.

The observed source has **19,086 records**. The official ZeroUnlearn-compatible sampling is:

```text
half = len(data) // 2 = 9,543

retain pool = data[0:9543]        -> 9,543 records
forget pool = data[9543:19086]    -> 9,543 records
```

For every seed in `1..10`, one `random.Random(seed)` is created and the draws happen in this order:

1. sample **50 forget records** from the second-half forget pool;
2. sample **1000 retain records** from the first-half retain pool.

The forget draw happens first. Therefore Stage 1/2 and final evaluation use the same 50 forget facts for a given seed even though the method sees zero benchmark-retain records before final evaluation.

The split is built by `scripts/build_zsre_zerounlearn_locked_no_neutral_split.py` and checked by `scripts/verify_zsre_zerounlearn_locked_split.py`.

### Training-visible artifact

For the 50 sampled forget records, the method-visible copy contains only:

- the direct `requested_rewrite` prompt;
- the subject;
- the original sensitive `target_true` answer.

It explicitly strips/omits:

- `target_new`;
- `Unknown`/IDK/replacement targets;
- rephrases;
- locality/neighborhood probes;
- retain records;
- PPL text.

This is a **prompt-level holdout**: the same 50 underlying forget facts are used as deletion requests, but their benchmark rephrases and locality probes are held out until the final checkpoint is frozen.

## 3. Data visibility by stage

| Data | Stage 1 | Stage 2 | Final eval |
|---|---:|---:|---:|
| 50 direct forget rewrites | yes | yes | yes |
| original sensitive answer tokens | yes | yes | yes |
| forget rephrases | **no** | **no** | yes |
| forget locality/neighborhood | **no** | **no** | yes |
| 1000 benchmark retain records | **no** | **no** | yes |
| Wikidata PPL text | **no** | **no** | yes |
| `target_new` | **no** | **no** | evaluator legacy metadata only |
| Unknown/IDK/replacement target | **no** | **no** | not used by SURE |

The ZeroUnlearn-compatible evaluator still constructs legacy `target_new="Unknown"` metadata when adapting raw ZsRE rows. This is **not used by SURE training, repair, selection, or scoring**. Eff/Gen score whether the model still predicts the original sensitive `target_true` tokens.

## 4. Stage 1 — aggressive Emb+LM GA/GD

Entrypoint: `scripts/zsre_no_neutral_stage1_gagd.py`.

### 4.1 Unit of training

Each of the 50 direct forget records is expanded into teacher-forced next-token `PredictionCase`s for every token of its original answer. If an answer has `T` evaluated tokens, it contributes `T` cases with progressively longer target prefixes.

Therefore the number of Stage-1 cases is **seed-dependent**:

```text
N_sensitive_prediction_cases
= sum over 50 forget records of evaluated target_true token length
```

The exact per-seed value is stored in each Stage-1 config as `visible_rewrite_total_tokens`.

### 4.2 Frozen base teacher cache

Before any parameter update, the base model logits are cached on the same direct forget PredictionCases only.

```text
cache shape = [N_sensitive_prediction_cases, 128256]
cache dtype = FP32
cache device = CPU
```

No retain/rephrase/locality/PPL input is used to build the teacher cache.

### 4.3 Trainable matrix

Llama-3.2-3B-Instruct uses:

```text
vocabulary size = 128,256
hidden size     = 3,072
```

The input embedding and LM head are tied during Stage 1, so there is **one unique shared trainable matrix**:

```text
W_vocab = 128256 × 3072
        = 394,002,432 values
```

Observed total model parameters: `3,212,749,824`.

Thus Stage 1 trains:

```text
394,002,432 / 3,212,749,824
= 12.263713% of model parameters
```

Raw BF16 storage for the shared matrix is `788,004,864 bytes` ≈ `751.5 MiB`.

Parameter scope:

```text
Transformer blocks     frozen
Input embeddings       trainable
LM head                trainable
Input/LM weights       tied
Unique trainable tensor 1
```

Do not double-count embedding and LM-head parameters: they are the same storage during Stage 1.

### 4.4 GA objective

For each teacher-forced original sensitive token `y`:

```text
L_GA = mean(log p_theta(y | x, y_<t))
```

The total objective is minimized, so minimizing `log p(y)` pushes the original sensitive token probability downward. Equivalently, it performs ascent on that token's NLL.

### 4.5 GD objective

For each same direct forget decision, remove the sensitive token from both the frozen base distribution and current distribution, renormalize the remaining vocabulary, then minimize:

```text
L_GD = KL(p_base(-y) || p_theta(-y))
```

Because vocabulary size is `128256`, each row's GD distribution contains **128255 non-sensitive vocabulary entries**.

This GD is **not benchmark-retain GD**. It preserves the base model's non-sensitive alternatives on the same 50 direct forget prompts.

### 4.6 Total Stage-1 loss and frozen aggressive hyperparameters

```text
L_stage1 = 2.0 * L_GA + 1.0 * L_GD

steps              = 600
batch size          = 1 PredictionCase
base-cache batch    = 8
learning rate       = 1e-4
GA weight           = 2.0
GD weight           = 1.0
optimizer           = AdamW
weight decay        = 0
gradient clip       = 1.0
dtype               = BF16
device map          = single GPU
```

### 4.7 Vocabulary restoration

A base snapshot of the tied vocabulary matrix is taken before training.

After 600 updates:

1. collect the unique vocabulary row IDs corresponding to all sensitive `target_true` tokens in that seed;
2. save the trained values of those sensitive rows;
3. restore the **entire shared matrix** to its base snapshot;
4. write back only the trained sensitive rows.

So after Stage 1:

```text
sensitive answer-token rows  -> keep trained displacement
all other 128256-row entries -> exactly base
```

The number of sensitive rows is seed-dependent and stored in each seed's `vocabulary_restoration.json`.

## 5. Stage 2 — sparse sensitive-row LM-head repair

Entrypoint: `scripts/zsre_no_neutral_active_sensitive_row_repair.py`.

Stage 2 runs **before any held-out evaluation**.

### 5.1 Model scope

The Stage-1 checkpoint has tied input/output weights. Stage 2 clones and unties the output LM head, then freezes the model.

```text
full LM-head shape          = 128256 × 3072
full LM-head entries        = 394,002,432
transformer trainables      = 0
input embedding trainables  = 0
input embeddings modified   = false
```

Only an auxiliary sparse output-row delta is trainable.

### 5.2 Active residual definition

Stage 2 re-expands only the 50 direct rewrites into sensitive token PredictionCases.

A case is **active** if, after reloading Stage 1:

```text
argmax(logits) == original sensitive target token
```

The selected LM-head rows are the unique sensitive target token IDs among these active cases.

### 5.3 Repair matrix size

This current ZsRE repair is **not low-rank**, **not SVD repair**, and **not LoRA**.

For `R` selected sensitive rows it directly learns:

```text
DeltaW_selected shape = R × 3072
trainable parameters  = R * 3072
parameter dtype       = FP32
```

Across the ten AWS seeds:

```text
selected rows mean = 4.400 ± 4.142
```

Therefore the mean number of directly trainable Stage-2 coordinates is:

```text
4.4 × 3072 = 13,516.8
```

This is only about **0.00343064%** of the `128256 × 3072` full LM-head coordinate count on average.

### 5.4 Stage-2 loss

For each active sensitive token:

```text
L_sensitive = ReLU(
    z_sensitive
    - stopgrad(max_{j != sensitive} z_j)
    + 0.05
)
```

with:

```text
L_total = L_sensitive + 1e-6 * mean(DeltaW^2)
```

The goal is simply to make the original sensitive token lose the top-1 decision by a small margin. There is no replacement answer.

### 5.5 Repair hyperparameters

```text
max steps           = 800
learning rate       = 0.005
optimizer           = AdamW
weight decay        = 0
margin              = 0.05
L2 coefficient      = 1e-6
batch size          = 8
gradient clip       = 1.0
correctness checks  = step 1, every 25 steps, final
stop condition      = zero direct sensitive-token correct decisions
```

### 5.6 Candidate scale sweep

After optimization, the best learned full delta is tested at scales:

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

Selection uses **direct requested_rewrite cases only**.

If one or more scales produce zero direct sensitive-token correct decisions, choose the **smallest** such scale. Otherwise minimize `(direct_correct_count, scale)` lexicographically.

No rephrase, locality, retain, or PPL value enters this selection.

## 6. Stage-2 aggregate repair diagnostics

All valid values below are mean ± sample SD across the ten AWS seeds:

| Quantity | Mean ± SD |
|---|---:|
| active residual token decisions before repair | 4.400 ± 4.142 |
| selected sensitive LM-head rows | 4.400 ± 4.142 |
| best optimization step | 5.700 ± 10.177 |
| selected delta scale | 0.303 ± 0.204 |
| unscaled full delta Frobenius norm | 1.344 ± 1.983 |

### `active_after` warning

Do **not** use the originally printed:

```text
active_after = 0.000 ± 0.471
```

The aggregator used `-1` when an older no-op Stage-2 summary omitted `active_after`, and then averaged that sentinel as if it were a real count. This bug affects only that diagnostic aggregate, not the official Stage-1 or final metrics.

The authoritative final direct-forgetting statistic is `F-Eff = 0.025 ± 0.079`, with exact zero in 9/10 seeds.

## 7. Final evaluation

Only after Stage 2 has saved/frozen its checkpoint does `scripts/zsre_zero_unlearn_official_eval.py` reopen the unchanged source dataset and sample:

```text
50 same forget facts
1000 evaluation-only retain facts
```

For each adapted ZsRE record:

- rewrite: evaluate every token of the original answer under teacher-forced prefixes;
- paraphrase: evaluate the same original-answer tokens on the held-out rephrase;
- neighborhood/locality: evaluate the provided locality-answer token-prefix decisions.

Metrics use ZeroUnlearn-compatible per-record macro averaging and 0–100 scaling:

- **F-Eff ↓** — direct rewrite accuracy of the original sensitive answer tokens on forget records;
- **F-Gen ↓** — held-out rephrase accuracy of those same sensitive tokens;
- **F-Spe ↑** — locality/neighborhood token accuracy for forget records;
- **R-Eff/R-Gen/R-Spe** — the corresponding metrics over the 1000 sampled retain records;
- **PPL ↓/stable** — ZeroUnlearn-compatible Wikidata perplexity.

PPL uses the first 20 Wikidata train texts joined together, truncates to a maximum tokenized input length of 100, and stores the same scalar under `forget_PPL` and `retain_PPL`.

## 8. Protocol audit

The completed 10-seed aggregate reports:

```text
all_Unknown_used_false     = True
all_IDK_used_false         = True
all_target_new_seen_false  = True
all_retain_seen_zero       = True
all_rephrases_seen_zero    = True
all_locality_seen_zero     = True
all_PPL_seen_false         = True
```

Thus no held-out utility/generalization metric is used for training or checkpoint/scale selection.

## 9. Seed-10 terminal excerpt

Seed 10, Stage 1 post-hoc:

```text
F-Eff  0.571429
F-Gen  1.071429
F-Spe 24.888586
R-Eff 27.568532
R-Gen 27.370357
R-Spe 27.873671
PPL   19.125
```

Seed 10, final:

```text
F-Eff  0.000000
F-Gen  0.500000
F-Spe 24.888586
R-Eff 27.568532
R-Gen 27.370357
R-Spe 27.873671
PPL   19.125
```

This seed illustrates the intended Stage-2 behavior: residual forgetting is cleaned up while specificity, retain metrics, and PPL remain unchanged.

## 10. AWS artifacts and checkpoint policy

Run root:

```text
outputs/aws_zsre_sure_no_neutral_gagd_aggressive_seeds1_10_3b
```

Aggregate outputs:

```text
aggregate_10seeds.csv
aggregate_10seeds.json
aggregate_10seeds.md
```

After each seed successfully produced both final and Stage-1 post-hoc evaluations, multi-GB model checkpoint directories were intentionally deleted. JSON configs, split manifests, repair summaries, logs, and evaluation results were retained.

Because checkpoints were deleted intentionally, checkpoint SHA-256 hashes are not available for this AWS record.

## 11. Code provenance

Current-run code paths:

```text
scripts/build_zsre_zerounlearn_locked_no_neutral_split.py
scripts/verify_zsre_zerounlearn_locked_split.py
scripts/zsre_no_neutral_stage1_gagd.py
scripts/zsre_no_neutral_stage1_emb_lm.py
scripts/zsre_no_neutral_active_sensitive_row_repair.py
scripts/zsre_zero_unlearn_official_eval.py
scripts/run_zsre_sure_no_neutral_gagd.sh
scripts/run_zsre_sure_no_neutral_gagd_strong.sh
scripts/run_zsre_sure_no_neutral_gagd_aggressive.sh
scripts/aggregate_zsre_sure_gagd_10seeds.py
```

## 12. Scientific interpretation

The current result supports the following implementation-level description:

> SURE performs aggressive no-neutral GA/GD on the tied embedding/output vocabulary matrix using only the direct deletion requests. GA suppresses the original sensitive answer token, while same-prompt GD distills the frozen base distribution over every non-sensitive vocabulary item. Post-training restoration resets all non-sensitive vocabulary rows. A second stage then unties the LM head and directly optimizes only the residual sensitive output rows, using no benchmark-retain or held-out probes. Across ten seeds this yields final ZsRE F-Eff `0.025 ± 0.079` and held-out F-Gen `0.627 ± 0.510`, with unchanged mean PPL between Stage 1 and Stage 2.
