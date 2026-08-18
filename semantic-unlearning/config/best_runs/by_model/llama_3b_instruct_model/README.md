# Llama 3B Instruct Model — benchmark result index

Model family: `meta-llama/Llama-3.2-3B-Instruct`

This directory is the model-centric snapshot for the latest MCF, ZsRE, and TOFU results used by this project. MCF and ZsRE now point to the **canonical shared SURE-LM architecture** run completed on 2026-08-18. Historical benchmark-specific snapshots remain in their dataset folders for provenance.

## Current canonical main-table results

All values below are **mean ± sample standard deviation across seeds 1–10**. The main paper table intentionally reports only `Eff. ↓`, `Gen. ↓`, `Spe. ↑`, and `PPL ↓`.

| Dataset | Method | Eff. ↓ | Gen. ↓ | Spe. ↑ | PPL ↓ |
|---|---|---:|---:|---:|---:|
| MCF | Base | **13.6000 ± 4.5019** | **14.8000 ± 2.3476** | **11.4620 ± 2.0742** | **11.0625 ± 0.0000** |
| MCF | Canonical SURE-LM | **0.0000 ± 0.0000** | **4.1000 ± 4.3830** | **9.6170 ± 2.3648** | **11.1937 ± 0.1779** |
| ZsRE | Base | **33.0896 ± 4.4061** | **32.1979 ± 4.4302** | **28.1210 ± 2.7368** | **11.0625 ± 0.0000** |
| ZsRE | Canonical SURE-LM | **0.1317 ± 0.2342** | **0.6617 ± 0.5272** | **26.1068 ± 2.1947** | **11.9750 ± 1.4482** |

## Current dataset records

| Dataset | Canonical record | Status |
|---|---|---|
| MCF | `mcf/canonical_sure_shared_arch_seeds1_10_20260818.json` | **Current architecture-aligned MCF run**. Detailed matrix/Stage-1/Stage-2 specification is in the matching `.md`. |
| ZsRE | `zsre/canonical_sure_shared_arch_seeds1_10_20260818.json` | **Current architecture-aligned ZsRE run**. Detailed matrix/Stage-1/Stage-2 specification is in the matching `.md`. |
| TOFU | `tofu/fullutility_official_f01_f05_f10_20260808.json` | F01/F05/F10 project-local full evaluator record. |

## Shared canonical MCF/ZsRE architecture

The canonical cross-benchmark pipeline is:

```text
locked direct-only per-seed split
  -> shared teacher-forced GA/GD Stage 1
  -> restore full vocabulary matrix to base and reapply only sensitive rows
  -> detect residual direct failures
  -> untie LM head
  -> shared sparse sensitive-output-row Stage 2
  -> direct-only rank selection
  -> direct-only scale selection
  -> freeze checkpoint
  -> held-out evaluation and canonical PPL
```

Shared model dimensions:

```text
V = 128,256 vocabulary rows
d = 3,072 hidden dimensions
W0 shape = 128256 x 3072
W0 parameter count = 394,002,432
```

Shared Stage-1 hyperparameters:

```text
steps = 600
batch = 1 PredictionCase
base-cache batch = 8
lr = 1e-4
optimizer = AdamW
weight decay = 0
grad clip = 1.0
GA weight = 2.0
GD weight = 1.0
transformer = frozen
trainable matrix = tied embedding/LM-head vocabulary matrix
```

Shared Stage-2 hyperparameters:

```text
candidate ranks = 2, 8, 0
0 = unrestricted selected-row delta
max steps = 800
lr = 0.005
optimizer = AdamW
weight decay = 0
L2 = 1e-6
batch = 8
check interval = 25
candidate scales = 1, .875, .75, .625, .5, .375, .25, .1875,
                   .125, .09375, .0625, .046875, .03125, .015625,
                   .0078125, 0
```

The benchmark adapter differs only in sensitive semantics and the direct constraint:

- MCF sensitive answer: `target_new`; direct constraint `NLL(target_new)-NLL(target_true) >= 0.25`.
- ZsRE sensitive answer: original `target_true`; no neutral target; direct constraint puts the sensitive token at least `0.05` below the best non-sensitive logit.

The detailed dataset-specific Markdown records explain the complete matrix algebra, including `DeltaW_stage1`, the SVD basis, low-rank coefficient matrix, unrestricted fallback, scale factor, and the final untied output-head change.

## PPL alignment

Both MCF and ZsRE canonical results use:

```text
data/wikidata
```

The dataset is loaded with `datasets.load_from_disk`; `train['text'][:20]` is joined with spaces and tokenized with truncation to 100 input tokens. Both canonical base evaluations produce `PPL = 11.0625`, confirming a matched base-model/PPL fixture.

## Historical records

Do not delete the older MCF/ZsRE records. They capture earlier benchmark-specific methods and are important for provenance.

In particular:

- the historical MCF rank-2 record can be numerically stronger on `Spe`, but it uses the older benchmark-specific Stage-1/restoration/repair path;
- the historical ZsRE aggressive GA/GD record uses the previous ZsRE-specific Stage-2 parameterization and historical PPL setup.

The 2026-08-18 canonical records should be used when the paper claims that MCF and ZsRE follow the **same SURE-LM architecture**.

## Future model folders

Use sibling directories under `config/best_runs/by_model/` for each model family, keeping the same dataset subfolder layout (`mcf/`, `zsre/`, `tofu/`) so cross-model result collection remains consistent.
