# Llama 3B Instruct Model — benchmark result index

Model family: `meta-llama/Llama-3.2-3B-Instruct`

This directory is the model-centric snapshot for the latest MCF, ZsRE, and TOFU results used by this project. The canonical dataset-specific records remain under `config/best_runs/{mcf,zsre,tofu}/`; files here are organized copies/summary pointers for cross-model comparisons.

## Dataset status

| Dataset | Record | Status | Important note |
|---|---|---|---|
| MCF | `mcf/zerounlearn_locked_forget_only_rank2_seeds1_10_20260810.json` | **Current best fair ZeroUnlearn-style locked-probe result** | Seeds 1–10; Stage 1/2 use 50 forget records and 0 MCF retain records, with only `requested_rewrite` visible. Final Eff = **0.0000 ± 0.0000**, held-out Gen = **4.0000 ± 3.6332**, Spe = **27.7110 ± 3.6742**, PPL = **11.5500 ± 0.6771** (population SD). 500/500 direct prompts pass and 960/1000 held-out paraphrases pass. The old `official_protected_v2_seeds0_9` record remains for provenance but is evaluator-conditioned. |
| ZsRE | `zsre/setting5e_active_repair_u1p20_ppl1p16_cal384_seeds1_10.json` | Best accepted 10-seed result | 10/10 configured gates pass; selected forget Eff/Gen = 0. Checkpoint weight hashes remain to be captured. |
| TOFU | `tofu/fullutility_official_f01_f05_f10_20260808.json` | F01/F05/F10 project-local full evaluator PASS | TOFU starts from the Full-TOFU-finetuned Llama-3.2-3B-Instruct target. Benchmark-official Forget Quality/KS still requires oracle/retain-only comparison. |

## MCF current-best protocol note

For the current MCF record:

- **training/unlearning:** `50 forget + 0 MCF retain`
- **visible forget prompt:** `requested_rewrite` only
- **held out until final evaluation:** official paraphrases and neighborhood prompts
- **final evaluation:** same 50 forget facts + 1000 evaluation-only MCF retain records
- **Stage-2 sparse LM-head repair:** rank `2`, no retain KL, no retain calibration, no retain hidden projection
- **Wulver job:** `1171704`, all 10 array tasks completed with exit code `0:0`

This means the reported `Gen=4.0` is a genuine prompt-level holdout result: **same facts, unseen formulations**. The 1000 MCF retain examples do not protect the model during training.

A full 10-seed base-vs-unlearned retain-delta analysis and checkpoint SHA-256 capture are still pending. Absolute PPL should not yet be claimed as directly better than the ZeroUnlearn paper until evaluator/base-PPL reproduction is aligned.

## Future model folders

Use sibling directories under `config/best_runs/by_model/` for each architecture/model family. The intended next folders are:

- `llama_8b_instruct_model/`
- `qwen_model/`

Keep the same dataset subfolder layout (`mcf/`, `zsre/`, `tofu/`) so cross-model result collection stays consistent.
