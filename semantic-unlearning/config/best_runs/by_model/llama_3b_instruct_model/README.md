# Llama 3B Instruct Model — benchmark result index

Model family: `meta-llama/Llama-3.2-3B-Instruct`

This directory is the model-centric snapshot for the latest MCF, ZsRE, and TOFU results used by this project. The canonical dataset-specific records remain under `config/best_runs/{mcf,zsre,tofu}/`; files here are organized copies/summary pointers for cross-model comparisons.

## Dataset status

| Dataset | Record | Status | Important note |
|---|---|---|---|
| MCF | `mcf/official_protected_v2_seeds0_9.json` | Official-compatible 10-seed evaluation recorded | Eff/Gen = 0 on all 10 seeds; strict post-reload gate passes 9/10 because seed 7 margin is 0.09375 < 0.10. Exact training config and checkpoint hashes remain to be captured. |
| ZsRE | `zsre/setting5e_active_repair_u1p20_ppl1p16_cal384_seeds1_10.json` | Best accepted 10-seed result | 10/10 configured gates pass; selected forget Eff/Gen = 0. Checkpoint weight hashes remain to be captured. |
| TOFU | `tofu/fullutility_official_f01_f05_f10_20260808.json` | F01/F05/F10 project-local full evaluator PASS | TOFU starts from the Full-TOFU-finetuned Llama-3.2-3B-Instruct target. Benchmark-official Forget Quality/KS still requires oracle/retain-only comparison. |

## Future model folders

Use sibling directories under `config/best_runs/by_model/` for each architecture/model family. The intended next folders are:

- `llama_8b_instruct_model/`
- `qwen_model/`

Keep the same dataset subfolder layout (`mcf/`, `zsre/`, `tofu/`) so cross-model result collection stays consistent.
