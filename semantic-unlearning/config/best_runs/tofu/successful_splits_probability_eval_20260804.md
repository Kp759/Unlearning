# TOFU successful stored checkpoints — fresh probability-ratio evaluation

Evaluation date: **2026-08-04**  
Seed: **42**  
Reference model: `outputs/finetuned_model_3B_instruct`

These are fresh checkpoint rescoring results for the deterministic local probability-ratio protocol. They are not the full ECO-style TOFU truth-ratio/ROUGE evaluation.

| Split | Forget probability mean ↓ | Forget probability max ↓ | Retain ratio ↑ | World facts ratio ↑ | Real authors ratio ↑ | Full retain ratio ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Forget01 | 9.6829042562e-09 | 1.0623305258e-07 | 0.9997282624 | 0.9962049127 | 0.9947001934 | 0.9997276559 |
| Forget05 | 1.8976859792e-05 | 2.5230229949e-04 | 0.9995309114 | 0.9987162352 | 0.9994698763 | 0.9995319671 |
| Forget10 | 6.6001062805e-05 | 4.6693140757e-04 | 0.9996438622 | 0.9892992377 | 1.0024807453 | 0.9996450014 |

## Checkpoints and exact configs

### Forget01

- Checkpoint: `outputs/tofu_forget01_setting3_restore_repair_seed42/lm_head_repair_unique100_overlap000/checkpoint`
- Config: `outputs/tofu_forget01_setting3_restore_repair_seed42/lm_head_repair_unique100_overlap000/config_used.json`
- Fresh result: `outputs/tofu_successful_splits_eval_20260804/forget01/probability_ratio_eval.json`

### Forget05

- Checkpoint: `outputs/tofu_setting3_5e_ultra_seed42/lm_head_repair_alpha000/checkpoint`
- Config: `outputs/tofu_setting3_5e_ultra_seed42/lm_head_repair_alpha000/config_used.json`
- Fresh result: `outputs/tofu_successful_splits_eval_20260804/forget05/probability_ratio_eval.json`

### Forget10

- Checkpoint: `outputs/tofu_forget10_setting3_5e_repair_seed42/lm_head_repair_unique050_overlap000/checkpoint`
- Config: `outputs/tofu_forget10_setting3_5e_repair_seed42/lm_head_repair_unique050_overlap000/config_used.json`
- Fresh result: `outputs/tofu_successful_splits_eval_20260804/forget10/probability_ratio_eval.json`

## Storage

The model weights are not stored as ordinary Git blobs because each `model.safetensors` file is approximately 6.8 GiB. Use `scripts/publish_tofu_successful_checkpoints_release.sh` to publish the exact configs, evaluation artifacts, SHA-256 manifests, tokenizer files, and split checkpoint weights under the GitHub Release tag `tofu-successful-splits-seed42`.

The released `config_used.json` files and checkpoint SHA-256 hashes are authoritative.
