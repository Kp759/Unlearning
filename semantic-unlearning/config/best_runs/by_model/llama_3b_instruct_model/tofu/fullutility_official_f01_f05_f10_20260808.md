# TOFU full-utility F01/F05/F10 — Llama-3B project record

Record date: **2026-08-08**  
Seed: **42**  
Base architecture: `meta-llama/Llama-3.2-3B-Instruct`  
TOFU unlearning target: Full-TOFU-fine-tuned checkpoint selected at epoch 5, LR `4e-5`

## Scientific gates

- Forget answer probability target: **<= 0.0003**
- Final materialized utility probability ratio: **>= 0.999800**
- Utility constraint mode: `aggregate`
- Utility reference policy: `reference`
- Full utility calibration: complete complementary retain split + 100 real-authors + 117 world-facts examples
- BF16 materialization

## Final checkpoint registry

| Split | Forget / retain sizes | Final checkpoint | Local repair | Full evaluator |
|---|---:|---|---|---|
| F01 / `forget01` | 40 / 3960 | `outputs/tofu_final_fullutility_v2/f01/active_repair/checkpoint` | **PASS** | **PASS** |
| F05 / `forget05` | 200 / 3800 | `outputs/tofu_final_fullutility_v2/f05/active_repair/checkpoint` | **PASS** | **PASS** |
| F10 / `forget10` | 400 / 3600 | `outputs/tofu_final_fullutility_f10_final/active_repair/checkpoint` | **PASS** | **PASS** |

## Full-evaluator results

| Split | Forget AP ↓ | Forget Truth Ratio ↓ | Forget ROUGE-L ↓ | Retain AP ↑ | Retain Truth Ratio ↑ | Retain ROUGE-L ↑ |
|---|---:|---:|---:|---:|---:|---:|
| F01 | **0.000074** | **0.370591** | **0.021873** | **0.988549** | **0.545852** | **0.989718** |
| F05 | **6.4741423e-05** | **0.415082079** | **0.043354469** | **0.988577754** | **0.549220516** | **0.990388557** |
| F10 | **6.4906246e-05** | **0.389179362** | **0.038005263** | **0.988521509** | **0.553359810** | **0.990001361** |

## External utility

| Split | Real authors norm. AP ↑ | Real authors TR ↑ | World facts norm. AP ↑ | World facts TR ↑ |
|---|---:|---:|---:|---:|
| F01 | **0.725967** | **0.868330** | **0.715407** | **0.882108** |
| F05 | **0.733031048** | **0.874651004** | **0.716988901** | **0.882123681** |
| F10 | **0.737271352** | **0.877209421** | **0.717929427** | **0.883249411** |

## F01 local materialized repair gate

- Best step: `564`
- Active forget cases after repair: `0`
- Forget AP mean: `7.384817581623793e-05`
- Forget AP max: `2.353683958062902e-04`
- Retain probability ratio: `0.9998026490211487`
- Real-authors probability ratio: `0.9998013973236084`
- World-facts probability ratio: `0.9999599456787109`
- Full-retain probability ratio: `0.9998023358370192`

## Reporting constraints

1. F01/F05/F10 now have homogeneous project-local `tofu_eval.py` results.
2. Do **not** describe forget Truth Ratio as near zero; the final values are about `0.371`, `0.415`, and `0.389`.
3. Do not mix local probability ratios (`~0.9998`) with absolute evaluator answer probabilities (`~0.9885`).
4. Benchmark-official TOFU Forget Quality / KS still requires the corresponding oracle or retain-only reference comparison.
5. TOFU uses Llama-3.2-3B-Instruct as the base architecture, but the unlearning target is the Full-TOFU-fine-tuned model rather than the untouched base checkpoint.

Canonical machine-readable record: `config/best_runs/tofu/fullutility_official_f01_f05_f10_20260808.json`.
Canonical Markdown record: `config/best_runs/tofu/fullutility_official_f01_f05_f10_20260808.md`.

## Weights and hashes

Model weights remain on Wulver. Preserve the final checkpoints and capture SHA-256 hashes plus exact per-run configs before storage cleanup.
