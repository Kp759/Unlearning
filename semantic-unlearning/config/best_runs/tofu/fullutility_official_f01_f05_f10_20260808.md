# TOFU full-utility F01/F05/F10 — project official record

Record date: **2026-08-08**  
Seed: **42**  
Starting/reference checkpoint: `outputs/tofu_full_utility_sweep_v7/lr4e-5_epochs6_slurm/checkpoint_epoch_5`  
Fine-tuned checkpoint selection: epoch 5, LR `4e-5`

This file is the authoritative project record for the new full-utility TOFU runs. It supersedes the older `successful_splits_probability_eval_20260804.*` files for the checkpoints listed here, but does not delete or rewrite those historical records.

## Scientific gates

- Forget answer probability target: **<= 0.0003**
- Final materialized utility probability ratio: **>= 0.999800**
- Utility constraint mode: `aggregate`
- Utility reference policy: `reference`
- Full utility calibration is used: the entire complementary retain split plus all 100 real-authors examples and all 117 world-facts examples.
- Checkpoint must be materially saved only after the final BF16 gate is satisfied.

## Final checkpoint registry

| Split | Forget / retain sizes | Final checkpoint | Local repair status | Full evaluator status |
|---|---:|---|---|---|
| F01 / `forget01` | 40 / 3960 | `outputs/tofu_final_fullutility_v2/f01/active_repair/checkpoint` | **PASS** | **PASS** |
| F05 / `forget05` | 200 / 3800 | `outputs/tofu_final_fullutility_v2/f05/active_repair/checkpoint` | **PASS** | **PASS** |
| F10 / `forget10` | 400 / 3600 | `outputs/tofu_final_fullutility_f10_final/active_repair/checkpoint` | **PASS** | **PASS** |

## Materialized active-repair gate

These are local answer-probability-ratio / reference-utility checks on the actual repaired candidates. They are not the same quantities as the absolute metrics from `tofu_eval.py`.

| Split | Active forget after ↓ | Forget AP mean ↓ | Forget AP max ↓ | Retain ratio ↑ | Real-authors ratio ↑ | World-facts ratio ↑ | Full-retain ratio ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| F01 | **0** | **7.3848176e-05** | **2.3536840e-04** | **0.9998026490** | **0.9998013973** | **0.9999599457** | **0.9998023358** |
| F05 | **0** | **6.3872641e-05** | **2.3592392e-04** | **0.9998035431** | **0.9998438954** | **0.9998476505** | **0.9998029340** |
| F10 | **0** | Passed final materialized gate | **<= 3.0e-04** | Passed | Passed | Passed | **>= 0.999800** |

For F10, the final job independently gated the materialized BF16 checkpoint and then ran the full evaluator. Exact local materialized values should be copied from `outputs/tofu_final_fullutility_f10_final/active_repair/repair_summary.json` if a row-level archival dump is required; the checkpoint was accepted only after all configured final gates passed.

## Project-official full-evaluator table

All three final checkpoints have now been evaluated with the same project-local `tofu_eval.py` pipeline. F01 values below are the six-decimal values printed by the completed evaluator; the exact machine values remain in the local summary JSON.

| Split | Forget AP baseline ↓ | Forget AP final ↓ | Forget Truth Ratio baseline ↓ | Forget Truth Ratio final ↓ | Forget ROUGE-L baseline ↓ | Forget ROUGE-L final ↓ | Retain AP baseline ↑ | Retain AP final ↑ | Retain AP ratio ↑ | Retain Truth Ratio final ↑ | Retain ROUGE-L final ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| F01 | 0.991081 | **0.000074** | 0.493958 | **0.370591** | 0.999091 | **0.021873** | 0.988739 | **0.988549** | **~0.999808** | **0.545852** | **0.989718** |
| F05 | 0.989575 | **6.4741423e-05** | 0.462503 | **0.415082079** | 0.987769 | **0.043354469** | 0.988720 | **0.988577754** | **~0.999856** | **0.549220516** | **0.990388557** |
| F10 | 0.989239909 | **6.4906246e-05** | 0.462889826 | **0.389179362** | 0.987487212 | **0.038005263** | 0.988709772 | **0.988521509** | **0.99980958699** | **0.553359810** | **0.990001361** |

### External utility from the full evaluator

| Split | Real authors norm. AP baseline ↑ | Final ↑ | Real authors Truth Ratio baseline ↑ | Final ↑ | World facts norm. AP baseline ↑ | Final ↑ | World facts Truth Ratio baseline ↑ | Final ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| F01 | 0.722977 | **0.725967** | 0.864981 | **0.868330** | 0.714694 | **0.715407** | 0.880273 | **0.882108** |
| F05 | 0.722977 | **0.733031048** | 0.864981 | **0.874651004** | 0.714694 | **0.716988901** | 0.880273 | **0.882123681** |
| F10 | 0.722976848 | **0.737271352** | 0.864980804 | **0.877209421** | 0.714693619 | **0.717929427** | 0.880273285 | **0.883249411** |

## F01 full-evaluator artifact

- Checkpoint: `outputs/tofu_final_fullutility_v2/f01/active_repair/checkpoint`
- Repair summary: `outputs/tofu_final_fullutility_v2/f01/active_repair/repair_summary.json`
- Evaluation: `outputs/tofu_final_fullutility_v2/f01/eval_final/fullutility_final_f01_summary.json`
- Forget AP: `0.000074`
- Forget Truth Ratio: `0.370591`
- Forget ROUGE-L recall: `0.021873`
- Retain AP: `0.988549`
- Retain AP / baseline retain AP: approximately `0.999807836` from the six-decimal console values
- Retain Truth Ratio: `0.545852`
- Retain ROUGE-L recall: `0.989718`
- Real authors normalized AP: `0.725967`
- Real authors Truth Ratio: `0.868330`
- World facts normalized AP: `0.715407`
- World facts Truth Ratio: `0.882108`

## F05 full-evaluator artifact

- Checkpoint: `outputs/tofu_final_fullutility_v2/f05/active_repair/checkpoint`
- Repair summary: `outputs/tofu_final_fullutility_v2/f05/active_repair/repair_summary.json`
- Evaluation: `outputs/tofu_final_fullutility_v2/f05/eval_final/fullutility_final_f05_summary.json`
- Forget AP: `6.474142263271326e-05`
- Forget Truth Ratio: `0.41508207853604207`
- Forget ROUGE-L recall: `0.04335446835522295`
- Retain AP: `0.9885777537693319`
- Retain Truth Ratio: `0.5492205159013261`
- Retain ROUGE-L recall: `0.9903885571197951`
- Real authors normalized AP: `0.7330310477491317`
- Real authors Truth Ratio: `0.8746510031718766`
- World facts normalized AP: `0.7169889014923924`
- World facts Truth Ratio: `0.8821236806025088`

## F10 full-evaluator artifact

- Checkpoint: `outputs/tofu_final_fullutility_f10_final/active_repair/checkpoint`
- Repair summary: `outputs/tofu_final_fullutility_f10_final/active_repair/repair_summary.json`
- Evaluation: `outputs/tofu_final_fullutility_f10_final/eval_final/fullutility_final_f10_summary.json`
- Combined report: `outputs/tofu_final_fullutility_f10_final/final_report.json`
- Forget AP: `6.49062456e-05`
- Forget Truth Ratio: `0.389179362`
- Forget ROUGE-L recall: `0.0380052629`
- Retain AP: `0.988521509`
- Retain AP / baseline retain AP: `0.9998095869889359`
- Retain Truth Ratio: `0.55335981`
- Retain ROUGE-L recall: `0.990001361`
- Real authors normalized AP: `0.737271352`
- Real authors Truth Ratio: `0.877209421`
- World facts normalized AP: `0.717929427`
- World facts Truth Ratio: `0.883249411`

## F01 materialized repair artifact

- Checkpoint: `outputs/tofu_final_fullutility_v2/f01/active_repair/checkpoint`
- Repair summary: `outputs/tofu_final_fullutility_v2/f01/active_repair/repair_summary.json`
- Best step: `564`
- Active forget cases after repair: `0`
- Forget AP mean: `7.384817581623793e-05`
- Forget AP max: `0.0002353683958062902`
- Retain ratio: `0.9998026490211487`
- Real authors ratio: `0.9998013973236084`
- World facts ratio: `0.9999599456787109`
- Full-retain ratio: `0.9998023358370192`
- Full evaluator: **PASS**.

## Interpretation and reporting constraints

1. F01, F05, and F10 all show near-zero direct answer probability and very low generation overlap on their forget splits while preserving retain behavior.
2. Do **not** describe Truth Ratio as near zero. Final forget Truth Ratios are approximately `0.371`, `0.415`, and `0.389` for F01/F05/F10 respectively.
3. Do not mix the local reference probability ratios (`~0.9998`) with the absolute full-evaluator retain answer probabilities (`~0.9885`).
4. A single homogeneous project-local F01/F05/F10 full-evaluation table may now be reported because all three splits have completed the same evaluator.
5. This is the project's local full-TOFU evaluation pipeline. A paper claim of benchmark-official TOFU Forget Quality / KS p-value still requires the corresponding oracle/retain-only comparison if not already present.

## Weights and hashes

Model weight files are not committed as ordinary Git blobs. Preserve the Wulver checkpoints above and capture SHA-256 hashes plus the exact per-run configs before any storage cleanup. Those hashes should be added to this record when captured.
