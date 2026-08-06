# ZsRE Setting 5e + Active LM-Head Repair

## Result

All configured metric and utility gates passed for all ten seeds.

- Seeds: 1–10
- Accepted: 10/10
- Rejected: 0/10
- Selected repair scale: 1.0 for every seed
- Forget Eff: 0.0000 ± 0.0000
- Forget Gen: 0.0000 ± 0.0000
- Retain calibration records: 384
- Utility-drop tolerance: 1.20 percentage points
- Maximum PPL ratio: 1.16
- Target Eff: 0.0
- Target Gen: 0.0

Uncertainty is the sample standard deviation across ten seeds (`ddof=1`).

| Method | Forget Eff ↓ | Forget Gen ↓ | Forget Spe ↑ | Retain Eff ↑ | Retain Gen ↑ | Retain Spe ↑ | PPL ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Setting 5e | 20.8133 ± 3.1632 | 20.8110 ± 3.7952 | 27.7231 ± 2.8844 | 32.8411 ± 1.0764 | 31.8887 ± 0.7420 | 28.4971 ± 0.9018 | 11.9250 ± 0.5942 |
| Selected active repair | **0.0000 ± 0.0000** | **0.0000 ± 0.0000** | 27.7231 ± 2.8844 | 32.2089 ± 1.0834 | 31.2619 ± 0.7399 | 28.2136 ± 0.8199 | 12.3250 ± 0.9717 |

## Per-seed selected results

| Seed | Accepted | Scale | Forget Eff | Forget Gen | PPL |
|---:|:---:|---:|---:|---:|---:|
| 1 | yes | 1.0 | 0.0 | 0.0 | 11.6250 |
| 2 | yes | 1.0 | 0.0 | 0.0 | 11.0625 |
| 3 | yes | 1.0 | 0.0 | 0.0 | 11.8125 |
| 4 | yes | 1.0 | 0.0 | 0.0 | 11.4375 |
| 5 | yes | 1.0 | 0.0 | 0.0 | 13.3750 |
| 6 | yes | 1.0 | 0.0 | 0.0 | 12.0000 |
| 7 | yes | 1.0 | 0.0 | 0.0 | 12.5625 |
| 8 | yes | 1.0 | 0.0 | 0.0 | 12.9375 |
| 9 | yes | 1.0 | 0.0 | 0.0 | 12.1875 |
| 10 | yes | 1.0 | 0.0 | 0.0 | 14.2500 |

## Model and checkpoints

- Base model: `meta-llama/Llama-3.2-3B-Instruct`
- Repository commit used for the run: `a8d6b85450d662d28d31ee2f0f01e1c3b4be706e`
- Python: 3.10.20
- Hardware: NVIDIA A100-SXM4-80GB
- Slurm job: 1166025

The model weights remain in persistent Wulver storage. The JSON manifest records the exact source and output roots, per-seed checkpoint path patterns, configuration, aggregate metrics, and the metadata needed to recreate the procedure for another compatible model.

### Checkpoint path patterns

```text
Setting 5e source:
/scratch/yl258/kp759/Unlearning/semantic-unlearning/outputs/zsre_from_scratch_relaxed_u1p0_ppl1p10_cal384_seeds1_10/seed{seed}/setting5e/checkpoint

Active candidate:
/scratch/yl258/kp759/Unlearning/semantic-unlearning/outputs/zsre_repair_only_exploratory_u1p20_ppl1p16_cal384_seeds1_10/seed{seed}/active_candidate_checkpoint

Selected checkpoint:
/scratch/yl258/kp759/Unlearning/semantic-unlearning/outputs/zsre_repair_only_exploratory_u1p20_ppl1p16_cal384_seeds1_10/seed{seed}/selected_checkpoint
```

## Recreation command

Replace `SEED`, `SETTING5_CHECKPOINT`, and `OUTPUT_DIR` for each model and seed.

```bash
python scripts/zsre_bf16_safe_active_repair_v2.py \
  --setting5-checkpoint "$SETTING5_CHECKPOINT" \
  --output-dir "$OUTPUT_DIR" \
  --zsre-path data/zsre_mend_eval.json \
  --wikidata-dir data/wikidata \
  --seed "$SEED" \
  --forget-num 50 \
  --retain-num 1000 \
  --repair-steps 800 \
  --repair-lr 0.005 \
  --repair-optimizer adamw \
  --active-logit-margin 0.25 \
  --selection-logit-margin 0.05 \
  --repair-rank 0 \
  --repair-l2-lambda 1e-6 \
  --retain-calibration-num 384 \
  --retain-calibration-seed 1729 \
  --project-away-protected-hidden \
  --candidate-scales "1.0,0.875,0.75,0.625,0.5,0.375,0.25,0.1875,0.125,0.09375,0.0625,0.046875,0.03125,0.015625,0.0078125,0.0" \
  --utility-drop-tolerance 1.20 \
  --max-ppl-ratio 1.16 \
  --target-eff-max 0.0 \
  --target-gen-max 0.0 \
  --strict-utility-gates \
  --eval-batch-size 8 \
  --cache-batch-size 4 \
  --dtype bf16 \
  --device-map single \
  --save-candidate-checkpoint \
  --save-selected-checkpoint \
  --no-fail-if-target-missed
```

For another architecture, verify that its tokenizer, input embeddings, and LM-head vocabulary dimensions agree, and regenerate that model's own Setting 5e utility baseline before evaluating the gates.
