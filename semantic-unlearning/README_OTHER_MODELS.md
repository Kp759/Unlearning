# Paper table on other models (Llama-3.1-8B-Instruct, Qwen3-4B)

Same driver (`scripts/run_l16_paper_one.sh`), same evaluators, same baselines as the
Llama-3.2-3B table. Only the model and its layers change; all of that lives in
`scripts/paper_model_config.sh`.

## Run on Wulver (login node)

```bash
cd /scratch/yl258/kp759/Unlearning && git pull --ff-only && cd semantic-unlearning

# 1. weights (Llama is gated: accept the licence on huggingface.co, `hf auth login` once)
bash scripts/submit_paper_model.sh llama31_8b fetch
bash scripts/submit_paper_model.sh qwen3_4b fetch

# 2. smoke test: MCF seed 1, base | ours | genie + GA/GD/NPO/RMU/ZeroUnlearn
bash scripts/submit_paper_model.sh llama31_8b smoke
bash scripts/submit_paper_model.sh qwen3_4b smoke

# 3. after the smoke rows look sane: 4 datasets x 5 seeds (finished stages are skipped)
bash scripts/submit_paper_model.sh llama31_8b full
bash scripts/submit_paper_model.sh qwen3_4b full

python scripts/l16_paper_table.py --root outputs/paper_llama31_8b_v1
python scripts/l16_paper_table.py --root outputs/paper_qwen3_4b_v1
```

## Settings

| | Llama-3.2-3B (paper) | Llama-3.1-8B-Instruct | Qwen3-4B |
|---|---|---|---|
| blocks | 28 | 32 | 36 |
| router layer | 19 | 22 | 24 |
| write layer (ours) | 16 | 18 | 21 |
| logit-lens layer | 24 | 27 | 31 |
| v2.2 subject / relation candidates | 6 / 19 23 24 27 | 7 / 22 26 27 31 | 8 / 24 30 31 35 |
| RMU layer | 7 | 8 | 9 |
| GA/GD/NPO trained weights | all MLPs | **down_proj only** | all MLPs |
| ZeroUnlearn hparams (upstream) | Llama-3.2-3B-Instruct.json | Llama-3.1-8B-Instruct.json | Qwen3-4B.json |

- Layers default to the 3B choices at the same relative depth. `ROUTER_LAYER`, `EARLY_WRITE_LAYER`,
  `WRITE_LAYER` and `LENS_LAYER` in the environment override them; measure them first (next section).

## Choosing the layers (smoke test on MCF seed 1, before the full run)

The paper's configuration on the 3B model is the router at block 19 and Δe written at block 16
(`outputs/l16_ours_v1`). On another model both blocks are measured, not assumed:

```bash
# 1. router layer: fit the ours-recipe router at the default block, cache the last-token state of
#    every block for the router's eval / null / perturbed / train / aug prompts, refit the same
#    classifier at every block (one GPU task, ~1 h on the 8B model)
bash scripts/submit_layer_smoke.sh qwen3_4b router
bash scripts/submit_layer_smoke.sh llama31_8b router
python scripts/layer_smoke_table.py qwen3_4b        # F1 / FPR / FNR per block -> pick R
#    rule: the earliest block within 0.01 F1 of the best whose FPR and null FPR are not worse;
#    on the 3B model that gave L19 (F1 plateau from ~L19 to the penultimate block)

# 2. write layer: rows at W in {R, R-3, R-5} under the R router, official MCF eval of base, ours
#    and genie (one base task, then one task per W, ~1 h each)
bash scripts/submit_layer_smoke.sh qwen3_4b write 24          # or: ... write 24 24,21,19,17
python scripts/layer_smoke_table.py qwen3_4b        # Eff / Gen / Spe / PPL per W -> pick W
#    rule: the lowest Gen whose Spe stays at base; genie across W isolates the write layer's own
#    effect from routing

# 3. the paper run with the chosen layers: 4 datasets x 5 seeds, base | ours | genie
ONEPASS=1 ROUTER_LAYER=24 WRITE_LAYER=21 bash scripts/submit_paper_model.sh qwen3_4b ours
python scripts/l16_paper_table.py --root outputs/paper_qwen3_4b_r24_w21_ours_v1 \
    --ours-label "Ours (L24 router, Δe L21)"
```

`DATASETS="mcf zsre"` runs both stages on ZsRE as well. Outputs: `outputs/layer_smoke_<MODEL>/`
(router sweep, `SUMMARY.md`), `outputs/layer_smoke_<MODEL>_r<R>_base/`, `outputs/layer_smoke_<MODEL>_r<R>_w<W>/`.
The routers fitted by the smoke test (`outputs/<ds>_<MODEL>_multiseed_ours_v1/seed1/L<R>/router`)
are reused by the paper run when it picks the same R.
- 8B GA/GD/NPO: fp32 AdamW over every MLP needs ~100 GB (weights + grads + Adam states),
  more than one 80 GB GPU; down_proj only (~55 GB) is the largest scope that fits.
- Qwen3: RWKU chat prompts use the template's no-thinking mode (`CHAT_ENABLE_THINKING=0`),
  otherwise the model writes a `<think>` block before answering. Llama prompts are unchanged.
- ZeroUnlearn computes each model's Wikipedia covariance once (`data/stats/<snapshot>/`);
  the submit script runs MCF seed 1 first so the other ZeroUnlearn tasks reuse it.
- Outputs: routers in `outputs/<ds>_<MODEL>_multiseed_*`, table runs in
  `outputs/paper_<MODEL>_v1/`. The driver refuses a router fit on a different model.
- Jobs check up front that the weights exist, the layer count matches, and the GPU has enough
  memory (8B: 75 GB, Qwen3-4B: 40 GB), and fail with a message instead of running.
