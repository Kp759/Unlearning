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

- Layers are the 3B choices at the same relative depth, not re-tuned per model.
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
