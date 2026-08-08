# Best runs organized by model

This directory groups benchmark-result snapshots by model family for cross-model experiments.

Current layout:

- `llama_3b_instruct_model/` — populated with the latest MCF, ZsRE, and TOFU results for the Llama-3.2-3B-Instruct model family.
- `llama_8b_instruct_model/` — reserved for upcoming 8B Instruct MCF/ZsRE/TOFU runs.
- `qwen_model/` — reserved for upcoming Qwen MCF/ZsRE/TOFU runs.

Each populated model directory should contain dataset-specific subdirectories (`mcf/`, `zsre/`, `tofu/`) and a `manifest.json`. Canonical dataset-specific records remain under `config/best_runs/mcf`, `config/best_runs/zsre`, and `config/best_runs/tofu`.
