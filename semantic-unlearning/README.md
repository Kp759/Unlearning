# Semantic Token Unlearning

This project identifies the token IDs that semantically encode a "forget" concept in a large language model by training linear probes on per-layer hidden states. Once identified, these **semantic tokens** (T_f) can be targeted for embedding-level erasure without retraining the full model, enabling efficient machine unlearning via concept direction suppression.

## Pipeline

```
TOFU Dataset
    |
    v
[1] extract_hidden_states.py
    |  Runs LLM forward pass on forget/retain texts
    |  Saves per-layer representations as .npy
    v
[2] train_probe.py
    |  Trains LogisticRegression probe per layer
    |  Saves accuracy curve + probe_layer_NNN.pkl
    v
[3] identify_tokens.py
    |  Token-level probing to find T_f
    |  Filters by forget/retain selectivity ratio
    v
outputs/semantic_tokens.json
outputs/probe_accuracy_per_layer.png
outputs/token_scores.png
```

## Setup

```bash
pip install -r requirements.txt
```

For HuggingFace gated models (LLaMA), authenticate first:
```bash
huggingface-cli login
```

## Quick Start

```bash
python scripts/run_pipeline.py --config config/config.yaml
```

Skip extraction if hidden states are already cached:
```bash
python scripts/run_pipeline.py --config config/config.yaml --skip-extraction
```

## CPU Test Mode

To run a quick sanity check on CPU with a small subset:

```yaml
model:
  device: cpu
  dtype: float32

data:
  n_forget: 20
  n_retain: 20
```

## Datasets

| Dataset | Source | Description |
|---------|--------|-------------|
| TOFU | `locuslab/TOFU` (HuggingFace) | Fictional author biographies for machine unlearning benchmarks |

**Forget splits:** `forget01` (1%), `forget05` (5%), `forget10` (10%)  
**Retain splits:** `retain99`, `retain95`, `retain90` (complement of forget)

## Supported Models

| Model | HuggingFace ID |
|-------|---------------|
| LLaMA-3.2-1B | `meta-llama/Llama-3.2-1B` |
| LLaMA-3.1-8B | `meta-llama/Llama-3.1-8B` |
| Phi-3.5-mini | `microsoft/Phi-3.5-mini-instruct` |

## Key Outputs

| File | Description |
|------|-------------|
| `outputs/semantic_tokens.json` | Token IDs + probe scores for all identified T_f tokens |
| `outputs/probe_accuracy_per_layer.png` | Line plot of probe accuracy across layers |
| `outputs/token_scores.png` | Horizontal bar chart of top-k semantic tokens by score |
| `outputs/layer_accuracies.json` | Raw accuracy values per layer |
| `outputs/probes/probe_layer_NNN.pkl` | Saved LinearProbe objects for each layer |

## HPC (NJIT)

Submit via SLURM:
```bash
sbatch job.slurm
```

Logs are written to `outputs/logs/<job_id>.out` and `outputs/logs/<job_id>.err`.

## Related Work

- **ECO Prompts** — Hernandez et al., NeurIPS 2024. Concept erasure via embedding optimization.
- **LEACE** — Belrose et al., NeurIPS 2023. Least-squares concept erasure from representations.
- **Token Erasure** — Stolfo et al., EMNLP 2024. Erasing factual associations at the token embedding level.
