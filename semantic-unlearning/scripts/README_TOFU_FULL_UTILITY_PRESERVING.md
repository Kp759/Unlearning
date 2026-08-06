# TOFU full-model utility-preserving fine-tuning

This experiment creates a Full-TOFU target checkpoint while explicitly
monitoring real-author and world-fact utility. It is separate from
`scripts/finetune_tofu.py`; the existing script and all existing checkpoints
remain unchanged.

This is supervised full-model fine-tuning, not an unlearning method. The
post-epoch probes may be used to choose the Full-TOFU checkpoint that will
later serve as a target model. They must not be used as unlearning training
data, repair data, or an unlearning stopping criterion.

## Training protocol

- Dataset: `locuslab/TOFU`, configuration `full`, split `train`.
- Records: exactly all 4,000 question/answer examples.
- Default base: the pinned Llama-3.2-3B-Instruct snapshot
  `0cb88a4f764b7a12671c53f0838cd831a0843b95`.
- Prompt: identical to `tofu_eval.py`. The user message is
  `Question: {question} Answer:` and the tokenizer chat template is applied
  with `add_generation_prompt=True`.
- Target text: one leading space, the answer, then `tokenizer.eos_token`.
- Supervision: every prompt token is masked with `-100`; only answer and EOS
  tokens contribute to the loss.
- Padding: `tokenizer.pad_token_id` for inputs and `-100` for labels.
- Optimizer: AdamW with gradient clipping at 1.0.
- Schedule: linear warmup followed by linear decay.
- Shuffle: deterministic per-epoch `DataLoader` generator derived from the run
  seed. This also makes an interrupted run reproducible when resumed from an
  epoch checkpoint.
- Parameters: the complete model is trainable. Total and trainable parameter
  counts are logged and recorded.

Defaults:

| Argument | Default |
| --- | ---: |
| learning rate | `1e-5` |
| epochs | `5` |
| batch size | `2` |
| gradient accumulation | `8` |
| effective batch size | `16` |
| weight decay | `0.01` |
| warmup ratio | `0.10` |
| maximum length | `256` |
| seed | `42` |
| dtype | `bf16` |
| gradient checkpointing | enabled |
| save after every epoch | enabled |

## Fixed utility probes

After each epoch, the script greedily generates answers for 20 fixed records
from each of:

- Full TOFU;
- `real_authors`;
- `world_facts`.

The fixed records are selected by sorted record SHA-256 rather than mutable
dataset row order. Generation uses the same prompt construction and greedy
settings as `tofu_eval.py`: `do_sample=False`, 64 new tokens, and the EOS token
as the generation pad token. Each `epoch_N_probe.json` records exact match,
ROUGE-L recall, and every question, gold answer, and generated answer.

These are lightweight checkpoint-selection probes. The trainer deliberately
does not invoke the full ECO evaluation.

## One run

On the NJIT HPC environment:

```bash
cd /scratch/yl258/kp759/Unlearning/semantic-unlearning

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /scratch/yl258/kp759/conda_envs/semantic_unlearning

python scripts/finetune_tofu_utility_preserving.py \
  --model-path /scratch/yl258/kp759/hf/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95 \
  --output-dir outputs/tofu_full_utility_preserving/lr_1e-5_epochs_5_seed_42 \
  --learning-rate 1e-5 \
  --epochs 5 \
  --batch-size 2 \
  --gradient-accumulation-steps 8 \
  --weight-decay 0.01 \
  --warmup-ratio 0.10 \
  --max-length 256 \
  --seed 42 \
  --dtype bf16 \
  --gradient-checkpointing \
  --save-every-epoch
```

The script refuses a nonempty output directory. It never deletes or replaces
an existing run.

## Resume an interrupted run

Resume only from a `checkpoint_epoch_N` created by the same configuration and
use the original output directory and total epoch count:

```bash
python scripts/finetune_tofu_utility_preserving.py \
  --model-path /scratch/yl258/kp759/hf/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95 \
  --output-dir outputs/tofu_full_utility_preserving/lr_1e-5_epochs_5_seed_42 \
  --resume-from-checkpoint outputs/tofu_full_utility_preserving/lr_1e-5_epochs_5_seed_42/checkpoint_epoch_2 \
  --learning-rate 1e-5 \
  --epochs 5 \
  --batch-size 2 \
  --gradient-accumulation-steps 8 \
  --weight-decay 0.01 \
  --warmup-ratio 0.10 \
  --max-length 256 \
  --seed 42 \
  --dtype bf16
```

Optimizer, scheduler, global-step, token-count, and random states are restored.
The script rejects a resume whose immutable training configuration differs.

## Full learning-rate/epoch sweep

The runner defaults to all 12 combinations of:

```text
LEARNING_RATES="1e-5 2e-5 5e-5"
EPOCHS_LIST="1 2 3 5"
```

Run it with:

```bash
cd /scratch/yl258/kp759/Unlearning/semantic-unlearning

export PYTHON_BIN="$(command -v python)"
export BASE_MODEL_PATH=/scratch/yl258/kp759/hf/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/0cb88a4f764b7a12671c53f0838cd831a0843b95
export OUTPUT_ROOT=/scratch/yl258/kp759/Unlearning/semantic-unlearning/outputs/tofu_full_utility_sweep
export LEARNING_RATES="1e-5 2e-5 5e-5"
export EPOCHS_LIST="1 2 3 5"
export SEED=42

bash scripts/run_tofu_full_utility_sweep.sh
```

Optional runner environment variables are `BATCH_SIZE`,
`GRADIENT_ACCUMULATION_STEPS`, `WEIGHT_DECAY`, `WARMUP_RATIO`, `MAX_LENGTH`,
`DTYPE`, `GRADIENT_CHECKPOINTING`, and `SAVE_EVERY_EPOCH`.

Every pair receives a separate directory. A completed directory is skipped on
a later invocation; an incomplete nonempty directory produces an error. No
existing run is overwritten.

The runner writes `sweep_summary.csv` and `sweep_summary.md`. A run is eligible
only when:

- Full-TOFU probe ROUGE-L is at least 0.95;
- real-author probe ROUGE-L is at least 0.75;
- world-fact probe ROUGE-L is at least 0.75.

Selection is lexicographic: highest Full-TOFU ROUGE-L, highest mean external
ROUGE-L, lowest learning rate, then fewest epochs. If no run satisfies every
gate, the summary explicitly selects none.

## Outputs

Each run contains:

- `config_used.json` — complete training configuration and its SHA-256;
- `finetune_metadata.json` — base revision, dataset fingerprints, dependency
  versions, Git commit/dirty state, parameter counts, loss curve, and probe
  history;
- `train_progress.jsonl` — incrementally flushed optimizer-step loss, learning
  rate, gradient norm, tokens per step, and cumulative tokens;
- `checkpoint_epoch_N/` — model, tokenizer, and resumable trainer state;
- `epoch_N_probe.json` — fixed utility probe metrics and generations;
- `final/` — final model and tokenizer plus copied provenance files.

Do not substitute `final/` for an official Full-TOFU target until the sweep
eligibility gates and the subsequent native TOFU evaluation have both been
checked.
