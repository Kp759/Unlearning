# AGENTS.md

## Cursor Cloud specific instructions

This repository is a **monorepo of four independent LLM machine-unlearning research
projects**, each with its own pinned (and mutually conflicting) dependencies. They are
CLI / batch experiment tools — there is **no long-running server and no GUI**. Test them
by running scripts and inspecting terminal output / generated files.

| Subproject | What it is | Main entry point |
|---|---|---|
| `open-unlearning` | Benchmarking framework (TOFU/MUSE/WMDP), Hydra-configured, pip-installable | `src/train.py`, `src/eval.py` |
| `semantic-unlearning` | Probing pipeline that finds "semantic tokens" of a forget concept | `scripts/run_pipeline.py` |
| `eco` | Inference-time unlearning via embedding-corrupted prompts (forward hook) | `demo.py`, `python -m scripts.*` |
| `ZeroUnlearn` | Few-shot unlearning as knowledge editing (ROME/MEMIT/AlphaEdit baselines) | `experiments/evaluate.py` |

### Environment layout (important)
- **Each project has its own venv at `<project>/.venv`** because their pinned `torch`
  / `transformers` versions conflict. Always use the matching venv, e.g.
  `open-unlearning/.venv/bin/python ...`. Do not try to share one environment.
- The startup **update script** creates/refreshes all four venvs. All torch installs use
  the **CPU wheel index** (`https://download.pytorch.org/whl/cpu`) — this VM has **no GPU**.
- Dependency installs are large; the first run on a fresh (non-snapshot) pod takes a while.

### GPU / model gotchas (this VM is CPU-only, ~15 GB RAM)
- No CUDA. For any run you must force CPU and avoid GPU-only options:
  - **open-unlearning**: override `trainer.args.bf16=false trainer.args.bf16_full_eval=false`,
    `trainer.args.optim=sgd` (default `paged_adamw_32bit` needs bitsandbytes+CUDA),
    `model.model_args.attn_implementation=eager model.model_args.torch_dtype=float32`,
    and set `CUDA_VISIBLE_DEVICES=`. Also set `trainer.args.warmup_epochs=0` — otherwise
    `load_trainer_args` divides by `torch.cuda.device_count()` (=0 on CPU) and crashes.
  - **eco / semantic-unlearning**: model configs default to `flash_attention_2`; use
    `eager`/`float32` and `device: cpu` instead.
- **Gated models**: meta-llama/* requires an `HF_TOKEN` + license acceptance. To run
  without one, use the **public re-uploads** under the `open-unlearning/` HF org, e.g.
  `open-unlearning/tofu_Llama-3.2-1B-Instruct_full` (this repo also ships its own tokenizer,
  so point both model and tokenizer args at it). Small public models such as
  `Qwen/Qwen1.5-0.5B-Chat` work well for CPU smoke tests. Full-size (8B) runs will OOM on CPU.

### Per-project run notes
- **open-unlearning**: lint = `open-unlearning/.venv/bin/ruff check ...` (see `Makefile`
  `quality`; CI runs only lint, there is **no `tests/` dir** despite the `make test` target).
  Note: `src/trainer/unlearn/rmu.py` has a **pre-existing** lint/`SyntaxWarning` issue
  (`#import deepspeed` is commented out) — not caused by setup; do not "fix" as part of setup.
  A minimal CPU unlearning smoke test (GradAscent, `+trainer.args.max_steps=2`) works.
- **semantic-unlearning**: `scripts/run_pipeline.py --config <cfg>` chains extract →
  train_probe → identify_tokens. Point `model.name` at a small public model and set
  `device: cpu`, `dtype: float32`, small `n_forget`/`n_retain` for a quick CPU run.
  Scripts import via `sys.path.insert`, so run from the project root.
- **eco**: run scripts with `PYTHONPATH=<repo>/eco` (the `eco` package lives under
  `eco/eco`). Core mechanism = `apply_corruption_hook` on the embedding module. Needs
  the spaCy model: `eco/.venv/bin/python -m spacy download en_core_web_sm`.
- **ZeroUnlearn**: requirements file is misspelled `requirments.txt`. `spacy==3.4.1` (pinned)
  **cannot build on Python 3.12**, but spaCy is never imported by the code (only `nltk`),
  so the update script installs everything except that pin plus a modern `spacy>=3.8`.
  For its evaluation code you also need NLTK data: `python -c "import nltk;
  nltk.download('punkt'); nltk.download('punkt_tab')"`. Full editing/eval runs (MEMIT/ROME
  covariance stats) realistically require a GPU + model weights.
