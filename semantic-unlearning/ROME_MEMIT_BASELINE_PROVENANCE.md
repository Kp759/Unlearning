# ROME / MEMIT baseline provenance for SURE-LM comparisons

This note locks the source of the ROME and MEMIT baselines used for comparisons against SURE-LM.

## Which code should be used

The ZeroUnlearn paper (arXiv:2605.18879) evaluates ROME and MEMIT on Llama-3.2-3B-Instruct, Llama-3.1-8B-Instruct, and Qwen3-4B. The relevant Llama-compatible implementations are the copies shipped by the authors in the official ZeroUnlearn repository, not a fresh checkout of the older standalone ROME/MEMIT repositories.

Official paper repository:

- `XMUDeepLIT/ZeroUnlearn`
- pinned paper-source commit: `deff011c3df367b700b9ad0aa0f5d7aad0cca9b9`

This repository already vendors that source under `../ZeroUnlearn/`. In particular:

- ROME implementation: `../ZeroUnlearn/rome/`
- MEMIT implementation: `../ZeroUnlearn/memit/`
- ROME Llama-3.2-3B-Instruct hparams: `../ZeroUnlearn/hparams/ROME/Llama-3.2-3B-Instruct.json`
- MEMIT Llama-3.2-3B-Instruct hparams: `../ZeroUnlearn/hparams/MEMIT/Llama-3.2-3B-Instruct.json`
- dispatch/evaluation entry point: `../ZeroUnlearn/experiments/evaluate.py`

The blob IDs of the ROME and MEMIT implementation files in this repository match the corresponding files in the pinned ZeroUnlearn paper repository. Run:

```bash
python scripts/verify_rome_memit_baseline_source.py
```

to re-check the vendored source after pulls/rebases.

## Original algorithm repositories

For historical provenance, the original algorithm repositories are:

- ROME: `kmeng01/rome`, pinned latest `main` commit `0874014cd9837e4365f3e6f3c71400ef11509e04`
- MEMIT: `kmeng01/memit`, pinned latest `main` commit `80426fd9316cf9a50c5ba15e0912f2c2c5bfe84b`

Both original projects use the MIT License. They are useful references for the original algorithms, but they are not the code path we should run for the SURE-LM comparison on Llama-3.2-3B-Instruct. The ZeroUnlearn paper repository contains the Llama-compatible adaptations and the exact paper hyperparameter files.

## Paper hyperparameters for Llama-3.2-3B-Instruct

ROME:

```text
layer = 18
fact_token = subject_last
v_num_grad_steps = 25
v_lr = 1e-1
v_loss_layer = 27
v_weight_decay = 0.5
kl_factor = 0.0625
mom2_adjustment = true
mom2_update_weight = 15000
rewrite module = model.layers.{}.mlp.down_proj
mom2 dataset = wikipedia
mom2 samples = 100000
```

MEMIT uses the same optimization/covariance settings but edits layers:

```text
16, 17, 18
```

## Fair-comparison rule for SURE-LM

Do **not** compare SURE-LM against a run produced by ZeroUnlearn's default random split logic. ROME and MEMIT must consume the same seed-specific training-visible forget facts used by the canonical SURE-LM protocol, and their final models must be evaluated by the same held-out benchmark evaluator and the same canonical PPL corpus/provenance path.

For MCF and ZsRE, the canonical SURE split exposes only direct forget requests during training/editing; paraphrases, neighborhood/locality probes, retain examples, and PPL text remain evaluation-only. MQuAKE follows the same canonical principle through its benchmark adapter.

The next integration layer should therefore wrap `../ZeroUnlearn/rome` and `../ZeroUnlearn/memit` rather than copy or modify their algorithm internals. This keeps the baseline implementation paper-faithful while making data access and evaluation identical to SURE-LM.
