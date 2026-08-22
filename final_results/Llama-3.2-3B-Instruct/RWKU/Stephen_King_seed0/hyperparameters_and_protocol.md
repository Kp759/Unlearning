# Hyperparameters and protocol

## Core experiment identity

```text
configuration_id: rwku-h-w1k-stephen-king-hidden-direction-seed0-v32-kl
schema: rwku_sure_head_hidden_direction_v32_kl_w1k_configuration_v1
seed: 0
target_entity: Stephen King
target_entity_id: rwku:1_Stephen_King
neutral_target: Unknown
method status: post-hoc development only
```

## Trainable module

```text
base model: Llama-3.2-3B-Instruct
trainable stage-2 module: final transformer block MLP down_proj only
adapter type: LoRA
rank ladder: [1, 2, 4]
alpha: rank
dropout: 0.0
last_n_layers: 1
target_modules: [down_proj]
all other transformer parameters: frozen
input embedding / frozen base readout W0: frozen
cloned output head: receives pre-existing sparse Stage-1 head edit, otherwise frozen during representation repair
```

## Optimization

```text
steps: 300
optimizer: AdamW
learning_rate: 5e-4
weight_decay: 0.0
grad_clip: 1.0
answer_batch_size: 8
answer_eval_batch_size: 8
checkpoint_interval: 25
candidate_scales: [0.125, 0.25, 0.5, 0.75, 1.0]
```

## Forgetting objective

```text
objective: answer_level_frozen_base_head_sensitive_vs_neutral_plus_exact_utility_kl
sensitive_view_scope: all_generated_atomic_views
frozen_base_head_training_margin: 0.5
frozen_base_head_answer_weight: 8.0
edited_head_pairwise_target: 0.5
edited_head_answer_weight: 2.0
adapter_l2_weight: 1e-4
```

For each generated prompt `q`:

```text
base_sep   = NLL_W0(sensitive)   - NLL_W0(Unknown)
edited_sep = NLL_Wedit(sensitive) - NLL_Wedit(Unknown)

L_base = ReLU(0.5 - base_sep)^2
L_edit = ReLU(0.5 - edited_sep)^2
```

## Utility preservation

```text
external utility source: Wikipedia only
utility_train_prompt_count: 1000
utility_kl_batch_size: 4
utility_context_batch_size: 4
utility_hidden_weight: 2.0
utility_kl_weight: 50.0
checkpoint_wiki_prompt_count: 128
checkpoint_kl_prompt_count: 128
KL direction: KL(P_base || P_edit)
KL vocabulary scope: full vocabulary
train / held-out overlap allowed: false
```

The four utility contexts per optimizer step are processed sequentially with gradient accumulation. At 300 steps, 1,000 utility contexts are all visited within the first 250 steps, with the final 50 steps cycling through the beginning of the fixed optimization pool.

## v3.2 checkpoint-selection policy

At each 25-step checkpoint, the checkpoint is behavior-eligible if all of the following hold on the generated sensitive views:

```text
frozen-base-head recovery count == 0
minimum frozen-base-head demotion margin >= 0.05
atomic margin failures == 0
```

Among behavior-eligible checkpoints, v3.2 selects lexicographically by lower optimization-pool utility damage:

```text
1. exact optimization KL mean
2. exact optimization KL p95
3. exact optimization KL max
4. hidden-state relative MSE
```

For rank 1, this selects **step 275**, whose 128-context optimization checkpoint report is:

```text
recovery: 0%
minimum frozen-head margin: +0.6961
atomic failures: 0
optimization KL mean: 0.000706
optimization KL p95: 0.002113
optimization KL max: 0.005346
hidden relative MSE mean: 0.000435
```

After checkpoint selection, the scale ladder is tested. For the selected rank-1 checkpoint:

```text
scale 0.125 -> recovery 100.00%, relative norm 0.1751%
scale 0.25  -> recovery 93.75%, relative norm 0.3633%
scale 0.50  -> recovery 89.58%, relative norm 0.7152%
scale 0.75  -> recovery 50.00%, relative norm 1.0630%
scale 1.00  -> recovery 0.00%, relative norm 1.4117%
```

Thus full scale is required for 0% recovery for the selected step, but that candidate exceeds the 1% intervention budget.

## Predeclared acceptance gates

```text
required_pairwise_margin: 0.01
required_direct_success: 100.0%
required_other_atomic_view_success: 100.0%
max_frozen_base_head_recovery: 0.0%
min_frozen_base_head_demotion_margin: 0.05
max_head_delta_norm: 1.5
max_relative_frobenius_delta: 0.01
utility_kl_mean_budget: 0.01
utility_kl_p95_budget: 0.05
utility_kl_max_budget: 0.5
checkpoint_dtype: bf16
device_map: single
```

## Held-out utility diagnostic

After explicit authorization to open the holdout, the already-selected rank-1 step-275 scale-1.0 candidate was evaluated on a separate **1,000-context disjoint Wikipedia utility set**.

```text
held-out KL mean: 0.000386   PASS <= 0.010
held-out KL p95:  0.001657   PASS <= 0.050
held-out KL max:  0.036320   PASS <= 0.500
```

This held-out set is now **opened**. It must not be used to tune future rank, step, scale, loss weight, margin, or any other method hyperparameter.

## Current gate vector for the v3.2 diagnostic candidate

```text
frozen-base-head recovery: PASS
minimum frozen-head margin: PASS
direct atomic success: PASS
other atomic success: PASS
held-out utility KL mean: PASS
held-out utility KL p95: PASS
held-out utility KL max: PASS
relative Frobenius <= 1%: FAIL (1.4117%)
fully feasible: NO
```
