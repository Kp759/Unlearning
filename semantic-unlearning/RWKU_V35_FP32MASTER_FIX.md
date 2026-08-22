# RWKU v3.5 FP32-master implementation fix

The first v3.5 run used BF16 physical input-embedding and LM-head parameters directly with AdamW at LR 1e-4. The run reported exactly zero embedding drift and an LM-head delta norm identical to the inherited Stage-1 sparse-head delta at every checkpoint, indicating that the intended new parameter path did not materially update in physical BF16.

This branch preserves the v3.5 scientific method unchanged and corrects only the optimizer implementation:

- input embeddings: BF16 physical parameter + FP32 master parameter;
- untied LM head: BF16 physical parameter + FP32 master parameter;
- final down_proj LoRA A/B: already FP32, unchanged;
- same learning rates, losses, ranks, scales, data boundaries, and gates;
- gradients are clipped on the original trainable tensors, copied to FP32 masters, AdamW updates the masters, and masters are materialized back to BF16 after every step.

The original v3.5 run remains a diagnostic implementation-failure artifact and must not be interpreted as evidence that joint embedding/head training fails.
