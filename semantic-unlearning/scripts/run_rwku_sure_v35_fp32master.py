#!/usr/bin/env python3
"""Run RWKU v3.5 with FP32 master weights for BF16 embeddings/LM head."""
from __future__ import annotations

import torch

from fp32_master_adamw import make_fp32_master_adamw_class

_BASE_ADAMW = torch.optim.AdamW
torch.optim.AdamW = make_fp32_master_adamw_class(_BASE_ADAMW)

import rwku_sure_v35_emb_head_hidden_direction_kl_w1k as v35

v35.SCHEMA = "rwku_sure_v35_emb_head_hidden_direction_kl_w1k_fp32master_configuration_v1"
v35.EXPERIMENT_ID = "rwku-h-w1k-stephen-king-emb-head-hidden-direction-seed0-v35-kl-fp32master"
v35.DEFAULT_CONFIGURATION = v35.PROJECT_ROOT / "config" / "rwku" / "sure_v35_emb_head_hidden_direction_kl_w1k_fp32master_seed0.json"
v35.LEARNER_DIR = "sure_v35_emb_head_hidden_direction_w1k_fp32master"

if __name__ == "__main__":
    v35.main()
