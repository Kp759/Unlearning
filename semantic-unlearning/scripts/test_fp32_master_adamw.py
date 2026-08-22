#!/usr/bin/env python3
"""Tiny CPU smoke test proving sub-BF16-step accumulation via FP32 masters."""
import torch
from fp32_master_adamw import make_fp32_master_adamw_class

Base = torch.optim.AdamW
Opt = make_fp32_master_adamw_class(Base)
p = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.bfloat16))
opt = Opt([{"params": [p], "lr": 1e-4, "weight_decay": 0.0}])
start = p.detach().clone()
for _ in range(100):
    opt.zero_grad(set_to_none=True)
    p.grad = torch.tensor([1.0], dtype=torch.bfloat16)
    opt.step()
assert float(p.item()) < float(start.item()), (start, p)
print("FP32-master AdamW smoke test PASS", float(start.item()), float(p.item()))
