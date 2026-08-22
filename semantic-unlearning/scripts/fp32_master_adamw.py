#!/usr/bin/env python3
"""Transparent FP32-master AdamW for low-precision trainable tensors.

This helper preserves the requested optimizer/LR while allowing BF16/FP16 model
weights to accumulate sub-ULP updates in FP32 master copies. FP32 parameters
(e.g. RWKU LoRA A/B matrices) are passed through unchanged.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping

import torch


def make_fp32_master_adamw_class(base_adamw):
    class FP32MasterAdamW:
        def __init__(self, params, *args, **kwargs):
            raw_groups = list(params)
            if raw_groups and not isinstance(raw_groups[0], Mapping):
                raw_groups = [{"params": raw_groups}]

            self._master_pairs = []
            master_groups = []
            for group in raw_groups:
                copied = dict(group)
                original_params = list(copied["params"])
                optimizer_params = []
                for parameter in original_params:
                    if not isinstance(parameter, torch.nn.Parameter):
                        raise TypeError("FP32MasterAdamW expects torch.nn.Parameter objects")
                    if parameter.dtype in (torch.bfloat16, torch.float16):
                        master = torch.nn.Parameter(
                            parameter.detach().float().clone(), requires_grad=True
                        )
                        self._master_pairs.append((parameter, master))
                        optimizer_params.append(master)
                    else:
                        optimizer_params.append(parameter)
                copied["params"] = optimizer_params
                master_groups.append(copied)

            self._optimizer = base_adamw(master_groups, *args, **kwargs)
            self.param_groups = self._optimizer.param_groups
            self.state = self._optimizer.state
            low_precision_count = len(self._master_pairs)
            low_precision_numel = sum(int(p.numel()) for p, _ in self._master_pairs)
            print(
                f"[FP32MasterAdamW] FP32 master tensors={low_precision_count} "
                f"params={low_precision_numel:,}"
            )

        def zero_grad(self, set_to_none: bool = True):
            self._optimizer.zero_grad(set_to_none=set_to_none)
            for original, _ in self._master_pairs:
                if set_to_none:
                    original.grad = None
                elif original.grad is not None:
                    original.grad.zero_()

        def step(self, closure=None):
            for original, master in self._master_pairs:
                if original.grad is None:
                    master.grad = None
                else:
                    master.grad = original.grad.detach().float().clone()

            loss = self._optimizer.step(closure=closure)

            with torch.no_grad():
                for original, master in self._master_pairs:
                    original.copy_(master.to(device=original.device, dtype=original.dtype))
            return loss

        def state_dict(self):
            return self._optimizer.state_dict()

        def load_state_dict(self, state_dict):
            return self._optimizer.load_state_dict(state_dict)

        def __getattr__(self, name: str) -> Any:
            if name.startswith("_"):
                raise AttributeError(name)
            return getattr(self._optimizer, name)

    FP32MasterAdamW.__name__ = "FP32MasterAdamW"
    return FP32MasterAdamW
