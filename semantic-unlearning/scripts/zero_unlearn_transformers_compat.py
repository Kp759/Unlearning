#!/usr/bin/env python3
"""Compatibility shims for the pinned ZeroUnlearn ROME/MEMIT source.

The paper-source MEMIT implementation was written against a Transformers
Llama decoder-layer contract where a traced decoder block exposed a tuple
whose first element was the hidden-state tensor.  Newer Transformers versions
can expose that decoder block output directly as a tensor.  MEMIT's compute_z
still indexes ``cur_out[0]`` and ``trace.output[0]``.

This module adapts only the TraceDict boundary for bare decoder-layer modules
(``model.layers.N``).  The underlying third-party ROME/MEMIT files remain
byte-for-byte unchanged, so provenance verification still succeeds.
"""
from __future__ import annotations

from typing import Any

import torch


def _is_llama_decoder_layer_name(layer: Any) -> bool:
    if not isinstance(layer, str):
        return False
    parts = layer.split(".")
    return len(parts) == 3 and parts[0] == "model" and parts[1] == "layers" and parts[2].isdigit()


class _TraceProxy:
    """Expose a direct hidden tensor as the legacy one-element tuple view."""

    def __init__(self, trace: Any):
        self._trace = trace

    @property
    def output(self):
        value = self._trace.output
        if torch.is_tensor(value):
            return (value,)
        return value

    def __getattr__(self, name: str):
        return getattr(self._trace, name)


def install_memit_decoder_output_compat(nethook_module) -> bool:
    """Install an idempotent TraceDict adapter for current Transformers Llama.

    Returns True when a new adapter is installed and False when it was already
    installed.  Only exact decoder-block traces are adapted; MLP/down-projection
    traces keep their native tensor contract.
    """
    if getattr(nethook_module, "_memit_decoder_output_compat_installed", False):
        return False

    original_trace_dict = nethook_module.TraceDict
    invoke = nethook_module.invoke_with_optional_args

    class CompatTraceDict(original_trace_dict):
        def __init__(self, *args, **kwargs):
            edit_output = kwargs.get("edit_output")
            if edit_output is not None:
                def wrapped_edit_output(output, layer=None):
                    if torch.is_tensor(output) and _is_llama_decoder_layer_name(layer):
                        legacy_output = (output,)
                        edited = invoke(edit_output, output=legacy_output, layer=layer)
                        if isinstance(edited, (tuple, list)) and len(edited) == 1 and torch.is_tensor(edited[0]):
                            return edited[0]
                        return edited
                    return invoke(edit_output, output=output, layer=layer)

                kwargs["edit_output"] = wrapped_edit_output
            super().__init__(*args, **kwargs)

        def __getitem__(self, key):
            trace = super().__getitem__(key)
            if _is_llama_decoder_layer_name(key):
                return _TraceProxy(trace)
            return trace

    nethook_module.TraceDict = CompatTraceDict
    nethook_module._memit_decoder_output_compat_installed = True
    nethook_module._memit_decoder_output_compat_original_trace_dict = original_trace_dict
    return True
