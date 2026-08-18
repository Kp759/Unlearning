#!/usr/bin/env python3
"""Compatibility shims for the pinned ZeroUnlearn ROME/MEMIT source.

The paper-source implementation predates two API/behavior changes in the
runtime used by this repository:

1. Newer Transformers can expose a Llama decoder block output directly as a
   tensor, while MEMIT's ``compute_z`` expects a one-element tuple containing
   that tensor.
2. The paper's ``TokenizedDataset`` prepends a Python retain list using
   ``retain_list + text_dataset``.  With current HuggingFace ``datasets``, the
   right-hand object is a lazy ``Dataset`` and Python list concatenation raises
   ``TypeError``.  The intended semantics are simple concatenation, so we keep
   a lightweight sequence view instead of materializing millions of Wikipedia
   rows into a Python list.

These shims are installed outside ``ZeroUnlearn/``.  The pinned ROME/MEMIT
files therefore remain byte-for-byte unchanged and provenance verification
continues to validate the paper source.
"""
from __future__ import annotations

from typing import Any

import torch


def _is_llama_decoder_layer_name(layer: Any) -> bool:
    if not isinstance(layer, str):
        return False
    parts = layer.split(".")
    return (
        len(parts) == 3
        and parts[0] == "model"
        and parts[1] == "layers"
        and parts[2].isdigit()
    )


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


class _PrefixedSequence:
    """Lazy list-prefix + dataset view with ordinary sequence semantics."""

    def __init__(self, prefix, base):
        self.prefix = prefix
        self.base = base

    def __len__(self):
        return len(self.prefix) + len(self.base)

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return [self[i] for i in range(start, stop, step)]
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        if index < len(self.prefix):
            return self.prefix[index]
        return self.base[index - len(self.prefix)]


def install_memit_decoder_output_compat(nethook_module) -> bool:
    """Install an idempotent TraceDict adapter for current Transformers Llama."""
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
                        edited = invoke(
                            edit_output, output=legacy_output, layer=layer
                        )
                        if (
                            isinstance(edited, (tuple, list))
                            and len(edited) == 1
                            and torch.is_tensor(edited[0])
                        ):
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
    nethook_module._memit_decoder_output_compat_original_trace_dict = (
        original_trace_dict
    )
    return True


def install_tokenized_dataset_concat_compat(tokenized_dataset_cls) -> bool:
    """Replace legacy ``list + HF Dataset`` with a lazy concatenation view.

    The transformation is representation-only: item order and values are the
    same as the paper code intended.  In the canonical ROME/MEMIT baseline the
    retain prefix is empty, so the underlying Wikipedia Dataset is retained
    directly with zero copying.
    """
    if getattr(tokenized_dataset_cls, "_hf_dataset_concat_compat", False):
        return False

    original_init = tokenized_dataset_cls.__init__

    def compat_init(
        self,
        text_dataset,
        retain_data=None,
        tokenizer=None,
        maxlen=None,
        field="text",
    ):
        self.text_dataset = text_dataset
        if retain_data is not None:
            self.retain_data = [
                {
                    "text": row["prompt"].format(row["subject"])
                    + " {}".format(row["target_true"]["str"])
                }
                for row in retain_data
                if row["target_true"]["str"][0] != " "
            ]
        else:
            self.retain_data = []

        print(f"add retain data: {retain_data}")
        print(
            f"wikipedia data length: {len(text_dataset)}, "
            f"retain data length: {len(self.retain_data)}"
        )

        if self.retain_data:
            self.text_dataset = _PrefixedSequence(self.retain_data, text_dataset)
        else:
            # This is the canonical baseline path: preserve the lazy HF Dataset
            # exactly rather than attempting [] + Dataset or materializing it.
            self.text_dataset = text_dataset

        print(f"Total text dataset length: {len(self.text_dataset)}")
        self.field = field
        self.tokenizer = tokenizer
        self.maxlen = maxlen
        if hasattr(text_dataset, "info"):
            self.info = text_dataset.info

    tokenized_dataset_cls.__init__ = compat_init
    tokenized_dataset_cls._hf_dataset_concat_compat = True
    tokenized_dataset_cls._hf_dataset_concat_compat_original_init = original_init
    return True
