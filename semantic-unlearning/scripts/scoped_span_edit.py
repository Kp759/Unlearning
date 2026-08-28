#!/usr/bin/env python3
"""Runtime support for exact-subject-scoped hidden and logit interventions.

The important invariant is deliberately simple: if no complete subject token
sequence is present in an input row, every hook returns its input unchanged.
The model is therefore bit-identical to Base for a closed gate.  A single
router drives both the hidden-state writer and the output reader; gating only
the writer would still leave a global LM-head edit active and would not solve
reader leakage.

The module has no dependency on the training script so a serialized sidecar
can be discovered and installed by the official evaluator after a normal
``AutoModelForCausalLM.from_pretrained`` load.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn


SIDECAR_NAME = "scoped_span_edit.pt"
SCHEMA_VERSION = 1


def find_decoder_layers(model: nn.Module) -> Sequence[nn.Module]:
    """Return decoder blocks for common Hugging Face causal-LM layouts."""
    candidates: List[Any] = []
    if hasattr(model, "model"):
        inner = getattr(model, "model")
        candidates.extend((getattr(inner, "layers", None), getattr(inner, "h", None)))
        decoder = getattr(inner, "decoder", None)
        if decoder is not None:
            candidates.append(getattr(decoder, "layers", None))
    transformer = getattr(model, "transformer", None)
    if transformer is not None:
        candidates.append(getattr(transformer, "h", None))
    for layers in candidates:
        if layers is not None and hasattr(layers, "__len__") and len(layers) > 0:
            return layers
    raise RuntimeError(
        "Could not locate transformer decoder blocks; expected model.model.layers "
        "or a compatible Hugging Face causal-LM layout"
    )


def _dedupe_patterns(patterns: Sequence[Sequence[int]]) -> List[List[int]]:
    seen = set()
    out: List[List[int]] = []
    for pattern in patterns:
        value = tuple(int(x) for x in pattern)
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(list(value))
    out.sort(key=lambda value: (-len(value), value))
    return out


def subject_token_patterns(tokenizer, subject: str) -> List[List[int]]:
    """Tokenizations that cover a complete subject at start or after space.

    Matching is performed against the *actual* model ``input_ids``, so BOS and
    padding positions remain aligned.  Encoding both the bare and
    whitespace-prefixed spellings handles SentencePiece/byte-BPE tokenizers
    whose first subject piece depends on left whitespace.
    """
    patterns = []
    for text in (str(subject), " " + str(subject)):
        encoded = tokenizer(text, add_special_tokens=False)["input_ids"]
        if encoded and isinstance(encoded[0], list):
            if len(encoded) != 1:
                raise ValueError("subject tokenizer unexpectedly returned a batch")
            encoded = encoded[0]
        patterns.append([int(x) for x in encoded])
    return _dedupe_patterns(patterns)


def build_subject_patterns(tokenizer, subjects: Sequence[str]) -> List[List[List[int]]]:
    return [subject_token_patterns(tokenizer, subject) for subject in subjects]


def _find_pattern(sequence: Sequence[int], pattern: Sequence[int]) -> Optional[int]:
    width = len(pattern)
    if width == 0 or width > len(sequence):
        return None
    target = list(pattern)
    for start in range(len(sequence) - width + 1):
        if list(sequence[start : start + width]) == target:
            return start
    return None


@dataclass
class RouteState:
    active: torch.Tensor       # [batch, records], bool
    span_masks: torch.Tensor   # [batch, records, seq], bool


class SpanGateRouter:
    """Route complete subject occurrences to record-specific interventions."""

    def __init__(
        self,
        embedding: nn.Module,
        subject_patterns: Sequence[Sequence[Sequence[int]]],
        subjects: Optional[Sequence[str]] = None,
        model: Optional[nn.Module] = None,
    ) -> None:
        self.subject_patterns = [
            _dedupe_patterns(record_patterns) for record_patterns in subject_patterns
        ]
        if any(not patterns for patterns in self.subject_patterns):
            raise ValueError("every scoped record needs at least one non-empty subject pattern")
        if subjects is not None and len(subjects) != len(self.subject_patterns):
            raise ValueError("subjects and subject_patterns must have identical lengths")
        self.subject_priorities = (
            [len(str(subject)) for subject in subjects]
            if subjects is not None else
            [max(len(pattern) for pattern in patterns) for patterns in self.subject_patterns]
        )
        self.enabled = True
        self.calls = 0
        self.matched_rows = 0
        self.state: Optional[RouteState] = None
        self._model_handle = (
            model.register_forward_pre_hook(self._clear_before_model_forward)
            if model is not None else None
        )
        self._handle = embedding.register_forward_pre_hook(self._pre_hook)

    @property
    def n_records(self) -> int:
        return len(self.subject_patterns)

    def route(self, input_ids: torch.Tensor) -> RouteState:
        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        if input_ids.ndim != 2:
            raise ValueError(f"expected [batch, seq] input_ids, got {tuple(input_ids.shape)}")
        batch, seq_len = input_ids.shape
        active = torch.zeros((batch, self.n_records), dtype=torch.bool)
        masks = torch.zeros((batch, self.n_records, seq_len), dtype=torch.bool)
        if not self.enabled:
            return RouteState(active=active, span_masks=masks)

        rows = input_ids.detach().cpu().tolist()
        for batch_index, sequence in enumerate(rows):
            matches: List[Tuple[int, int, int, int]] = []
            for record_index, patterns in enumerate(self.subject_patterns):
                best: Optional[Tuple[int, int]] = None
                for pattern in patterns:
                    start = _find_pattern(sequence, pattern)
                    if start is None:
                        continue
                    candidate = (len(pattern), start)
                    if best is None or candidate[0] > best[0]:
                        best = candidate
                if best is not None:
                    matches.append(
                        (
                            record_index,
                            self.subject_priorities[record_index],
                            best[0],
                            best[1],
                        )
                    )
            if not matches:
                continue

            # Prefer the longest complete subject.  This prevents a record for
            # "York" also firing inside "New York".  Equal longest matches are
            # retained so duplicate-subject records remain explicit rather
            # than being silently assigned to an arbitrary relation.
            longest = max(priority for _, priority, _, _ in matches)
            for record_index, priority, width, start in matches:
                if priority != longest:
                    continue
                active[batch_index, record_index] = True
                masks[batch_index, record_index, start : start + width] = True
        return RouteState(active=active, span_masks=masks)

    def _pre_hook(self, module, inputs):
        self.calls += 1
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            self.state = None
            return None
        self.state = self.route(inputs[0])
        self.matched_rows += int(self.state.active.any(dim=1).sum().item())
        return None

    def _clear_before_model_forward(self, module, inputs):
        # Prevent stale routing if a caller supplies inputs_embeds and bypasses
        # the embedding module. The embedding pre-hook will repopulate state for
        # ordinary input_ids forwards.
        self.state = None
        return None

    def close(self) -> None:
        self._handle.remove()
        if self._model_handle is not None:
            self._model_handle.remove()
        self.state = None


class SpanGatedWriter:
    """Per-record residual vectors applied only at routed subject positions."""

    def __init__(
        self,
        layer_module: nn.Module,
        router: SpanGateRouter,
        hidden_size: int,
        device: torch.device,
    ) -> None:
        self.router = router
        self.delta = nn.Parameter(
            torch.zeros((max(1, router.n_records), hidden_size), dtype=torch.float32, device=device)
        )
        self.enabled = True
        self.fired = 0
        self.calls = 0
        self._handle = layer_module.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self.calls += 1
        state = self.router.state
        if not self.enabled or state is None or not bool(state.active.any()):
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.ndim != 3:
            return output
        masks = state.span_masks
        if masks.shape[0] != hidden.shape[0] or masks.shape[2] != hidden.shape[1]:
            raise RuntimeError(
                "span gate shape does not match decoder hidden state: "
                f"gate={tuple(masks.shape)}, hidden={tuple(hidden.shape)}"
            )
        if masks.shape[1] != self.delta.shape[0]:
            raise RuntimeError("span gate record count does not match writer delta")
        delta = torch.einsum(
            "brs,rh->bsh",
            masks.to(device=hidden.device, dtype=hidden.dtype),
            self.delta.to(device=hidden.device, dtype=hidden.dtype),
        )
        self.fired += int(state.active.any(dim=1).sum().item())
        updated = hidden + delta
        if isinstance(output, tuple):
            return (updated,) + tuple(output[1:])
        return updated

    def close(self) -> None:
        self._handle.remove()


class ScopedLogitReader:
    """Sparse per-record LM-head residual controlled by the same span gate.

    ``row_ids[r, k]`` names the vocabulary row for ``deltas[r, k]``; ``-1``
    marks padding.  The base LM-head weights remain untouched.  Consequently
    a closed gate returns the exact base logits and removes the global ``L``
    term that survives a writer-only gate.
    """

    def __init__(
        self,
        output_layer: nn.Module,
        router: SpanGateRouter,
        row_ids: torch.Tensor,
        deltas: torch.Tensor,
        biases: Optional[torch.Tensor] = None,
        scale: float = 1.0,
    ) -> None:
        if row_ids.ndim != 2 or deltas.ndim != 3:
            raise ValueError("row_ids must be [records, rows] and deltas [records, rows, hidden]")
        if tuple(row_ids.shape) != tuple(deltas.shape[:2]):
            raise ValueError("reader row_ids/deltas shapes disagree")
        if biases is not None and tuple(biases.shape) != tuple(row_ids.shape):
            raise ValueError("reader biases must have the same [records, rows] shape as row_ids")
        if row_ids.shape[0] != router.n_records:
            raise ValueError("reader record count does not match router")
        self.router = router
        self.row_ids = row_ids.detach().to(dtype=torch.long)
        self.deltas = deltas.detach().to(dtype=torch.float32)
        self.biases = (
            None if biases is None else biases.detach().to(dtype=torch.float32)
        )
        self.scale = float(scale)
        self.enabled = True
        self.calls = 0
        self.fired = 0
        self._handle = output_layer.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self.calls += 1
        state = self.router.state
        if not self.enabled or state is None or not bool(state.active.any()):
            return output
        logits = output[0] if isinstance(output, tuple) else output
        if not inputs or not isinstance(inputs[0], torch.Tensor):
            return output
        hidden = inputs[0]
        if hidden.ndim != 3 or logits.ndim != 3:
            return output
        active = state.active
        if active.shape[0] != hidden.shape[0]:
            raise RuntimeError("reader route batch does not match LM-head batch")

        updated = logits.clone()
        for batch_index, record_index in active.nonzero(as_tuple=False).tolist():
            valid = self.row_ids[record_index] >= 0
            if not bool(valid.any()):
                continue
            ids = self.row_ids[record_index, valid].to(updated.device)
            delta = self.deltas[record_index, valid].to(
                device=hidden.device, dtype=hidden.dtype
            )
            shifts = hidden[batch_index] @ delta.T
            if self.biases is not None:
                bias = self.biases[record_index, valid].to(
                    device=hidden.device, dtype=hidden.dtype
                )
                shifts = shifts + bias.view(1, -1)
            shifts = shifts.to(updated.dtype) * float(self.scale)
            updated[batch_index].index_add_(1, ids, shifts)
            self.fired += 1
        if isinstance(output, tuple):
            return (updated,) + tuple(output[1:])
        return updated

    def close(self) -> None:
        self._handle.remove()


@dataclass
class ScopedSpanEditRuntime:
    router: SpanGateRouter
    writer: SpanGatedWriter
    reader: Optional[ScopedLogitReader]
    metadata: Optional[Dict[str, Any]] = None

    def close(self) -> None:
        if self.reader is not None:
            self.reader.close()
        self.writer.close()
        self.router.close()


def build_sidecar_state(
    *,
    subjects: Sequence[str],
    subject_patterns: Sequence[Sequence[Sequence[int]]],
    writer_layer: int,
    writer_delta: torch.Tensor,
    reader_row_ids: Optional[torch.Tensor] = None,
    reader_deltas: Optional[torch.Tensor] = None,
    reader_biases: Optional[torch.Tensor] = None,
    reader_scale: float = 1.0,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if len(subjects) != len(subject_patterns) or len(subjects) != writer_delta.shape[0]:
        raise ValueError("subjects, patterns, and writer rows must have identical lengths")
    state: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "kind": "exact_subject_scoped_span_edit",
        "subjects": [str(x) for x in subjects],
        "subject_patterns": [
            [[int(token) for token in pattern] for pattern in patterns]
            for patterns in subject_patterns
        ],
        "writer_layer": int(writer_layer),
        "writer_delta": writer_delta.detach().float().cpu(),
        "reader_scale": float(reader_scale),
        "metadata": dict(metadata or {}),
    }
    if (
        reader_row_ids is not None
        or reader_deltas is not None
        or reader_biases is not None
    ):
        if reader_row_ids is None or reader_deltas is None:
            raise ValueError("reader_row_ids and reader_deltas must be supplied together")
        state["reader_row_ids"] = reader_row_ids.detach().long().cpu()
        state["reader_deltas"] = reader_deltas.detach().float().cpu()
        if reader_biases is not None:
            if tuple(reader_biases.shape) != tuple(reader_row_ids.shape):
                raise ValueError(
                    "reader_biases must have the same [records, rows] shape "
                    "as reader_row_ids"
                )
            state["reader_biases"] = reader_biases.detach().float().cpu()
    return state


def save_sidecar(path: Path | str, state: Dict[str, Any]) -> Path:
    destination = Path(path)
    if destination.is_dir() or destination.suffix == "":
        destination = destination / SIDECAR_NAME
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, destination)
    return destination


def load_sidecar(path: Path | str) -> Dict[str, Any]:
    source = Path(path)
    if source.is_dir():
        source = source / SIDECAR_NAME
    state = torch.load(source, map_location="cpu", weights_only=False)
    if int(state.get("schema_version", -1)) != SCHEMA_VERSION:
        raise RuntimeError(
            f"unsupported scoped edit schema {state.get('schema_version')!r}; "
            f"expected {SCHEMA_VERSION}"
        )
    if state.get("kind") != "exact_subject_scoped_span_edit":
        raise RuntimeError(f"not a scoped span edit sidecar: {source}")
    return state


def attach_scoped_span_edit(model: nn.Module, state: Dict[str, Any]) -> ScopedSpanEditRuntime:
    layers = find_decoder_layers(model)
    layer_index = int(state["writer_layer"])
    if layer_index < 0 or layer_index >= len(layers):
        raise ValueError(
            f"writer layer {layer_index} outside decoder range [0, {len(layers) - 1}]"
        )
    embedding = model.get_input_embeddings()
    output_layer = model.get_output_embeddings()
    if embedding is None or output_layer is None:
        raise RuntimeError("scoped edit requires input and output embedding modules")
    writer_delta = state["writer_delta"].float()
    hidden_size = int(embedding.weight.shape[1])
    if writer_delta.ndim != 2 or writer_delta.shape[1] != hidden_size:
        raise ValueError("writer sidecar hidden size does not match model")
    if len(state["subject_patterns"]) != writer_delta.shape[0]:
        raise ValueError("writer sidecar record count is inconsistent")

    layer_device = next(layers[layer_index].parameters()).device
    router = SpanGateRouter(
        embedding,
        state["subject_patterns"],
        subjects=state.get("subjects"),
        model=model,
    )
    writer = SpanGatedWriter(layers[layer_index], router, hidden_size, layer_device)
    with torch.no_grad():
        writer.delta.copy_(writer_delta.to(writer.delta.device))

    reader: Optional[ScopedLogitReader] = None
    if "reader_row_ids" in state or "reader_deltas" in state:
        reader = ScopedLogitReader(
            output_layer,
            router,
            state["reader_row_ids"],
            state["reader_deltas"].to(output_layer.weight.device),
            (
                state["reader_biases"].to(output_layer.weight.device)
                if "reader_biases" in state
                else None
            ),
            float(state.get("reader_scale", 1.0)),
        )
    return ScopedSpanEditRuntime(
        router=router,
        writer=writer,
        reader=reader,
        metadata=dict(state.get("metadata") or {}),
    )


def load_and_attach_scoped_span_edit(
    model: nn.Module, path: Path | str
) -> ScopedSpanEditRuntime:
    return attach_scoped_span_edit(model, load_sidecar(path))


def maybe_attach_scoped_span_edit(
    model: nn.Module, model_dir: Path | str
) -> Optional[ScopedSpanEditRuntime]:
    sidecar = Path(model_dir) / SIDECAR_NAME
    if not sidecar.exists():
        return None
    return load_and_attach_scoped_span_edit(model, sidecar)
