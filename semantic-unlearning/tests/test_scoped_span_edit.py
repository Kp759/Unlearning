from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import scoped_span_edit as scoped


class TupleBlock(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        with torch.no_grad():
            self.proj.weight.copy_(torch.eye(hidden_size))

    def forward(self, hidden):
        return (self.proj(hidden), "cache")


class ToyBackbone(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.layers = nn.ModuleList([TupleBlock(hidden_size), TupleBlock(hidden_size)])


class ToyLM(nn.Module):
    def __init__(self, vocab_size: int = 12, hidden_size: int = 3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.model = ToyBackbone(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        with torch.no_grad():
            values = torch.arange(vocab_size * hidden_size, dtype=torch.float32)
            self.embed.weight.copy_(values.reshape(vocab_size, hidden_size) / 10.0)
            head = torch.arange(vocab_size * hidden_size, dtype=torch.float32)
            self.lm_head.weight.copy_(head.flip(0).reshape(vocab_size, hidden_size) / 20.0)

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids=None, inputs_embeds=None):
        hidden = self.embed(input_ids) if inputs_embeds is None else inputs_embeds
        for layer in self.model.layers:
            hidden = layer(hidden)[0]
        return SimpleNamespace(logits=self.lm_head(hidden), hidden=hidden)


def _runtime(model: ToyLM):
    router = scoped.SpanGateRouter(model.embed, [[[2, 3]], [[5, 6]]])
    writer = scoped.SpanGatedWriter(model.model.layers[0], router, 3, torch.device("cpu"))
    row_ids = torch.tensor([[7, -1], [8, -1]], dtype=torch.long)
    deltas = torch.zeros((2, 2, 3), dtype=torch.float32)
    deltas[0, 0] = torch.tensor([0.5, 0.0, 0.0])
    deltas[1, 0] = torch.tensor([0.0, 0.25, 0.0])
    reader = scoped.ScopedLogitReader(model.lm_head, router, row_ids, deltas, scale=2.0)
    return scoped.ScopedSpanEditRuntime(router, writer, reader)


def test_closed_gate_is_bit_identical_through_writer_and_reader():
    model = ToyLM()
    # Token 3 is shared with the scoped subject [2, 3], but the complete
    # subject is absent. This is the "de in a different name" failure mode.
    inputs = torch.tensor([[0, 4, 3, 9]], dtype=torch.long)
    base = model(inputs)
    runtime = _runtime(model)
    try:
        with torch.no_grad():
            runtime.writer.delta[0] = torch.tensor([3.0, 4.0, 5.0])
            runtime.writer.delta[1] = torch.tensor([-2.0, 1.0, 7.0])
        edited = model(inputs)
        assert torch.equal(edited.hidden, base.hidden)
        assert torch.equal(edited.logits, base.logits)
        assert runtime.writer.fired == 0
        assert runtime.reader is not None and runtime.reader.fired == 0
    finally:
        runtime.close()


def test_open_gate_moves_only_subject_positions_and_scoped_logit_row():
    model = ToyLM()
    inputs = torch.tensor([[0, 2, 3, 9]], dtype=torch.long)
    base = model(inputs)
    runtime = _runtime(model)
    try:
        with torch.no_grad():
            runtime.writer.delta[0] = torch.tensor([1.0, -2.0, 3.0])
        edited = model(inputs)

        expected_hidden = base.hidden.clone()
        expected_hidden[0, 1:3] += runtime.writer.delta[0].detach()
        torch.testing.assert_close(edited.hidden, expected_hidden)

        # The writer naturally changes all base-head logits at its two span
        # positions.  The scoped reader adds one extra correction only to row 7.
        # Use the raw weight so constructing the expectation does not itself
        # invoke the installed reader hook a second time.
        expected_logits = torch.nn.functional.linear(
            expected_hidden, model.lm_head.weight
        )
        expected_logits[0, :, 7] += 2.0 * 0.5 * expected_hidden[0, :, 0]
        torch.testing.assert_close(edited.logits, expected_logits)
        torch.testing.assert_close(
            edited.logits[..., :7], expected_logits[..., :7]
        )
        assert runtime.writer.fired == 1
        assert runtime.reader is not None and runtime.reader.fired == 1
    finally:
        runtime.close()


def test_batch_routes_each_record_independently_and_keeps_negative_row_exact():
    model = ToyLM()
    inputs = torch.tensor(
        [[0, 2, 3, 9], [0, 5, 6, 9], [0, 4, 4, 9]], dtype=torch.long
    )
    base = model(inputs)
    runtime = _runtime(model)
    try:
        with torch.no_grad():
            runtime.writer.delta[0] = torch.tensor([1.0, 0.0, 0.0])
            runtime.writer.delta[1] = torch.tensor([0.0, 2.0, 0.0])
        edited = model(inputs)
        torch.testing.assert_close(
            edited.hidden[0, 1:3] - base.hidden[0, 1:3],
            runtime.writer.delta[0].detach().expand(2, -1),
        )
        torch.testing.assert_close(
            edited.hidden[1, 1:3] - base.hidden[1, 1:3],
            runtime.writer.delta[1].detach().expand(2, -1),
        )
        assert torch.equal(edited.hidden[2], base.hidden[2])
        assert torch.equal(edited.logits[2], base.logits[2])
        assert runtime.router.state is not None
        assert runtime.router.state.active.tolist() == [
            [True, False], [False, True], [False, False]
        ]
    finally:
        runtime.close()


def test_scoped_reader_bias_is_context_invariant_inside_scope():
    model = ToyLM()
    inputs = torch.tensor([[0, 2, 3, 9], [0, 4, 4, 9]], dtype=torch.long)
    base = model(inputs)
    router = scoped.SpanGateRouter(model.embed, [[[2, 3]]])
    writer = scoped.SpanGatedWriter(
        model.model.layers[0], router, 3, torch.device("cpu")
    )
    row_ids = torch.tensor([[7]], dtype=torch.long)
    deltas = torch.zeros((1, 1, 3), dtype=torch.float32)
    biases = torch.tensor([[-3.0]], dtype=torch.float32)
    reader = scoped.ScopedLogitReader(
        model.lm_head,
        router,
        row_ids,
        deltas,
        biases=biases,
        scale=2.0,
    )
    runtime = scoped.ScopedSpanEditRuntime(router, writer, reader)
    try:
        edited = model(inputs)
        expected = base.logits.clone()
        expected[0, :, 7] -= 6.0
        torch.testing.assert_close(edited.logits, expected)
        assert torch.equal(edited.logits[1], base.logits[1])
    finally:
        runtime.close()


def test_writer_gradient_reaches_only_the_routed_record():
    model = ToyLM()
    runtime = _runtime(model)
    try:
        assert runtime.reader is not None
        runtime.reader.enabled = False
        out = model(torch.tensor([[0, 2, 3, 9]], dtype=torch.long))
        out.hidden.sum().backward()
        assert runtime.writer.delta.grad is not None
        torch.testing.assert_close(
            runtime.writer.delta.grad[0], torch.full((3,), 2.0)
        )
        torch.testing.assert_close(
            runtime.writer.delta.grad[1], torch.zeros(3)
        )
    finally:
        runtime.close()


def test_router_prefers_longest_complete_subject_and_keeps_bos_alignment():
    model = ToyLM()
    # Record 0 is the nested short name; record 1 is the complete longer name.
    router = scoped.SpanGateRouter(model.embed, [[[3]], [[2, 3]]])
    try:
        state = router.route(torch.tensor([[11, 2, 3, 9]], dtype=torch.long))
        assert state.active.tolist() == [[False, True]]
        assert state.span_masks[0, 1].nonzero().flatten().tolist() == [1, 2]
    finally:
        router.close()


def test_top_level_pre_hook_clears_stale_route_for_inputs_embeds():
    model = ToyLM()
    router = scoped.SpanGateRouter(model.embed, [[[2, 3]]], model=model)
    writer = scoped.SpanGatedWriter(model.model.layers[0], router, 3, torch.device("cpu"))
    with torch.no_grad():
        writer.delta[0] = torch.tensor([5.0, 5.0, 5.0])
    runtime = scoped.ScopedSpanEditRuntime(router, writer, None)
    try:
        model(torch.tensor([[0, 2, 3, 9]], dtype=torch.long))
        raw = model.embed.weight.detach()[torch.tensor([[0, 4, 4, 9]])]
        expected = raw.clone()
        writer.enabled = False
        for layer in model.model.layers:
            expected = layer(expected)[0]
        writer.enabled = True
        actual = model(inputs_embeds=raw).hidden
        assert torch.equal(actual, expected)
    finally:
        runtime.close()


def test_sidecar_round_trip_reinstalls_identical_runtime(tmp_path):
    model = ToyLM()
    runtime = _runtime(model)
    try:
        with torch.no_grad():
            runtime.writer.delta[0] = torch.tensor([1.0, 2.0, 3.0])
            runtime.writer.delta[1] = torch.tensor([-1.0, 0.5, 2.0])
        assert runtime.reader is not None
        runtime.reader.biases = torch.tensor(
            [[-0.75, 0.0], [-1.25, 0.0]], dtype=torch.float32
        )
        state = scoped.build_sidecar_state(
            subjects=["first", "second"],
            subject_patterns=runtime.router.subject_patterns,
            writer_layer=0,
            writer_delta=runtime.writer.delta,
            reader_row_ids=runtime.reader.row_ids,
            reader_deltas=runtime.reader.deltas,
            reader_biases=runtime.reader.biases,
            reader_scale=runtime.reader.scale,
        )
        path = scoped.save_sidecar(tmp_path, state)
        expected_active = model(torch.tensor([[0, 2, 3, 9]], dtype=torch.long))
        expected_closed = model(torch.tensor([[0, 4, 4, 9]], dtype=torch.long))
    finally:
        runtime.close()

    reloaded = scoped.load_and_attach_scoped_span_edit(model, path)
    try:
        actual_active = model(torch.tensor([[0, 2, 3, 9]], dtype=torch.long))
        actual_closed = model(torch.tensor([[0, 4, 4, 9]], dtype=torch.long))
        torch.testing.assert_close(actual_active.hidden, expected_active.hidden)
        torch.testing.assert_close(actual_active.logits, expected_active.logits)
        assert torch.equal(actual_closed.hidden, expected_closed.hidden)
        assert torch.equal(actual_closed.logits, expected_closed.logits)
    finally:
        reloaded.close()
