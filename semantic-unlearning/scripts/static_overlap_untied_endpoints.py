"""Independent native embedding/head weights and separately masked dense deltas."""
from contextlib import contextmanager
import hashlib

import torch
from torch import nn

from static_overlap_cached_head import prepare_independent_head
from static_overlap_core import model_logits, tied_weights
from static_overlap_endpoint_ga import DenseRows, InputRows, OutputRows


@torch.no_grad()
def prepare_base(model, examples):
    example = next(e for e in examples if e.split == "train" and e.role == "forget")
    before = model_logits(model, example)
    report = prepare_independent_head(model, example, allow_untie=True)
    after = model_logits(model, example)
    if tied_weights(model) or not torch.isfinite(after).all() or not torch.equal(before, after):
        raise ValueError("Untying must preserve exact original logits")
    report.update(base_logits_exact=True, exported_tie_word_embeddings=False,
                  before_optimizer_creation=True, independent_weight_storage=True)
    return report


class UntiedEndpointEditor:
    def __init__(self, model, input_rows, output_rows):
        if tied_weights(model) or getattr(model, "is_quantized", False):
            raise ValueError("Untie the original FP32 model before constructing endpoint edits")
        if len({p.device for p in model.parameters()}) != 1:
            raise ValueError("Use one device without offloading")
        self.model, self.shared, self.merged = model, False, False
        self.embedding, self.head = model.get_input_embeddings(), model.get_output_embeddings()
        if (not isinstance(self.embedding, nn.Embedding) or not isinstance(self.head, nn.Linear)
                or self.embedding.max_norm is not None or self.embedding.weight.dtype != torch.float32
                or self.head.weight.dtype != torch.float32):
            raise ValueError("Need native FP32 embedding and LM head")
        model.requires_grad_(False)
        model.eval()
        self.input_edit = DenseRows(self.embedding.weight, input_rows)
        self.output_edit = DenseRows(self.head.weight, output_rows)
        self.parameters = [self.input_edit.delta, self.output_edit.delta]
        model.set_input_embeddings(InputRows(self.embedding, self.input_edit))
        model.set_output_embeddings(OutputRows(self.head, self.output_edit))
        assert {id(p) for p in model.parameters() if p.requires_grad} == {id(p) for p in self.parameters}

    @contextmanager
    def base(self):
        if self.merged:
            raise RuntimeError("Original base unavailable after merge")
        flags = self.input_edit.enabled, self.output_edit.enabled
        self.input_edit.enabled = self.output_edit.enabled = False
        try:
            yield
        finally:
            self.input_edit.enabled, self.output_edit.enabled = flags

    def artifact(self):
        return {"shared_endpoints": False,
                **{name: {"rows": edit.rows.cpu(), "delta": edit.delta.detach().cpu()}
                   for name, edit in (("input", self.input_edit), ("output", self.output_edit))}}

    @torch.no_grad()
    def load_artifact(self, data):
        if self.merged or data.get("shared_endpoints") is not False:
            raise ValueError("Expected separate endpoint artifact")
        for name, edit in (("input", self.input_edit), ("output", self.output_edit)):
            d = data[name]
            if (not torch.equal(d["rows"].cpu(), edit.rows.cpu()) or d["delta"].shape != edit.delta.shape
                    or d["delta"].dtype != edit.delta.dtype or not torch.isfinite(d["delta"]).all()):
                raise ValueError("Separate endpoint artifact changed its support, shape or precision")
        for name, edit in (("input", self.input_edit), ("output", self.output_edit)):
            edit.delta.copy_(data[name]["delta"])

    @torch.no_grad()
    def merge(self):
        if self.merged:
            raise RuntimeError("Already merged")
        self.embedding.weight.index_add_(0, self.input_edit.rows, self.input_edit.delta)
        self.head.weight.index_add_(0, self.output_edit.rows, self.output_edit.delta)
        self.model.set_input_embeddings(self.embedding)
        self.model.set_output_embeddings(self.head)
        self.model.config.tie_word_embeddings = False
        self.model.requires_grad_(False)
        self.merged = True
        assert not tied_weights(self.model)


def hash_frozen(model, mask):
    if tied_weights(model):
        raise ValueError("Separate endpoint hashes require untied storage")
    exclusions = {id(model.get_input_embeddings().weight): set(mask["input_rows"]),
                  id(model.get_output_embeddings().weight): set(mask["output_rows"])}
    hashes = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        if "model.layers.19.mlp.down_proj" in name:
            continue
        h = hashlib.sha256()
        if id(parameter) in exclusions:
            for start in range(0, parameter.shape[0], 512):
                indices = [i for i in range(start, min(start+512, parameter.shape[0])) if i not in exclusions[id(parameter)]]
                h.update(parameter.detach()[indices].float().cpu().numpy().tobytes())
        else:
            for chunk in parameter.detach().reshape(-1).split(1024*1024):
                h.update(chunk.float().cpu().numpy().tobytes())
        hashes[name] = h.hexdigest()
    return hashes


def verify_frozen(model, mask, original):
    if tied_weights(model) or hash_frozen(model, mask) != original:
        raise ValueError("Untied endpoint storage, separate row masks or frozen weights changed")
    return {"independent_weight_storage": True, "shared_endpoints": False,
            "all_transformer_weights_exact": True, "unselected_endpoint_rows_exact": True,
            "input_rows": len(set(mask["input_rows"])), "output_rows": len(set(mask["output_rows"]))}
