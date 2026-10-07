"""Compressed residual banks trained in the loop (not post hoc).

The N residual rows are produced by a small module from compact parameters,
and those parameters are what the unlearning objective optimizes. Shared
pieces (a basis, per-answer or per-relation directions, a map from the answer
embedding) are therefore learned for forgetting, not fitted to rows afterwards.

Value modes (d = hidden size, N facts):

  full                 P[N, d]                       per fact: d       (reference)
  shared               v[d], identical for every fact per fact: 0      shared: d
  lowrank:K            codes[N, K] @ basis[K, d]     per fact: K       shared: K*d
  tied_answer          s[N] * D[answer(i)]           per fact: 1       shared: A*d
  tied_relation        s[N] * D[relation(i)]         per fact: 1       shared: R*d
  answer_fixed         s[N] * u(answer(i))           per fact: 1       shared: 0
  answer_map:r         s[N] * (u + u A B)            per fact: 1       shared: 2*d*r
  relation_plus_answer a[N] * D[relation(i)] + b[N] * u(answer(i))
                                                     per fact: 2       shared: R*d

u(answer) = -normalize(E_out[t]) for the first answer token t: the direction
that lowers t's logit under the logit lens. It is fixed (a function of the
frozen model), so answer_fixed stores one scalar per fact and nothing shared.

Per-fact group ids (answer, relation) and answer token ids are data the bank
already holds (facts, subject patterns); they are counted as small integers.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

from linear_router import LinearClassifierAssociationBank

VALUE_MODES = (
    "full", "shared", "lowrank", "tied_answer", "tied_relation",
    "answer_fixed", "answer_map", "relation_plus_answer",
)


def parse_value_mode(text):
    name, _, arg = str(text).partition(":")
    if name not in VALUE_MODES:
        raise ValueError(f"Unknown value mode {text!r}; choose from {VALUE_MODES}")
    if name in ("lowrank", "answer_map"):
        if not arg:
            raise ValueError(f"{name} needs a rank, e.g. {name}:16")
        return name, int(arg)
    if arg:
        raise ValueError(f"{name} takes no argument")
    return name, None


def _groups(values):
    names, index = [], []
    for value in values:
        if value not in names:
            names.append(value)
        index.append(names.index(value))
    return torch.tensor(index, dtype=torch.long), names


def answer_directions(model, answer_token_ids):
    """-normalize(output embedding of each fact's first answer token)."""
    unembed = model.get_output_embeddings().weight.detach().float()
    ids = torch.as_tensor(answer_token_ids, dtype=torch.long, device=unembed.device)
    return -F.normalize(unembed[ids], dim=-1).cpu()


class CompressedValues(nn.Module):
    """rows() -> [N, d] float32, a differentiable function of compact params."""

    def __init__(self, mode, rank, facts, hidden_size, *, answer_token_ids=None,
                 answer_dirs=None, seed=1):
        super().__init__()
        self.mode, self.rank = mode, rank
        self.n, self.d = len(facts), int(hidden_size)
        answer_index, answer_names = _groups(
            " ".join(str(f["object"]).casefold().split()) for f in facts
        )
        relation_index, relation_names = _groups(str(f.get("relation", "")) for f in facts)
        self.register_buffer("answer_index", answer_index)
        self.register_buffer("relation_index", relation_index)
        self.answer_groups, self.relation_groups = len(answer_names), len(relation_names)
        self.answer_token_ids = None if answer_token_ids is None else list(map(int, answer_token_ids))
        g = torch.Generator().manual_seed(int(seed))
        n, d = self.n, self.d
        if mode in ("answer_fixed", "answer_map", "relation_plus_answer"):
            if answer_dirs is None:
                raise ValueError(f"{mode} needs answer directions")
            self.register_buffer("answer_dirs", answer_dirs.float().clone())
        if mode == "full":
            self.rows_param = nn.Parameter(torch.zeros(n, d))
        elif mode == "shared":
            self.shared_vector = nn.Parameter(torch.zeros(d))
        elif mode == "lowrank":
            k = int(rank)
            basis = torch.linalg.qr(torch.randn(d, k, generator=g))[0].T.contiguous()
            self.basis = nn.Parameter(basis)                  # [K, d], orthonormal init
            self.codes = nn.Parameter(torch.zeros(n, k))
        elif mode in ("tied_answer", "tied_relation"):
            groups = self.answer_groups if mode == "tied_answer" else self.relation_groups
            self.directions = nn.Parameter(torch.zeros(groups, d))
            self.scale = nn.Parameter(torch.ones(n))
        elif mode == "answer_fixed":
            self.scale = nn.Parameter(torch.zeros(n))
        elif mode == "answer_map":
            r = int(rank)
            self.map_a = nn.Parameter(torch.randn(d, r, generator=g) / math.sqrt(d))
            self.map_b = nn.Parameter(torch.zeros(r, d))
            self.scale = nn.Parameter(torch.zeros(n))
        elif mode == "relation_plus_answer":
            self.directions = nn.Parameter(torch.zeros(self.relation_groups, d))
            self.relation_scale = nn.Parameter(torch.ones(n))
            self.answer_scale = nn.Parameter(torch.zeros(n))
        else:
            raise ValueError(mode)

    def rows(self):
        m = self.mode
        if m == "full":
            return self.rows_param
        if m == "shared":
            return self.shared_vector.unsqueeze(0).expand(self.n, -1)
        if m == "lowrank":
            return self.codes @ self.basis
        if m == "tied_answer":
            return self.scale[:, None] * self.directions[self.answer_index]
        if m == "tied_relation":
            return self.scale[:, None] * self.directions[self.relation_index]
        if m == "answer_fixed":
            return self.scale[:, None] * self.answer_dirs
        if m == "answer_map":
            u = self.answer_dirs
            return self.scale[:, None] * (u + (u @ self.map_a) @ self.map_b)
        if m == "relation_plus_answer":
            return (self.relation_scale[:, None] * self.directions[self.relation_index]
                    + self.answer_scale[:, None] * self.answer_dirs)
        raise ValueError(m)

    def storage(self):
        """Stored numbers: per fact vs shared (fixed buffers from the model are free)."""
        n, d, k = self.n, self.d, self.rank
        per_fact_floats, shared_floats, per_fact_ints = {
            "full": (d, 0, 0),
            "shared": (0, d, 0),
            "lowrank": (k, (k or 0) * d, 0),
            "tied_answer": (1, self.answer_groups * d, 1),
            "tied_relation": (1, self.relation_groups * d, 1),
            "answer_fixed": (1, 0, 1),
            "answer_map": (1, 2 * d * (k or 0), 1),
            "relation_plus_answer": (2, self.relation_groups * d, 2),
        }[self.mode]
        total = n * per_fact_floats + shared_floats
        return {
            "mode": self.mode if k is None else f"{self.mode}:{k}",
            "facts": n,
            "per_fact_floats": per_fact_floats,
            "per_fact_small_ints": per_fact_ints,
            "shared_floats": shared_floats,
            "total_floats": total,
            "full_rows_floats": n * d,
            "ratio_to_full_rows": total / (n * d),
            "shared_vectors": shared_floats / d,
            "answer_groups": self.answer_groups,
            "relation_groups": self.relation_groups,
            # Shared size is constant in N for lowrank/answer_map/answer_fixed;
            # for tied_* it grows with distinct answers/relations (unknown here).
            "extrapolated_total_floats": {
                str(size): (
                    size * d if self.mode == "full"
                    else size * per_fact_floats + shared_floats
                    if self.mode in ("shared", "lowrank", "answer_map", "answer_fixed")
                    else None
                )
                for size in (1_000, 10_000, 100_000)
            },
            "extrapolation_note": (
                "tied_* shared sizes grow with the number of distinct answers / "
                "relations, which the N=50 run cannot measure"
            ),
        }

    def compact_state(self):
        return {name: tensor.detach().cpu().clone() for name, tensor in self.state_dict().items()}


class CompressedValueBank(LinearClassifierAssociationBank):
    """The linear-router bank whose rows come from a CompressedValues module."""

    def __init__(self, *args, values: CompressedValues, **kwargs):
        super().__init__(*args, **kwargs)
        for row in self.rows:
            row.requires_grad_(False)
        self.values = values

    @property
    def extra(self):
        dtype = self.rows[0].dtype if len(self.rows) else torch.float32
        return self.values.rows().to(dtype)
