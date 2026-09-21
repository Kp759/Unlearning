"""Oracle and subject-only routing arms for the decomposition measurements.

Every failure number currently reported mixes two causes: the router did not
fire, or it fired and the residual did not generalize. These need opposite
fixes, so until they are separated no result points at a next step. These bank
variants supply the reference arms that separate them.

  OracleAssociationBank      ground-truth association supplied per prompt.
                             Routing is perfect by construction, so whatever
                             remains is the actuator's ceiling (G1). The gap
                             between this and Router V2 is the total value of
                             all router work -- if it is small, most router
                             effort is unnecessary and that is worth knowing
                             on day one.

  SubjectOnlyAssociationBank subject eligibility with no context confirmation,
                             i.e. Router V1's unique-subject bypass applied
                             uniformly. Sits between V2 and the oracle and
                             isolates what the context test costs on forget
                             prompts and buys on retain prompts. This is the
                             cheapest explanation for why V2 is worse than V1
                             on every forgetting metric.

  RandomRouterBank           routes uniformly among eligible candidates. The
                             floor: suppression that survives random routing
                             is relation-generic rather than fact-specific,
                             which would say something uncomfortable about
                             what the residuals actually learned. An oracle
                             without a matching floor is an unanchored number.

All three keep the frozen backbone, the single-position intervention at the
request boundary, and the trained rows untouched. Only selection changes, so
they drop into the existing evaluator unmodified.

Oracle lookup
-------------
The evaluator does not pass association labels down to the bank, so the oracle
carries a table keyed by the hash of the prompt-prefix token sequence, built
ahead of time from the evaluation records. A prompt absent from the table
routes to nothing, which is the correct oracle behaviour on retain,
neighborhood and utility sets: ground truth there says "not a protected
association". That makes the oracle arm a genuine ceiling on BOTH axes --
perfect suppression and perfect locality -- rather than a forget-only bound.
"""
from __future__ import annotations

from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F

from static_overlap_fact_association_embeddings import (
    AssociationCausalLM,
    _contains_subsequence,
)


def prompt_key(token_ids):
    """Stable key for a prompt-prefix token sequence."""
    return hash(tuple(int(t) for t in token_ids))


def build_oracle_table(tokenizer, prompt_to_fact_index, add_special_tokens=False):
    """Map prompt text -> gold association index, keyed by token sequence.

    Keying on tokens rather than text means the lookup matches exactly what
    the hook sees, including whitespace and chat-template differences.
    """
    table = {}
    collisions = []
    for prompt, fact_index in prompt_to_fact_index.items():
        ids = tokenizer(
            str(prompt), add_special_tokens=add_special_tokens
        )["input_ids"]
        key = prompt_key(ids)
        if key in table and table[key] != int(fact_index):
            collisions.append({
                "prompt": str(prompt),
                "existing": table[key],
                "incoming": int(fact_index),
            })
            continue
        table[key] = int(fact_index)
    return table, collisions


class _RoutingBankBase(nn.Module):
    """Shared plumbing: frozen rows, boundary resolution, subject eligibility."""

    def __init__(self, base_model, layer, rows, subject_patterns, facts):
        super().__init__()
        device = next(base_model.parameters()).device
        dtype = next(base_model.parameters()).dtype
        initial = rows.to(device=device, dtype=dtype)
        if initial.shape[0] != len(facts):
            raise ValueError("Row count must match the fact list")
        self.rows = nn.ParameterList(
            nn.Parameter(row.clone(), requires_grad=False) for row in initial
        )
        self.layer = int(layer)
        self.subject_patterns = subject_patterns
        self.facts = list(facts)
        self._input_ids = None
        self._attention_mask = None
        self._prefix_lengths = None
        self.calls = 0
        self.active_batch_rows = 0
        self.active_token_positions = 0
        self.active_fact_counts = [0 for _ in facts]
        self.last_active_fact_indices = []
        self.last_route_scores = []
        self._hook_handle = base_model.model.layers[self.layer].register_forward_hook(
            self._hook
        )

    @property
    def extra(self):
        return torch.stack(list(self.rows))

    def bind(self, input_ids, attention_mask=None, prefix_lengths=None):
        self._input_ids = input_ids
        self._attention_mask = attention_mask
        self._prefix_lengths = prefix_lengths

    def unbind(self):
        self._input_ids = None
        self._attention_mask = None
        self._prefix_lengths = None

    def _prefix_lengths_for(self, hidden):
        batch, width, _ = hidden.shape
        if self._prefix_lengths is not None:
            lengths = self._prefix_lengths.to(hidden.device, dtype=torch.long)
        elif self._attention_mask is not None:
            mask = self._attention_mask.to(hidden.device).bool()
            lengths = (
                torch.arange(mask.shape[1], device=hidden.device)[None, :]
                .expand_as(mask)
                .masked_fill(~mask, -1)
                .max(dim=1)
                .values + 1
            )
        else:
            lengths = torch.full(
                (batch,), width, device=hidden.device, dtype=torch.long
            )
        if bool(((lengths <= 0) | (lengths > width)).any()):
            raise ValueError("Association boundary outside the input sequence")
        return lengths

    def _prompt_tokens(self, batch_index, boundary):
        tokens = self._input_ids[batch_index].detach().cpu().tolist()
        if self._attention_mask is None:
            return tokens[:int(boundary)]
        attention = self._attention_mask[batch_index].detach().cpu().bool().tolist()
        return [
            token for position, token in enumerate(tokens)
            if position < int(boundary) and attention[position]
        ]

    def _eligible(self, prompt_tokens):
        return [
            index
            for index, patterns in enumerate(self.subject_patterns)
            if any(_contains_subsequence(prompt_tokens, p) for p in patterns)
        ]

    def select(self, batch_index, prompt_tokens):
        """Return the chosen association index, or None to abstain."""
        raise NotImplementedError

    def _hook(self, module, args, output):
        if self._input_ids is None:
            raise RuntimeError("Routing bank hook fired without bound input_ids")
        hidden = output[0] if isinstance(output, tuple) else output
        batch, width, _ = hidden.shape
        prefix_lengths = self._prefix_lengths_for(hidden)
        positions = prefix_lengths - 1

        chosen, active = [], []
        for index in range(batch):
            prompt_tokens = self._prompt_tokens(index, int(prefix_lengths[index]))
            selection = self.select(index, prompt_tokens)
            active.append(selection is not None)
            chosen.append(int(selection) if selection is not None else 0)

        chosen_tensor = torch.tensor(chosen, device=hidden.device, dtype=torch.long)
        active_tensor = torch.tensor(active, device=hidden.device, dtype=torch.bool)
        rows = self.extra.to(device=hidden.device, dtype=hidden.dtype)
        selected = F.embedding(chosen_tensor, rows)
        position_mask = F.one_hot(positions, num_classes=width).to(hidden.dtype)
        delta = (
            position_mask.unsqueeze(-1)
            * selected.unsqueeze(1)
            * active_tensor[:, None, None].to(hidden.dtype)
        )
        edited = hidden + delta

        self.calls += 1
        with torch.no_grad():
            self.active_batch_rows += int(active_tensor.sum())
            self.active_token_positions += int(active_tensor.sum())
            self.last_active_fact_indices = [
                [chosen[i]] if active[i] else [] for i in range(batch)
            ]
            self.last_route_scores = [
                {
                    "fact_index": chosen[i] if active[i] else None,
                    "routing_policy": self.policy,
                }
                for i in range(batch)
            ]
            for index in range(batch):
                if active[index]:
                    self.active_fact_counts[chosen[index]] += 1

        if isinstance(output, tuple):
            return (edited, *output[1:])
        return edited

    def counters(self):
        return {
            "hook_calls": self.calls,
            "active_batch_rows": self.active_batch_rows,
            "active_token_positions": self.active_token_positions,
            "active_fact_counts": list(self.active_fact_counts),
        }

    def artifact(self):
        return {
            "architecture": f"{self.policy}_fact_association_bank",
            "layer": self.layer,
            "rows": self.extra.detach().cpu(),
            "subject_patterns": self.subject_patterns,
            "facts": self.facts,
            "routing_policy": self.policy,
            "trainable_parameters": 0,
            "base_parameters_trainable": 0,
            "diagnostic_arm_not_a_deployable_router": True,
        }

    def close(self):
        self._hook_handle.remove()


class OracleAssociationBank(_RoutingBankBase):
    """Ground-truth routing. The actuator's ceiling (G1)."""

    policy = "oracle_ground_truth"

    def __init__(self, base_model, layer, rows, subject_patterns, facts,
                 oracle_table, require_subject_eligibility=False):
        super().__init__(base_model, layer, rows, subject_patterns, facts)
        self.oracle_table = dict(oracle_table)
        # Off by default: an oracle constrained by the lexical gate is not an
        # oracle, it is Router V2's Stage A with a perfect Stage B, and it
        # would silently inherit the P1 ceiling this arm exists to measure past.
        self.require_subject_eligibility = bool(require_subject_eligibility)
        self.lookup_hits = 0
        self.lookup_misses = 0

    def select(self, batch_index, prompt_tokens):
        gold = self.oracle_table.get(prompt_key(prompt_tokens))
        if gold is None:
            self.lookup_misses += 1
            return None
        self.lookup_hits += 1
        if self.require_subject_eligibility and gold not in self._eligible(prompt_tokens):
            return None
        return gold

    def counters(self):
        record = super().counters()
        record.update({
            "oracle_lookup_hits": self.lookup_hits,
            "oracle_lookup_misses": self.lookup_misses,
        })
        return record


class SubjectOnlyAssociationBank(_RoutingBankBase):
    """Subject eligibility, no context confirmation (Router V1's bypass)."""

    policy = "subject_eligibility_only"

    def __init__(self, base_model, layer, rows, subject_patterns, facts,
                 ambiguous="abstain"):
        super().__init__(base_model, layer, rows, subject_patterns, facts)
        if ambiguous not in ("abstain", "first"):
            raise ValueError("ambiguous must be 'abstain' or 'first'")
        self.ambiguous = ambiguous

    def select(self, batch_index, prompt_tokens):
        eligible = self._eligible(prompt_tokens)
        if not eligible:
            return None
        if len(eligible) == 1:
            return eligible[0]
        return eligible[0] if self.ambiguous == "first" else None


class RandomRouterBank(_RoutingBankBase):
    """Uniform choice among eligible candidates. The interpretive floor."""

    policy = "random_among_eligible"

    def __init__(self, base_model, layer, rows, subject_patterns, facts, seed=0):
        super().__init__(base_model, layer, rows, subject_patterns, facts)
        self.generator = torch.Generator().manual_seed(int(seed))

    def select(self, batch_index, prompt_tokens):
        eligible = self._eligible(prompt_tokens)
        if not eligible:
            return None
        draw = int(
            torch.randint(
                len(eligible), (1,), generator=self.generator
            ).item()
        )
        return eligible[draw]


class ForcedRowBank(_RoutingBankBase):
    """Route to whatever row the caller sets, independent of the prompt.

    `forced_row` is None (abstain), an int (every prompt in the batch), or a
    list with one entry per batch row. This is the primitive behind genies
    whose ground truth is not a single prompt->row mapping -- for example the
    RWKU subject genie, which tries each of a person's trained rows on a
    held-out probe that has no row of its own and keeps the best one. A prompt
    hash table cannot express "try row k" without colliding.
    """

    policy = "forced_row"

    def __init__(self, base_model, layer, rows, subject_patterns, facts):
        super().__init__(base_model, layer, rows, subject_patterns, facts)
        self.forced_row = None

    def select(self, batch_index, prompt_tokens):
        forced = self.forced_row
        if isinstance(forced, (list, tuple)):
            forced = forced[batch_index]
        if forced is None:
            return None
        forced = int(forced)
        if not 0 <= forced < len(self.facts):
            raise IndexError(f"Forced row {forced} outside the bank")
        return forced


ARMS = {
    "oracle": OracleAssociationBank,
    "subject_only": SubjectOnlyAssociationBank,
    "random": RandomRouterBank,
    "forced": ForcedRowBank,
}


def load_arm(base_model, artifact, arm, oracle_table=None, **kwargs):
    """Wrap a trained artifact in one of the diagnostic routing arms."""
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {sorted(ARMS)}")
    common = dict(
        base_model=base_model,
        layer=int(artifact["layer"]),
        rows=artifact["rows"],
        subject_patterns=artifact["subject_patterns"],
        facts=artifact["facts"],
    )
    if arm == "oracle":
        if oracle_table is None:
            raise ValueError("The oracle arm requires an oracle table")
        bank = OracleAssociationBank(oracle_table=oracle_table, **common, **kwargs)
    else:
        bank = ARMS[arm](**common, **kwargs)
    return AssociationCausalLM(base_model, bank), bank
