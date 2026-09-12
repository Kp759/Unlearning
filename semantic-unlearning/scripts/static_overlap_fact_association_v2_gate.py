"""Relation-sensitive routing for V2 fact-association embeddings.

This module keeps the V1 one-vector-per-fact, frozen-backbone, one-position
intervention.  It changes only routing: exact subject occurrence defines
candidate facts, while frozen positive/negative relation prototypes determine
whether a candidate is actually in scope.  Unique subjects do not bypass the
relation test.
"""
from __future__ import annotations

from collections import defaultdict
import torch
from torch import nn
from torch.nn import functional as F

from static_overlap_fact_association_embeddings import (
    AssociationCausalLM,
    _contains_subsequence,
    extract_prompt_queries,
    relation_negative_prompts,
)


def _split_nonempty(items, first_fraction=0.5):
    if len(items) < 2:
        raise ValueError("Prototype/calibration split requires at least two items")
    cut = max(1, min(len(items) - 1, int(round(len(items) * first_fraction))))
    return items[:cut], items[cut:]


@torch.no_grad()
def build_relation_prototype_gate(
    model,
    tokenizer,
    facts,
    examples,
    layer,
    relation_negative_count,
    u_slack=0.01,
    d_slack=0.01,
):
    """Build disjoint prototype/calibration banks from training-visible prompts.

    Positive prototype construction and positive threshold calibration use
    disjoint training prompts.  Development prompts remain held out.  Negative
    relation controls are split into prototype, calibration, and audit subsets.
    """
    train_by_fact = defaultdict(list)
    dev_by_fact = defaultdict(list)
    for example in examples:
        target = train_by_fact if example.split == "train" else dev_by_fact
        target[example.fact_id].append(example.prompt)

    positive_prototypes = []
    negative_prototypes = []
    alpha = []
    tau = []
    diagnostics = {"layer": int(layer), "per_fact": []}

    for fact in facts:
        positives = list(train_by_fact[fact["id"]])
        proto_prompts, calibration_prompts = _split_nonempty(positives, 0.5)
        negative_prompts = relation_negative_prompts(
            fact, facts, relation_negative_count
        )
        if len(negative_prompts) < 4:
            raise ValueError("V2 relation gate needs at least four negative controls")
        neg_proto_count = max(1, len(negative_prompts) // 2)
        neg_cal_count = max(1, (len(negative_prompts) - neg_proto_count) // 2)
        neg_proto_prompts = negative_prompts[:neg_proto_count]
        neg_cal_prompts = negative_prompts[
            neg_proto_count:neg_proto_count + neg_cal_count
        ]
        neg_audit_prompts = negative_prompts[
            neg_proto_count + neg_cal_count:
        ]
        if not neg_audit_prompts:
            neg_audit_prompts = list(neg_cal_prompts)

        pos_proto = extract_prompt_queries(
            model, tokenizer, proto_prompts, layer
        ).float()
        neg_proto = extract_prompt_queries(
            model, tokenizer, neg_proto_prompts, layer
        ).float()
        pos_cal = extract_prompt_queries(
            model, tokenizer, calibration_prompts, layer
        ).float()
        neg_cal = extract_prompt_queries(
            model, tokenizer, neg_cal_prompts, layer
        ).float()
        neg_audit = extract_prompt_queries(
            model, tokenizer, neg_audit_prompts, layer
        ).float()
        dev_prompts = list(dev_by_fact[fact["id"]])
        dev = extract_prompt_queries(
            model, tokenizer, dev_prompts, layer
        ).float()

        pos_proto = F.normalize(pos_proto, dim=-1)
        neg_proto = F.normalize(neg_proto, dim=-1)

        def scores(queries):
            q = F.normalize(queries, dim=-1)
            u = (q @ pos_proto.T).max(dim=-1).values
            v = (q @ neg_proto.T).max(dim=-1).values
            return u, u - v

        pos_u, pos_d = scores(pos_cal)
        neg_u, neg_d = scores(neg_cal)
        audit_u, audit_d = scores(neg_audit)
        dev_u, dev_d = scores(dev)

        pos_u_floor = float(pos_u.min())
        pos_d_floor = float(pos_d.min())
        neg_u_ceiling = float(neg_u.max())
        neg_d_ceiling = float(neg_d.max())

        # When finite calibration controls are separable, place the threshold
        # between them. Otherwise preserve calibration-positive recall and
        # expose the overlap in diagnostics rather than bypassing the relation.
        alpha_i = (
            0.5 * (pos_u_floor + neg_u_ceiling)
            if neg_u_ceiling < pos_u_floor
            else pos_u_floor - float(u_slack)
        )
        tau_i = (
            0.5 * (pos_d_floor + neg_d_ceiling)
            if neg_d_ceiling < pos_d_floor
            else pos_d_floor - float(d_slack)
        )
        alpha_i = max(-1.0, min(1.0, alpha_i))
        tau_i = max(-2.0, min(2.0, tau_i))

        def pass_rate(u, d):
            return float(((u >= alpha_i) & (d >= tau_i)).float().mean())

        positive_prototypes.append(pos_proto.cpu())
        negative_prototypes.append(neg_proto.cpu())
        alpha.append(alpha_i)
        tau.append(tau_i)
        diagnostics["per_fact"].append({
            "fact_id": fact["id"],
            "subject": fact["subject"],
            "relation": fact["relation"],
            "prototype_positive_count": len(proto_prompts),
            "calibration_positive_count": len(calibration_prompts),
            "prototype_negative_count": len(neg_proto_prompts),
            "calibration_negative_count": len(neg_cal_prompts),
            "audit_negative_count": len(neg_audit_prompts),
            "positive_u_min": pos_u_floor,
            "positive_d_min": pos_d_floor,
            "negative_u_max": neg_u_ceiling,
            "negative_d_max": neg_d_ceiling,
            "u_separable_on_calibration": neg_u_ceiling < pos_u_floor,
            "d_separable_on_calibration": neg_d_ceiling < pos_d_floor,
            "alpha": alpha_i,
            "tau": tau_i,
            "calibration_positive_pass_fraction": pass_rate(pos_u, pos_d),
            "calibration_negative_fire_fraction": pass_rate(neg_u, neg_d),
            "audit_negative_fire_fraction": pass_rate(audit_u, audit_d),
            "development_positive_pass_fraction": pass_rate(dev_u, dev_d),
        })

    diagnostics.update({
        "gate": (
            "prompt-prefix subject eligibility AND positive prototype similarity "
            "AND positive-minus-negative relation margin"
        ),
        "unique_subject_bypass": False,
        "official_paraphrases_used": False,
        "official_neighborhoods_used": False,
        "development_used_for_prototypes_or_thresholds": False,
        "mean_calibration_negative_fire_fraction": sum(
            row["calibration_negative_fire_fraction"]
            for row in diagnostics["per_fact"]
        ) / len(facts),
        "mean_audit_negative_fire_fraction": sum(
            row["audit_negative_fire_fraction"]
            for row in diagnostics["per_fact"]
        ) / len(facts),
        "mean_development_positive_pass_fraction": sum(
            row["development_positive_pass_fraction"]
            for row in diagnostics["per_fact"]
        ) / len(facts),
    })
    return (
        positive_prototypes,
        negative_prototypes,
        torch.tensor(alpha, dtype=torch.float32),
        torch.tensor(tau, dtype=torch.float32),
        diagnostics,
    )


class RelationPrototypeAssociationBank(nn.Module):
    """One vector per fact with relation-sensitive prototype routing."""

    def __init__(
        self,
        base_model,
        layer,
        positive_prototypes,
        negative_prototypes,
        alpha,
        tau,
        subject_patterns,
        facts,
        rows=None,
    ):
        super().__init__()
        if len(facts) != len(positive_prototypes) or len(facts) != len(negative_prototypes):
            raise ValueError("Prototype banks must match facts")
        if tuple(alpha.shape) != (len(facts),) or tuple(tau.shape) != (len(facts),):
            raise ValueError("Relation thresholds must match facts")
        hidden_size = int(positive_prototypes[0].shape[-1])
        if any(int(x.shape[-1]) != hidden_size for x in positive_prototypes):
            raise ValueError("Positive prototype dimensions differ")
        if any(int(x.shape[-1]) != hidden_size for x in negative_prototypes):
            raise ValueError("Negative prototype dimensions differ")
        device = next(base_model.parameters()).device
        dtype = next(base_model.parameters()).dtype
        if rows is None:
            initial = torch.zeros(
                (len(facts), hidden_size), device=device, dtype=dtype
            )
        else:
            initial = rows.to(device=device, dtype=dtype)
            if tuple(initial.shape) != (len(facts), hidden_size):
                raise ValueError("Saved V2 rows have wrong shape")
        self.rows = nn.ParameterList(nn.Parameter(row.clone()) for row in initial)
        self.positive_prototypes = [
            F.normalize(x.detach().float().cpu(), dim=-1)
            for x in positive_prototypes
        ]
        self.negative_prototypes = [
            F.normalize(x.detach().float().cpu(), dim=-1)
            for x in negative_prototypes
        ]
        self.register_buffer("alpha", alpha.detach().float().clone())
        self.register_buffer("tau", tau.detach().float().clone())
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
        layer_module = base_model.model.layers[self.layer]
        self._hook_handle = layer_module.register_forward_hook(self._hook)

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
        if tuple(lengths.shape) != (batch,):
            raise ValueError("V2 prefix lengths must have one value per sequence")
        if bool(((lengths <= 0) | (lengths > width)).any()):
            raise ValueError("V2 association boundary outside input")
        return lengths

    def _subject_mask(self, input_ids, prefix_lengths, attention_mask=None):
        rows = input_ids.detach().cpu().tolist()
        prefixes = prefix_lengths.detach().cpu().tolist()
        attention = (
            attention_mask.detach().cpu().bool().tolist()
            if attention_mask is not None else None
        )
        mask = torch.zeros(
            (len(rows), len(self.facts)),
            dtype=torch.bool,
            device=input_ids.device,
        )
        for batch_index, (tokens, boundary) in enumerate(zip(rows, prefixes)):
            if attention is None:
                prompt_tokens = tokens[:int(boundary)]
            else:
                prompt_tokens = [
                    token for position, token in enumerate(tokens)
                    if position < int(boundary) and attention[batch_index][position]
                ]
            for fact_index, patterns in enumerate(self.subject_patterns):
                if any(
                    _contains_subsequence(prompt_tokens, pattern)
                    for pattern in patterns
                ):
                    mask[batch_index, fact_index] = True
        return mask

    def _relation_scores(self, query):
        u_columns, d_columns = [], []
        for positive, negative in zip(
            self.positive_prototypes, self.negative_prototypes
        ):
            p = positive.to(query.device)
            n = negative.to(query.device)
            u = (query @ p.T).max(dim=-1).values
            v = (query @ n.T).max(dim=-1).values
            u_columns.append(u)
            d_columns.append(u - v)
        return torch.stack(u_columns, dim=-1), torch.stack(d_columns, dim=-1)

    def _hook(self, module, args, output):
        if self._input_ids is None:
            raise RuntimeError("V2 association hook fired without input_ids")
        hidden = output[0] if isinstance(output, tuple) else output
        batch, width, _ = hidden.shape
        prefix_lengths = self._prefix_lengths_for(hidden)
        subject_mask = self._subject_mask(
            self._input_ids,
            prefix_lengths,
            attention_mask=self._attention_mask,
        )
        positions = prefix_lengths - 1
        query = hidden[
            torch.arange(batch, device=hidden.device), positions
        ].float()
        query = F.normalize(query, dim=-1)
        u, d = self._relation_scores(query)
        alpha = self.alpha.to(hidden.device)[None, :]
        tau = self.tau.to(hidden.device)[None, :]
        qualifies = subject_mask & (u >= alpha) & (d >= tau)

        ranked = d.masked_fill(~qualifies, float("-inf"))
        best_d, best_fact = ranked.max(dim=-1)
        active = torch.isfinite(best_d)

        rows = self.extra.to(device=hidden.device, dtype=hidden.dtype)
        selected = F.embedding(best_fact, rows)
        position_mask = F.one_hot(positions, num_classes=width).to(hidden.dtype)
        delta = (
            position_mask.unsqueeze(-1)
            * selected.unsqueeze(1)
            * active[:, None, None].to(hidden.dtype)
        )
        edited = hidden + delta

        self.calls += 1
        with torch.no_grad():
            self.active_batch_rows += int(active.sum())
            self.active_token_positions += int(active.sum())
            self.last_active_fact_indices = [
                [int(best_fact[index])] if bool(active[index]) else []
                for index in range(batch)
            ]
            self.last_route_scores = [
                {
                    "fact_index": int(best_fact[index]) if bool(active[index]) else None,
                    "u": (
                        float(u[index, best_fact[index]])
                        if bool(active[index]) else None
                    ),
                    "d": (
                        float(d[index, best_fact[index]])
                        if bool(active[index]) else None
                    ),
                }
                for index in range(batch)
            ]
            for fact_index in best_fact[active].detach().cpu().tolist():
                self.active_fact_counts[int(fact_index)] += 1

        if isinstance(output, tuple):
            return (edited, *output[1:])
        return edited

    def artifact(self):
        return {
            "architecture": "relation_prototype_fact_association_bank_v2",
            "layer": self.layer,
            "positive_prototypes": self.positive_prototypes,
            "negative_prototypes": self.negative_prototypes,
            "alpha": self.alpha.detach().cpu(),
            "tau": self.tau.detach().cpu(),
            "rows": self.extra.detach().cpu(),
            "subject_patterns": self.subject_patterns,
            "facts": self.facts,
            "routing_policy": "subject_candidate_plus_relation_prototype_confirmation",
            "unique_subject_bypass": False,
            "subject_scan_scope": "prompt_prefix_only",
            "teacher_forced_suffix_can_affect_routing": False,
            "generation_contract": (
                "uncached recomputation with fixed original request boundary"
            ),
            "trainable_parameters": sum(row.numel() for row in self.rows),
            "base_parameters_trainable": 0,
            "tokenizer_extended": False,
            "lm_head_edited": False,
            "object_required_in_runtime_input": False,
        }

    def counters(self):
        return {
            "hook_calls": self.calls,
            "active_batch_rows": self.active_batch_rows,
            "active_token_positions": self.active_token_positions,
            "active_fact_counts": list(self.active_fact_counts),
        }

    def close(self):
        self._hook_handle.remove()


def load_relation_prototype_artifact(base_model, artifact):
    bank = RelationPrototypeAssociationBank(
        base_model=base_model,
        layer=int(artifact["layer"]),
        positive_prototypes=artifact["positive_prototypes"],
        negative_prototypes=artifact["negative_prototypes"],
        alpha=artifact["alpha"],
        tau=artifact["tau"],
        subject_patterns=artifact["subject_patterns"],
        facts=artifact["facts"],
        rows=artifact["rows"],
    )
    for row in bank.rows:
        row.requires_grad_(False)
    return AssociationCausalLM(base_model, bank), bank
