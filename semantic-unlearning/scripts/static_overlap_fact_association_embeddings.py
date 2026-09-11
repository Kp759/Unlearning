"""Fact-specific association embeddings with natural subject+relation activation.

The base causal LM is frozen. Each forget record owns one independent trainable
hidden-state vector. A frozen semantic key, computed from authored natural
subject+relation prompts, decides whether that vector is injected. Exact subject
surface tokens are an eligibility constraint; relation/context is resolved from
ordinary hidden states. No private token, fact-id argument, tokenizer extension,
LM-head edit, or base-weight update is used.

The forgotten object belongs to the association and supplies the suppression
target, but it is deliberately NOT required by the runtime trigger because the
object is absent from ordinary questions.
"""
from __future__ import annotations

from dataclasses import replace
from collections import defaultdict
import math
from types import SimpleNamespace

import torch
from torch import nn
from torch.nn import functional as F

from mcf_shadow_relation_prompts import RELATION_NOUN_PHRASES
from mcf_synthetic_paraphrase_templates import (
    GENERIC_CONTEXT_PREFIXES,
    RELATION_ALTERNATE_TEMPLATES,
)
from run_static_overlap_extended_tokens_standalone_v2 import (
    TRAIN_SCAFFOLDS,
    DEVELOPMENT_SCAFFOLDS,
)
from static_overlap_data import Example, _encode
from static_overlap_natural_writer import _encode_answer_example, _relation_noun


METHOD = "static_overlap_fact_association_embeddings_v1"

PLAN = {
    "layer": 19,
    "steps": 1500,
    "check_every": 50,
    "learning_rate": 0.05,
    "backtracks": 12,
    "max_training_seconds": 3600,
    "max_stalled_steps": 150,
    "target_probability": 1e-6,
    "unknown_completion": " I don't know.",
    "unknown_weight": 1.0,
    "proposal_objective": "phase_lexicographic",
    "post_feasible_gates": 2,
    "seed": 1,
    "max_length": 512,
    "gate_slack": 0.04,
    "relation_negative_count": 8,
    "log_phase": "fact_association_embedding",
    "natural_prompt_behavior": (
        "matched subject+relation may receive one learned fact vector; "
        "otherwise the frozen base path is exact"
    ),
    "radius_schedule": (
        (1e-3, 1.0),
        (1e-5, 0.35),
        (1e-6, 0.08),
        (0.0, 0.02),
    ),
}


def normalized(value):
    return " ".join(str(value).casefold().split())


def build_forget_examples(facts, tokenizer, max_length):
    """Build fitting/development natural prompts without benchmark paraphrases."""
    examples = []
    for fact in facts:
        relation = _relation_noun(fact["relation"])
        examples.append(
            _encode_answer_example(
                fact,
                "train",
                "canonical_rewrite",
                fact["canonical_prompt"],
                tokenizer,
                max_length,
            )
        )
        for split, scaffolds in (
            ("train", TRAIN_SCAFFOLDS),
            ("development", DEVELOPMENT_SCAFFOLDS),
        ):
            for family, scaffold in enumerate(scaffolds):
                prompt = scaffold.format(subject=fact["subject"], relation=relation)
                examples.append(
                    _encode_answer_example(
                        fact,
                        split,
                        f"authored_{family}",
                        prompt,
                        tokenizer,
                        max_length,
                    )
                )
        alternatives = RELATION_ALTERNATE_TEMPLATES.get(fact["relation"])
        if not alternatives:
            raise ValueError(f"No independent relation alternatives for {fact['relation']}")
        for family, template in enumerate(alternatives):
            prompt = template.format(fact["subject"])
            examples.append(
                _encode_answer_example(
                    fact,
                    "train",
                    f"relation_alternate_{family}",
                    prompt,
                    tokenizer,
                    max_length,
                )
            )
            prefix = GENERIC_CONTEXT_PREFIXES[
                (int(fact["case_id"]) + family) % len(GENERIC_CONTEXT_PREFIXES)
            ]
            examples.append(
                _encode_answer_example(
                    fact,
                    "train",
                    f"context_relation_alternate_{family}",
                    f"{prefix} {prompt}",
                    tokenizer,
                    max_length,
                )
            )

    seen = {}
    result = []
    for example in examples:
        key = (tuple(example.input_ids), tuple(example.labels))
        previous = seen.get(key)
        if previous is not None:
            if previous != (example.split, example.fact_id):
                raise ValueError(
                    f"Tokenized prompt crosses split/fact boundaries: {example.id}"
                )
            continue
        seen[key] = (example.split, example.fact_id)
        result.append(example)

    for fact in facts:
        for split in ("train", "development"):
            if not any(e.fact_id == fact["id"] and e.split == split for e in result):
                raise ValueError(f"Missing {split} views for {fact['id']}")
    return result


def replace_completion(example, completion, tokenizer, max_length):
    """Retokenize one natural prompt with an ordinary abstention completion."""
    full = example.prompt + completion
    ids, offsets = _encode(tokenizer, full, max_length)
    start = len(example.prompt) + 1
    positions = [
        index
        for index, (left, right) in enumerate(offsets)
        if right > start and left < len(full) and right > left
    ]
    if not positions or 0 in positions:
        raise ValueError(f"No completion tokens for {example.id}")
    labels = [token if index in positions else -100 for index, token in enumerate(ids)]
    return replace(
        example,
        id=f"{example.id}:unknown",
        input_ids=ids,
        labels=labels,
        completion=completion,
        group=f"{example.group}:unknown",
    )


def subject_token_patterns(tokenizer, subject):
    patterns = []
    for text in (str(subject), " " + str(subject)):
        ids = tuple(tokenizer(text, add_special_tokens=False)["input_ids"])
        if ids and ids not in patterns:
            patterns.append(ids)
    if not patterns:
        raise ValueError(f"Subject tokenization is empty: {subject!r}")
    return patterns


def _contains_subsequence(tokens, pattern):
    width = len(pattern)
    if width == 0 or width > len(tokens):
        return False
    return any(tuple(tokens[i:i + width]) == tuple(pattern)
               for i in range(len(tokens) - width + 1))


@torch.no_grad()
def extract_prompt_queries(model, tokenizer, prompts, layer, batch_size=16):
    """Return normalized hidden state at the final non-padding prompt token."""
    if not prompts:
        raise ValueError("Prompt query extraction requires prompts")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = next(model.parameters()).device
    rows = []
    for start in range(0, len(prompts), int(batch_size)):
        batch = prompts[start:start + int(batch_size)]
        encoded = tokenizer(
            batch,
            padding=True,
            return_tensors="pt",
            return_token_type_ids=False,
        ).to(device)
        result = model(
            **encoded,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = result.hidden_states[int(layer) + 1].float()
        mask = encoded["attention_mask"].bool()
        positions = (
            torch.arange(mask.shape[1], device=device)[None, :]
            .expand_as(mask)
            .masked_fill(~mask, -1)
            .max(dim=1)
            .values
        )
        if bool((positions < 0).any()):
            raise ValueError("Empty tokenized prompt")
        query = hidden[
            torch.arange(hidden.shape[0], device=device),
            positions,
        ]
        rows.append(F.normalize(query, dim=-1).cpu())
    return torch.cat(rows, dim=0)


def relation_negative_prompts(fact, facts, count):
    """Same subject, deliberately different relation: training-safe gate negatives."""
    relation_ids = sorted({
        other["relation"]
        for other in facts
        if other["relation"] != fact["relation"]
        and other["relation"] in RELATION_NOUN_PHRASES
    })
    if not relation_ids:
        raise ValueError("Need at least two relation types for relation-specific routing")
    offset = int(fact["case_id"]) % len(relation_ids)
    rotated = relation_ids[offset:] + relation_ids[:offset]
    chosen = rotated[: min(len(rotated), max(1, int(count)))]
    prompts = []
    for index, relation_id in enumerate(chosen):
        relation = _relation_noun(relation_id)
        scaffold = TRAIN_SCAFFOLDS[index % len(TRAIN_SCAFFOLDS)]
        prompts.append(scaffold.format(subject=fact["subject"], relation=relation))
    return prompts


@torch.no_grad()
def build_semantic_keys(model, tokenizer, facts, examples, layer, gate_slack,
                        relation_negative_count):
    """Build one frozen key/threshold per fact from fitting-only natural prompts."""
    train_by_fact = defaultdict(list)
    dev_by_fact = defaultdict(list)
    for example in examples:
        (train_by_fact if example.split == "train" else dev_by_fact)[
            example.fact_id
        ].append(example.prompt)

    all_train_prompts = []
    train_owner = []
    for fact_index, fact in enumerate(facts):
        prompts = train_by_fact[fact["id"]]
        if not prompts:
            raise ValueError(f"No fitting prompts for {fact['id']}")
        all_train_prompts.extend(prompts)
        train_owner.extend([fact_index] * len(prompts))
    queries = extract_prompt_queries(model, tokenizer, all_train_prompts, layer)

    groups = []
    cursor = 0
    keys = []
    for fact_index, fact in enumerate(facts):
        width = len(train_by_fact[fact["id"]])
        group = queries[cursor:cursor + width]
        cursor += width
        key = F.normalize(group.mean(dim=0), dim=0)
        keys.append(key)
        groups.append(group)
    keys = torch.stack(keys)

    negative_prompts = []
    negative_owner = []
    for fact_index, fact in enumerate(facts):
        prompts = relation_negative_prompts(
            fact, facts, relation_negative_count
        )
        negative_prompts.extend(prompts)
        negative_owner.extend([fact_index] * len(prompts))
    negative_queries = extract_prompt_queries(
        model, tokenizer, negative_prompts, layer
    )

    thresholds = []
    per_fact = []
    negative_cursor = 0
    for fact_index, fact in enumerate(facts):
        positive = groups[fact_index] @ keys[fact_index]
        nwidth = sum(owner == fact_index for owner in negative_owner)
        negative = (
            negative_queries[negative_cursor:negative_cursor + nwidth]
            @ keys[fact_index]
        )
        negative_cursor += nwidth
        positive_floor = float(positive.min())
        negative_ceiling = float(negative.max()) if int(negative.numel()) else -1.0
        # Preserve full fitting recall and add modest semantic slack for unseen
        # paraphrases. Same-subject/different-relation firing is audited rather
        # than silently hidden.
        threshold = max(-1.0, positive_floor - float(gate_slack))
        thresholds.append(threshold)
        per_fact.append({
            "fact_id": fact["id"],
            "subject": fact["subject"],
            "relation": fact["relation"],
            "object": fact["object"],
            "positive_min": positive_floor,
            "positive_mean": float(positive.mean()),
            "negative_relation_max": negative_ceiling,
            "threshold": threshold,
            "train_positive_count": int(positive.numel()),
            "relation_negative_count": int(negative.numel()),
            "train_positive_pass_fraction": float((positive >= threshold).float().mean()),
            "relation_negative_fire_fraction": (
                float((negative >= threshold).float().mean())
                if int(negative.numel()) else 0.0
            ),
        })

    dev_prompts = []
    dev_owner = []
    for fact_index, fact in enumerate(facts):
        prompts = dev_by_fact[fact["id"]]
        dev_prompts.extend(prompts)
        dev_owner.extend([fact_index] * len(prompts))
    dev_queries = extract_prompt_queries(model, tokenizer, dev_prompts, layer)
    thresholds_tensor = torch.tensor(thresholds, dtype=torch.float32)
    dev_correct = 0
    dev_active = 0
    for query, owner in zip(dev_queries, dev_owner):
        scores = query @ keys.T
        chosen = int(scores.argmax())
        active = float(scores[chosen]) >= float(thresholds_tensor[chosen])
        dev_active += int(active)
        dev_correct += int(active and chosen == owner)

    diagnostics = {
        "layer": int(layer),
        "gate": "exact subject eligibility AND frozen contextual cosine key",
        "gate_slack": float(gate_slack),
        "official_paraphrases_used": False,
        "official_neighborhoods_used": False,
        "development_used_for_key_or_threshold_fitting": False,
        "development_prompt_count": len(dev_owner),
        "development_any_key_active_fraction": (
            dev_active / len(dev_owner) if dev_owner else None
        ),
        "development_correct_key_fraction": (
            dev_correct / len(dev_owner) if dev_owner else None
        ),
        "per_fact": per_fact,
    }
    return keys, thresholds_tensor, diagnostics


class FactAssociationBank(nn.Module):
    """One independent hidden-state vector per forgotten association."""

    def __init__(self, base_model, layer, keys, thresholds, subject_patterns,
                 facts, rows=None):
        super().__init__()
        if keys.ndim != 2 or len(facts) != keys.shape[0]:
            raise ValueError("Association keys must be [num_facts, hidden_size]")
        if thresholds.shape != (keys.shape[0],):
            raise ValueError("Association thresholds must match keys")
        hidden_size = int(keys.shape[1])
        if rows is None:
            initial = torch.zeros(
                (len(facts), hidden_size),
                device=next(base_model.parameters()).device,
                dtype=next(base_model.parameters()).dtype,
            )
        else:
            initial = rows.to(
                device=next(base_model.parameters()).device,
                dtype=next(base_model.parameters()).dtype,
            )
            if tuple(initial.shape) != (len(facts), hidden_size):
                raise ValueError("Saved association rows have wrong shape")
        self.rows = nn.ParameterList(nn.Parameter(row.clone()) for row in initial)
        self.register_buffer("keys", keys.detach().float().clone())
        self.register_buffer("thresholds", thresholds.detach().float().clone())
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

    def _subject_mask(self, input_ids):
        rows = input_ids.detach().cpu().tolist()
        mask = torch.zeros(
            (len(rows), len(self.facts)),
            dtype=torch.bool,
            device=input_ids.device,
        )
        for batch_index, tokens in enumerate(rows):
            for fact_index, patterns in enumerate(self.subject_patterns):
                if any(_contains_subsequence(tokens, pattern) for pattern in patterns):
                    mask[batch_index, fact_index] = True
        return mask

    def _hook(self, module, args, output):
        if self._input_ids is None:
            raise RuntimeError("Association bank hook fired without bound input_ids")
        hidden = output[0] if isinstance(output, tuple) else output
        batch, width, _ = hidden.shape
        subject_mask = self._subject_mask(self._input_ids)

        if self._prefix_lengths is not None:
            prefix_lengths = self._prefix_lengths.to(hidden.device, dtype=torch.long)
        elif self._attention_mask is not None:
            mask = self._attention_mask.to(hidden.device).bool()
            prefix_lengths = (
                torch.arange(mask.shape[1], device=hidden.device)[None, :]
                .expand_as(mask)
                .masked_fill(~mask, -1)
                .max(dim=1)
                .values
                + 1
            )
        else:
            prefix_lengths = torch.full(
                (batch,), width, device=hidden.device, dtype=torch.long
            )
        if tuple(prefix_lengths.shape) != (batch,):
            raise ValueError("Association prefix lengths must have one value per sequence")
        if bool(((prefix_lengths <= 0) | (prefix_lengths > width)).any()):
            raise ValueError("Association prefix boundary is outside the input sequence")

        prompt_positions = prefix_lengths - 1
        query = hidden[
            torch.arange(batch, device=hidden.device),
            prompt_positions,
        ].float()
        query = F.normalize(query, dim=-1)
        keys = F.normalize(self.keys.to(hidden.device), dim=-1)
        scores = query @ keys.T
        scores = scores.masked_fill(~subject_mask, float("-inf"))
        best_score, best_fact = scores.max(dim=-1)
        threshold = self.thresholds.to(hidden.device)[best_fact]
        active = torch.isfinite(best_score) & (best_score >= threshold)

        rows = self.extra.to(device=hidden.device, dtype=hidden.dtype)
        selected = F.embedding(best_fact, rows)
        position_mask = F.one_hot(prompt_positions, num_classes=width).to(hidden.dtype)
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
            for fact_index in best_fact[active].detach().cpu().tolist():
                self.active_fact_counts[int(fact_index)] += 1

        if isinstance(output, tuple):
            return (edited, *output[1:])
        return edited

    def artifact(self):
        return {
            "architecture": "natural_subject_relation_fact_association_bank_v1",
            "layer": self.layer,
            "keys": self.keys.detach().cpu(),
            "thresholds": self.thresholds.detach().cpu(),
            "rows": self.extra.detach().cpu(),
            "subject_patterns": self.subject_patterns,
            "facts": self.facts,
            "trainable_parameters": sum(row.numel() for row in self.rows),
            "base_parameters_trainable": 0,
            "tokenizer_extended": False,
            "lm_head_edited": False,
            "requires_fact_id_token_injection": False,
            "runtime_gate_inputs": "ordinary input_ids plus frozen hidden states only",
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


class AssociationCausalLM(nn.Module):
    """Thin model wrapper that supplies ordinary input_ids to the internal bank."""

    def __init__(self, base_model, bank):
        super().__init__()
        self.base_model = base_model
        self.bank = bank
        self._next_prefix_lengths = None

    @property
    def config(self):
        return self.base_model.config

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.base_model.get_output_embeddings()

    def set_association_prefix_lengths(self, lengths):
        self._next_prefix_lengths = torch.as_tensor(lengths, dtype=torch.long)

    def forward(self, input_ids=None, **kwargs):
        if input_ids is None:
            raise ValueError("Association model requires input_ids")
        attention_mask = kwargs.get("attention_mask")
        prefix_lengths = self._next_prefix_lengths
        self._next_prefix_lengths = None
        if prefix_lengths is not None:
            prefix_lengths = prefix_lengths.to(input_ids.device)
        self.bank.bind(
            input_ids,
            attention_mask=attention_mask,
            prefix_lengths=prefix_lengths,
        )
        try:
            return self.base_model(input_ids=input_ids, **kwargs)
        finally:
            self.bank.unbind()


class FactAssociationEditor:
    """Compatibility shim for the row-wise V2.1 optimizer."""

    def __init__(self, base_model, bank):
        base_model.requires_grad_(False)
        base_model.eval()
        self.embedding = bank
        self.model = AssociationCausalLM(base_model, bank)
        self.parameters = list(bank.rows)
        trainable = [p for p in self.model.parameters() if p.requires_grad]
        if {id(p) for p in trainable} != {id(p) for p in self.parameters}:
            raise ValueError("Only the 50 association rows may be trainable")

    def artifact(self):
        return self.embedding.artifact()


def make_unknown_examples(examples, tokenizer, max_length, completion):
    return {
        example.id: replace_completion(
            example, completion, tokenizer, max_length
        )
        for example in examples
    }


def make_subject_patterns(tokenizer, facts):
    return [subject_token_patterns(tokenizer, fact["subject"]) for fact in facts]



@torch.no_grad()
def audit_runtime_routes(model, bank, tokenizer, examples, fact_to_row, batch_size=16):
    """Audit automatic routing with zero/nonzero rows without using eval labels."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = next(model.parameters()).device
    rows = []
    for start in range(0, len(examples), int(batch_size)):
        batch = examples[start:start + int(batch_size)]
        encoded = tokenizer(
            [example.prompt for example in batch],
            padding=True,
            return_tensors="pt",
            return_token_type_ids=False,
        ).to(device)
        model(**encoded, use_cache=False)
        active_sets = list(bank.last_active_fact_indices)
        if len(active_sets) != len(batch):
            raise RuntimeError("Association route audit did not capture one route set per prompt")
        for example, active_ids in zip(batch, active_sets):
            expected = fact_to_row[example.fact_id]
            rows.append({
                "id": example.id,
                "split": example.split,
                "fact_id": example.fact_id,
                "expected_row": expected,
                "active_rows": active_ids,
                "correct_row_active": expected in active_ids,
                "any_row_active": bool(active_ids),
                "wrong_row_active": any(index != expected for index in active_ids),
            })
    result = {}
    for split in ("train", "development"):
        current = [row for row in rows if row["split"] == split]
        if not current:
            raise ValueError(f"Route audit has no {split} prompts")
        result[split] = {
            "count": len(current),
            "correct_row_active_fraction": sum(
                row["correct_row_active"] for row in current
            ) / len(current),
            "any_row_active_fraction": sum(
                row["any_row_active"] for row in current
            ) / len(current),
            "wrong_row_active_fraction": sum(
                row["wrong_row_active"] for row in current
            ) / len(current),
            "failures": [
                row for row in current if not row["correct_row_active"]
            ][:20],
        }
    return result

def load_artifact_into_model(base_model, artifact):
    bank = FactAssociationBank(
        base_model=base_model,
        layer=int(artifact["layer"]),
        keys=artifact["keys"],
        thresholds=artifact["thresholds"],
        subject_patterns=artifact["subject_patterns"],
        facts=artifact["facts"],
        rows=artifact["rows"],
    )
    for row in bank.rows:
        row.requires_grad_(False)
    return AssociationCausalLM(base_model, bank), bank
