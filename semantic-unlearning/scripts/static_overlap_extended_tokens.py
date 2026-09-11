"""Private input-token rows for an explicit oracle-routing ablation.

This module deliberately keeps the output vocabulary unchanged.  Consequently
ordinary prompts are bit-exact to the base model, while IDs above the original
vocabulary are accepted only by the extended input embedding.
"""
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import math
import random
import time

import torch
from torch import nn

from run_static_overlap_mlp_pilot import emit, encode_pilot
from static_overlap_core import answer_nll, flat_parameters, model_logits, set_parameters
from static_overlap_training import TrainConfig, forget_target


def normalized_field(value):
    return " ".join(value.casefold().split())


def association_token_specs(facts):
    """Create stable unique labels, adding the subject only for collisions."""
    forgotten = sorted((fact for fact in facts if fact["role"] == "forget"),
                       key=lambda fact: fact["id"])
    relation_object = [
        f"{normalized_field(fact['relation'])} - {normalized_field(fact['object'])}"
        for fact in forgotten
    ]
    counts = Counter(relation_object)
    result, labels = [], set()
    for index, (fact, base) in enumerate(zip(forgotten, relation_object)):
        label = (base if counts[base] == 1 else
                 f"{normalized_field(fact['subject'])} - {base}")
        if label in labels:
            label = f"{label} - {fact['id']}"
        labels.add(label)
        digest = hashlib.sha256(f"{fact['id']}:{label}".encode()).hexdigest()[:16]
        result.append({"fact_id": fact["id"], "association": label,
                       "token": f"<|forget_assoc_{index:02d}_{digest}|>"})
    if len(result) != len(forgotten) or len({row["token"] for row in result}) != len(result):
        raise ValueError("Association-token construction is not one-to-one")
    return result


class InputOnlyExtendedEmbedding(nn.Module):
    """Preserve every original row and add independently trainable input rows."""
    def __init__(self, base, initial_rows):
        super().__init__()
        if not isinstance(base, nn.Embedding) or base.max_norm is not None:
            raise ValueError("Input-only extension requires a native embedding without max_norm")
        if (initial_rows.ndim != 2 or initial_rows.shape[1] != base.embedding_dim
                or not torch.isfinite(initial_rows).all()):
            raise ValueError("Invalid extended-token initialization")
        # Keep the preservation guarantee local to this module.  The wrapper
        # must never allow an optimizer to update an original vocabulary row,
        # even when it is instantiated outside ExtendedTokenEditor.
        base.requires_grad_(False)
        self.base = base
        self.original_vocab_size = base.num_embeddings
        self.extra = nn.Parameter(initial_rows.detach().clone().to(base.weight.device, base.weight.dtype))

    @property
    def weight(self):
        # Compatibility for dtype/device inspection. Forward does not allocate
        # this concatenation and original IDs always call the native embedding.
        return torch.cat((self.base.weight, self.extra), dim=0)

    def forward(self, ids):
        original = ids < self.original_vocab_size
        if not bool(((ids >= 0) & (ids < self.original_vocab_size + len(self.extra))).all()):
            raise ValueError("Input token ID is outside the extended vocabulary")
        base_values = self.base(ids.clamp_max(self.original_vocab_size - 1))
        extra_ids = (ids - self.original_vocab_size).clamp_min(0)
        extra_values = nn.functional.embedding(extra_ids, self.extra)
        return torch.where(original.unsqueeze(-1), base_values, extra_values)


class ExtendedTokenEditor:
    def __init__(self, model, initial_rows):
        model.requires_grad_(False)
        model.eval()
        self.model, self.merged, self.shared = model, False, False
        self.base_embedding = model.get_input_embeddings()
        self.embedding = InputOnlyExtendedEmbedding(self.base_embedding, initial_rows)
        model.set_input_embeddings(self.embedding)
        self.parameters = [self.embedding.extra]
        assert {id(parameter) for parameter in model.parameters() if parameter.requires_grad} == {
            id(self.embedding.extra)
        }

    def artifact(self):
        return {"architecture": "input_only_extended_embedding_v1",
                "original_vocab_size": self.embedding.original_vocab_size,
                "extra_input_rows": self.embedding.extra.detach().cpu(),
                "output_vocabulary_extended": False,
                "requires_association_token_injection": True}


def initialize_rows(model, tokenizer, facts_by_id, specs):
    weight = model.get_input_embeddings().weight.detach()
    rows = []
    for spec in specs:
        fact = facts_by_id[spec["fact_id"]]
        text = f"{fact['subject']} {fact['relation']} {fact['object']}"
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        ids = [token for token in ids if token < len(weight)]
        if not ids:
            raise ValueError(f"Association {fact['id']} has no original-vocabulary tokens")
        rows.append(weight[ids].float().mean(0))
    return torch.stack(rows).to(weight.dtype)


def insert_association_token(example, token_id, suffix):
    """Insert after BOS; optionally replace the labeled answer with a suffix."""
    if token_id in example.input_ids:
        raise ValueError("Private association token already occurs in an original example")
    if suffix is None:
        ids = [example.input_ids[0], token_id, *example.input_ids[1:]]
        labels = [-100, -100, *example.labels[1:]]
        completion = example.completion
    else:
        labeled = [index for index, label in enumerate(example.labels) if label != -100]
        if not labeled:
            raise ValueError(f"Forget example {example.id} has no labeled answer")
        prefix = example.input_ids[:labeled[0]]
        suffix_ids = suffix
        ids = [prefix[0], token_id, *prefix[1:], *suffix_ids]
        labels = [-100] * (len(ids) - len(suffix_ids)) + list(suffix_ids)
        completion = "<unknown-target>"
    return replace(example, id=f"extended:{example.id}:{'answer' if suffix is None else 'unknown'}",
                   input_ids=ids, labels=labels, prompt=f"<association-token> {example.prompt}",
                   completion=completion, group=f"extended:{example.group}")


def build_routed_examples(examples, tokenizer, specs, original_vocab_size, unknown_completion):
    token_ids = {spec["fact_id"]: tokenizer.convert_tokens_to_ids(spec["token"]) for spec in specs}
    if sorted(token_ids.values()) != list(range(original_vocab_size, original_vocab_size + len(specs))):
        raise ValueError("Private tokens must occupy one contiguous input-only extension")
    suffix = tokenizer(unknown_completion, add_special_tokens=False)["input_ids"]
    if not suffix or any(token >= original_vocab_size for token in suffix):
        raise ValueError("Unknown completion must use only the original output vocabulary")
    answer, unknown = {}, {}
    for example in examples:
        if example.role != "forget":
            continue
        token_id = token_ids[example.fact_id]
        answer[example.id] = insert_association_token(example, token_id, None)
        unknown[example.id] = insert_association_token(example, token_id, suffix)
    return answer, unknown


def paired_routed_loss(model, answer_examples, unknown_examples, base_nll, config, plan):
    gap = unknown_loss = 0.
    for answer, unknown in zip(answer_examples, unknown_examples):
        answer_loss = answer_nll(model_logits(model, answer), answer)
        current_gap = torch.relu(answer_loss.new_tensor(
            forget_target(base_nll[answer.id.removeprefix("extended:").removesuffix(":answer")], config)
        ) - answer_loss)
        current_unknown = answer_nll(model_logits(model, unknown), unknown)
        gap = gap + current_gap / len(answer_examples)
        unknown_loss = unknown_loss + current_unknown / len(answer_examples)
    return gap + plan["unknown_weight"] * unknown_loss, gap, unknown_loss


@torch.no_grad()
def routed_forgetting(model, routed, base_nll, config):
    rows = []
    for original_id, example in routed.items():
        nll = float(answer_nll(model_logits(model, example), example))
        rows.append({"id": original_id, "split": example.split, "nll": nll,
                     "base_nll": base_nll[original_id]})
    result = {}
    for split in ("train", "development"):
        current = [row for row in rows if row["split"] == split]
        probabilities = [math.exp(-row["nll"]) for row in current]
        result[split] = {"count": len(current),
                         "mean_token_probability": sum(probabilities) / len(probabilities),
                         "max_token_probability": max(probabilities),
                         "target_probability": config.target_probability,
                         "target_met": all(row["nll"] >= forget_target(row["base_nll"], config)
                                           for row in current)}
    return result


@torch.no_grad()
def routed_unknown_completion(model, routed):
    """Report how strongly routed prompts select the ordinary abstention text."""
    rows = []
    for original_id, example in routed.items():
        nll = float(answer_nll(model_logits(model, example), example))
        rows.append({"id": original_id, "split": example.split, "nll": nll})
    result = {}
    for split in ("train", "development"):
        current = [row for row in rows if row["split"] == split]
        probabilities = [math.exp(-row["nll"]) for row in current]
        result[split] = {
            "count": len(current),
            "mean_token_probability": sum(probabilities) / len(probabilities),
            "minimum_token_probability": min(probabilities),
            "mean_nll": sum(row["nll"] for row in current) / len(current),
        }
    return result


def train_extended_tokens(editor, original_examples, routed_answer, routed_unknown,
                          base_nll, config, plan, output):
    train = [example for example in original_examples
             if example.split == "train" and example.role == "forget"]
    by_fact = defaultdict(list)
    for example in train:
        by_fact[example.fact_id].append(example)
    facts = sorted(by_fact)
    random.Random(plan["seed"]).shuffle(facts)
    cursors = defaultdict(int)
    optimizer = torch.optim.Adam(editor.parameters, lr=plan["learning_rate"])
    history, gates, rejected, cursor = [], [], 0, 0
    started, stop = time.monotonic(), "step_budget"
    for step in range(1, plan["steps"] + 1):
        if time.monotonic() - started >= plan["max_training_seconds"]:
            stop = "wall_time_budget"
            break
        chosen = [facts[(cursor + index) % len(facts)]
                  for index in range(min(plan["forget_batch"], len(facts)))]
        cursor = (cursor + len(chosen)) % len(facts)
        originals = []
        for fact_id in chosen:
            views = by_fact[fact_id]
            originals.append(views[cursors[fact_id] % len(views)])
            cursors[fact_id] += 1
        answers = [routed_answer[example.id] for example in originals]
        unknowns = [routed_unknown[example.id] for example in originals]
        before = flat_parameters(editor.parameters).detach().clone()
        state = deepcopy(optimizer.state_dict())
        optimizer.zero_grad(set_to_none=True)
        loss, gap, unknown_loss = paired_routed_loss(
            editor.model, answers, unknowns, base_nll, config, plan)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(editor.parameters, 1., error_if_nonfinite=True)
        optimizer.step()
        proposal = flat_parameters(editor.parameters).detach() - before
        proposal *= min(1., plan["step_radius"] / max(float(proposal.norm()), 1e-30))
        accepted, backtracks, after = False, None, (loss, gap, unknown_loss)
        for index in range(plan["backtracks"] + 1):
            with torch.no_grad():
                set_parameters(editor.parameters, before + proposal * (.5 ** index))
                candidate = paired_routed_loss(
                    editor.model, answers, unknowns, base_nll, config, plan)
            if (all(bool(torch.isfinite(value).detach()) for value in candidate)
                    and float(candidate[0].detach()) < float(loss.detach()) - 1e-7):
                accepted, backtracks, after = True, index, candidate
                break
        if not accepted:
            with torch.no_grad():
                set_parameters(editor.parameters, before)
            optimizer.load_state_dict(state)
        rejected = 0 if accepted else rejected + 1
        record = {"step": step, "accepted": accepted, "backtracks": backtracks,
                  "before_loss": float(loss), "after_loss": float(after[0]),
                  "before_forget_gap": float(gap), "after_forget_gap": float(after[1]),
                  "before_unknown_nll": float(unknown_loss), "after_unknown_nll": float(after[2]),
                  "step_norm": float((flat_parameters(editor.parameters) - before).norm()),
                  "elapsed_seconds": time.monotonic() - started}
        history.append(record)
        emit(phase="extended_token_step", **record)
        if step % plan["check_every"] == 0:
            gate = routed_forgetting(editor.model, routed_answer, base_nll, config)
            unknown = routed_unknown_completion(editor.model, routed_unknown)
            gates.append({"step": step, "routed_forgetting": gate,
                          "routed_unknown_completion": unknown})
            emit(phase="extended_token_gate", step=step, routed_forgetting=gate,
                 routed_unknown_completion=unknown,
                 natural_prompt_behavior="bit_exact_base_by_construction")
            torch.save(editor.artifact(), output / "last_extended_input_rows.pt")
            if gate["train"]["target_met"] and gate["development"]["target_met"]:
                stop = "oracle_routed_forgetting_target_met"
                break
        if rejected >= plan["max_stalled_steps"]:
            stop = "consecutive_rejected_steps"
            break
    return {"stop_reason": stop, "history": history, "gates": gates,
            "elapsed_seconds": time.monotonic() - started}


def run(model, tokenizer, source, data, plan, output):
    original_vocab_size = model.get_input_embeddings().num_embeddings
    specs = association_token_specs(source["facts"])
    if len(specs) != plan["association_tokens"]:
        raise ValueError("Protocol requires exactly one private token per forget fact")
    added = tokenizer.add_special_tokens({"additional_special_tokens": [row["token"] for row in specs]})
    if added != len(specs) or len(tokenizer) != original_vocab_size + len(specs):
        raise ValueError("Tokenizer extension is not a clean 50-row append")
    examples = encode_pilot(source, data, tokenizer, plan["max_length"])
    contaminated = [example.id for example in examples
                    if any(token >= original_vocab_size for token in example.input_ids)]
    if contaminated:
        raise ValueError(f"Natural prompts unexpectedly contain private tokens: {contaminated[:3]}")
    if model.get_output_embeddings().weight.shape[0] != original_vocab_size:
        raise ValueError("Input-only experiment requires the original output vocabulary")
    facts_by_id = {fact["id"]: fact for fact in source["facts"]}
    initial = initialize_rows(model, tokenizer, facts_by_id, specs)
    base_nll = {example.id: float(answer_nll(model_logits(model, example), example))
                for example in examples if example.role == "forget"}
    parity_example = min(
        (example for example in examples if example.role in ("retain", "language")),
        key=lambda example: len(example.input_ids),
    )
    base_parity_logits = model_logits(model, parity_example).detach().clone()
    editor = ExtendedTokenEditor(model, initial)
    for example in examples:
        if example.role in ("retain", "language"):
            before = model.get_input_embeddings().base(
                torch.tensor([example.input_ids], device=initial.device))
            after = model.get_input_embeddings()(
                torch.tensor([example.input_ids], device=initial.device))
            if not torch.equal(before, after):
                raise ValueError("Original-token embedding parity failed")
    edited_parity_logits = model_logits(model, parity_example).detach()
    if not torch.equal(base_parity_logits, edited_parity_logits):
        raise ValueError("Natural-prompt full-logit parity failed")
    routed_answer, routed_unknown = build_routed_examples(
        examples, tokenizer, specs, original_vocab_size, plan["unknown_completion"])
    manifest = {"architecture": "input_only_extended_association_tokens_v1",
                "association_tokens": specs, "original_vocab_size": original_vocab_size,
                "extended_input_vocab_size": len(tokenizer),
                "output_vocab_size": model.get_output_embeddings().weight.shape[0],
                "unknown_completion": plan["unknown_completion"],
                "base_parameters_trainable": 0,
                "extended_input_parameters": editor.embedding.extra.numel(),
                "natural_prompt_logits_exact_base": True,
                "natural_prompt_parity_example_id": parity_example.id,
                "requires_fact_id_token_injection": True,
                "official_evaluation_eligible": False,
                "final_tests_touched": False}
    (output / "extended_token_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    tokenizer.save_pretrained(output / "extended_tokenizer")
    emit(phase="extended_tokens_ready", **{key: value for key, value in manifest.items()
                                            if key != "association_tokens"})
    config = TrainConfig(target_probability=plan["target_probability"],
                         retain_nll_budget=.05, retain_kl_budget=.01,
                         retain_nll_safety_margin=0., retain_kl_safety_margin=0.)
    report = train_extended_tokens(
        editor, examples, routed_answer, routed_unknown, base_nll, config, plan, output)
    report.update(manifest=manifest, method="static_overlap_extended_tokens_v1")
    (output / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")
    torch.save(editor.artifact(), output / "extended_input_rows.pt")
    emit(status="extended_token_oracle_ablation_complete", stop_reason=report["stop_reason"],
         report=str(output / "training_report.json"), official_evaluation_started=False,
         natural_prompt_forgetting_expected=False)
    return report
