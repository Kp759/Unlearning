#!/usr/bin/env python3
"""Bounded one-MLP pilot: select by sensitivity, fit train only, gate on development."""
from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import random
import time

import torch

from static_overlap_core import StaticEditor, answer_nll, model_logits, selected_logits
from static_overlap_data import Example, _encode, encode_bundle
from static_overlap_mlp_protocol import METHOD, load_pilot, write_new
from static_overlap_training import (TrainConfig, export_verified, forget_target,
                                    forgetting_status, sha256_file, training_protection)


def emit(**value):
    print(json.dumps(value, allow_nan=False), flush=True)


def encode_pilot(source, data, tokenizer, max_length):
    examples = []
    for e in encode_bundle(source, tokenizer, max_length, abstention=""):
        if e.role in ("retain", "language") or (e.role == "forget" and e.split == "train"):
            # Old retention validation was already formally reclassified as development
            # preservation. It is fitting-visible here; fresh authored development
            # families and fresh preservation facts/documents supply the gate.
            examples.append(replace(e, split="train"))
    for row in data["authored"]:
        prompt, completion = row["prompt"], " " + row["answer"]
        ids, offsets = _encode(tokenizer, prompt + completion, max_length)
        start = len(prompt) + 1
        positions = [i for i, (a, b) in enumerate(offsets) if b > start and a < len(prompt + completion) and b > a]
        if not positions or 0 in positions or any(a < start and (prompt + completion)[a:start].strip()
                                                for i, (a, b) in enumerate(offsets) if i in positions):
            raise ValueError(f"Invalid authored answer token boundary: {row['id']}")
        labels = [t if i in positions else -100 for i, t in enumerate(ids)]
        examples.append(Example(row["id"], row["split"], row["role"], row["fact_id"], ids, labels,
                                prompt, completion, row["family"]))
    for row in data["language"]:
        ids, offsets = _encode(tokenizer, row["text"], max_length)
        labels = [t if i > 0 and b > a else -100 for i, (t, (a, b)) in enumerate(zip(ids, offsets))]
        examples.append(Example(row["id"], row["split"], "language", None, ids, labels, "", row["text"], row["id"]))
    # Check actual tokenizer input/label identity, not just surface strings.
    seen, result = {}, []
    for e in examples:
        key = (tuple(e.input_ids), tuple(e.labels))
        if key in seen:
            if seen[key] != (e.split, e.role):
                raise ValueError(f"Tokenized data cross split or role boundaries: {e.id}")
            continue
        seen[key] = e.split, e.role
        result.append(e)
    for split in ("train", "development"):
        for role in ("forget", "retain", "language"):
            if not any(e.split == split and e.role == role for e in result):
                raise ValueError(f"Empty pilot stratum {split}/{role}")
    return result


def balanced_subset(examples, count, seed):
    groups = {}
    for e in examples:
        groups.setdefault(e.fact_id or e.group, []).append(e)
    rng = random.Random(seed)
    keys = list(groups)
    rng.shuffle(keys)
    chosen = []
    while len(chosen) < count and any(groups.values()):
        for key in keys:
            if groups[key] and len(chosen) < count:
                chosen.append(groups[key].pop(0))
    return chosen


def select_layer(model, examples, layers, count, seed):
    if not layers or len(set(layers)) != len(layers) or any(i < 0 or i >= len(model.model.layers) for i in layers):
        raise ValueError("Invalid registered MLP candidate layers for this architecture")
    forgotten = balanced_subset([e for e in examples if e.split == "train" and e.role == "forget"], count, seed)
    retained = balanced_subset([e for e in examples if e.split == "development" and e.role in ("retain", "language")], count, seed)
    if not forgotten or not retained:
        raise ValueError("Layer localization needs training forget and development preservation")
    params = [model.model.layers[i].mlp.down_proj.weight for i in layers]
    model.requires_grad_(False)
    norms = {}
    try:
        for p in params:
            p.requires_grad_(True)
        for label, items in (("forget", forgotten), ("retain", retained)):
            model.zero_grad(set_to_none=True)
            for j, e in enumerate(items):
                (answer_nll(model_logits(model, e), e) / len(items)).backward()
                if j == 0 or (j + 1) % 8 == 0:
                    emit(phase="localization_gradients", role=label, examples=j + 1, total=len(items))
            norms[label] = [float(p.grad.norm()) for p in params]
    finally:
        model.zero_grad(set_to_none=True)
        model.requires_grad_(False)
    rows = [{"layer": layer, "forget_gradient_norm": nf, "retain_gradient_norm": nr,
             "sensitivity_ratio": nf / (nr + 1e-12)}
            for layer, nf, nr in zip(layers, norms["forget"], norms["retain"])]
    if any(not math.isfinite(r["sensitivity_ratio"]) for r in rows):
        raise ValueError("Nonfinite layer sensitivity")
    selected = max(rows, key=lambda r: r["sensitivity_ratio"])["layer"]
    return selected, {"score": "norm(mean gradient forget NLL) / (norm(mean gradient retain NLL) + 1e-12)",
        "layers": rows, "selected_layer": selected, "forget_ids": [e.id for e in forgotten],
        "preservation_ids": [e.id for e in retained], "development_forget_used": False}


def weight_hashes(model):
    """Stream CPU copies; do not retain a second 3B-parameter snapshot."""
    hashes = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        h = hashlib.sha256()
        for chunk in parameter.detach().reshape(-1).split(1024 * 1024):
            h.update(chunk.float().cpu().numpy().tobytes())
        hashes[name] = h.hexdigest()
    return hashes


def verify_locality(model, original, layer):
    actual = weight_hashes(model)
    allowed = f"model.layers.{layer}.mlp.down_proj.weight"
    if set(actual) != set(original) or any(actual[n] != original[n] for n in original if n != allowed):
        raise ValueError("A parameter outside the selected MLP changed, including embedding/head")
    return {"only_allowed_parameter": allowed, "embedding_and_head_exact": True,
            "all_other_parameters_exact": True, "selected_parameter_changed": actual[allowed] != original[allowed]}


class References:
    """Full vocabulary base probabilities on disk, bounded CPU LRU; no edited cache."""
    def __init__(self, directory, max_bytes=256 * 1024**2):
        self.directory = Path(directory)
        self.directory.mkdir(exist_ok=False)
        self.max_bytes, self.bytes, self.cache = max_bytes, 0, OrderedDict()
        self.nll = {}

    def path(self, e):
        return self.directory / (hashlib.sha256(e.id.encode()).hexdigest() + ".pt")

    @torch.no_grad()
    def build(self, model, examples):
        started = time.monotonic()
        for i, e in enumerate(examples):
            values, labels = selected_logits(model_logits(model, e), e)
            lp = values.log_softmax(-1)
            self.nll[e.id] = -lp.gather(1, labels[:, None]).mean().item()
            if e.role != "forget":
                torch.save(lp.cpu(), self.path(e))
            if i == 0 or (i + 1) % 100 == 0 or i + 1 == len(examples):
                emit(phase="base_references", examples=i+1, total=len(examples), elapsed_seconds=time.monotonic()-started)

    def get(self, e, device):
        if e.id in self.cache:
            lp = self.cache.pop(e.id)
            self.bytes -= lp.numel() * lp.element_size()
        else:
            lp = torch.load(self.path(e), weights_only=True)
        size = lp.numel() * lp.element_size()
        while self.cache and self.bytes + size > self.max_bytes:
            _, old = self.cache.popitem(last=False)
            self.bytes -= old.numel() * old.element_size()
        if size <= self.max_bytes:
            self.cache[e.id] = lp
            self.bytes += size
        return lp.to(device)


@torch.no_grad()
def measure_pilot(model, examples, references):
    rows = []
    for e in examples:
        values, labels = selected_logits(model_logits(model, e), e)
        lq = values.double().log_softmax(-1)
        nll = -lq.gather(1, labels[:, None]).mean().item()
        base_nll = references.nll[e.id]
        kl = 0.0
        if e.role != "forget":
            lp = references.get(e, values.device).double()
            # Renormalize the stored FP32 log-probabilities before FP64 scoring.
            lp = lp - lp.logsumexp(-1, keepdim=True)
            base_nll = -lp.gather(1, labels[:, None]).mean().item()
            kl = (lp.exp() * (lp - lq)).sum(-1).mean().clamp_min(0).item()
        rows.append({"id": e.id, "split": e.split, "role": e.role, "nll": nll,
                     "base_nll": base_nll, "nll_increase": nll-base_nll, "kl": kl})
    return rows


def development_gate(rows, config):
    result = {}
    for split in ("train", "development"):
        current = [r for r in rows if r["split"] == split]
        result[split] = {"forgetting": forgetting_status(current, config),
                         "preservation": training_protection(current, config)[1]}
    result["passed"] = all(s["forgetting"]["target_met"] and s["preservation"]["retention_passed"]
                           for s in (result["train"], result["development"]))
    result["preservation_numeric_slack"] = 0.0
    return result


def fitting_batch(items, step, count, hard_ids=(), *, offset=None):
    if any(e.split != "train" for e in items):
        raise ValueError("Only training examples may enter objective gradients")
    hard = [e for e in items if e.id in set(hard_ids)][:count // 2]
    width = count - len(hard)
    start = (step - 1) * width if offset is None else offset
    rotating = [items[(start + j) % len(items)] for j in range(width)]
    return hard + rotating


def fit(editor, examples, references, plan, config, output, *, method=METHOD):
    train_f = [e for e in examples if e.split == "train" and e.role == "forget"]
    train_r = [e for e in examples if e.split == "train" and e.role in ("retain", "language")]
    rng = random.Random(plan["seed"])
    rng.shuffle(train_f)
    rng.shuffle(train_r)
    optimizer = torch.optim.Adam(editor.parameters, lr=plan["learning_rate"])
    started = time.monotonic()
    base_norm = next(iter(editor.downs.values())).base.weight.norm().item()
    history, hard_f, hard_r, seen_f, seen_r = [], [], [], set(), set()
    cursor_f = cursor_r = 0
    gate, selected, stop = None, None, "step_budget"
    for step in range(1, plan["steps"] + 1):
        if time.monotonic() - started > plan["max_training_seconds"]:
            stop = "wall_time_budget"
            break
        optimizer.zero_grad(set_to_none=True)
        batch_f = fitting_batch(train_f, step, plan["forget_batch"], hard_f, offset=cursor_f)
        batch_r = fitting_batch(train_r, step, plan["retain_batch"], hard_r, offset=cursor_r)
        cursor_f += len(batch_f) - min(len(hard_f), plan["forget_batch"] // 2)
        cursor_r += len(batch_r) - min(len(hard_r), plan["retain_batch"] // 2)
        loss_value = 0.0
        for e in batch_f:
            nll = answer_nll(model_logits(editor.model, e), e)
            gap = torch.relu(nll.new_tensor(forget_target(references.nll[e.id], config)) - nll)
            loss = gap / len(batch_f)
            loss.backward()
            loss_value += loss.item()
            seen_f.add(e.id)
        for e in batch_r:
            values, labels = selected_logits(model_logits(editor.model, e), e)
            lp = references.get(e, values.device)
            lq = values.log_softmax(-1)
            nll = -lq.gather(1, labels[:, None]).mean()
            kl = (lp.exp() * (lp-lq)).sum(-1).mean().clamp_min(0)
            excess = torch.relu((nll - references.nll[e.id] - config.training_nll_budget) / config.retain_nll_budget)
            loss = (plan["kl_weight"] * kl / config.training_kl_budget + plan["nll_weight"] * excess.square()) / len(batch_r)
            loss.backward()
            loss_value += loss.item()
            seen_r.add(e.id)
        norm = torch.nn.utils.clip_grad_norm_(editor.parameters, 1., error_if_nonfinite=True)
        optimizer.step()
        with torch.no_grad():
            delta_norm = editor.norm_sq().clamp_min(0).sqrt().item()
            cap = plan["relative_delta_cap"] * base_norm
            if delta_norm > cap:
                for edit in editor.downs.values():
                    edit.B.mul_(cap / delta_norm)
        if step == 1 or step % 5 == 0:
            emit(phase="mlp_step", step=step, objective=loss_value, gradient_norm=float(norm),
                 elapsed_seconds=time.monotonic()-started)
        if step % plan["check_every"] == 0 or step == plan["steps"]:
            emit(phase="development_gate_start", step=step, examples=len(examples))
            rows = measure_pilot(editor.model, examples, references)
            gate = development_gate(rows, config)
            history.append({"step": step, "gate": gate, "elapsed_seconds": time.monotonic()-started})
            write_report = {"method": method, "exploratory": True, "history": history,
                "selected_step": step if gate["passed"] else None, "last_gate": gate,
                "fitting_forget_seen": len(seen_f), "fitting_forget_total": len(train_f),
                "fitting_preservation_seen": len(seen_r), "fitting_preservation_total": len(train_r)}
            (output / "training_report.json").write_text(json.dumps(write_report, indent=2, allow_nan=False)+"\n")
            emit(phase="development_gate", step=step, **gate)
            torch.save(editor.artifact(), output / "last_factors.pt")
            if gate["passed"]:
                selected, stop = step, "development_gate_passed"
                torch.save(editor.artifact(), output / "training_factors.pt")
                break
            # Replay is derived strictly from FITTING rows. Development forget
            # scores can gate selection but never determine gradients or batches.
            hard_f = [r["id"] for r in sorted((r for r in rows if r["split"] == "train" and r["role"] == "forget"),
                                              key=lambda r: r["nll"])[:plan["forget_batch"] // 2]]
            hard_r = [r["id"] for r in sorted((r for r in rows if r["split"] == "train" and r["role"] != "forget"),
                       key=lambda r: max(r["nll_increase"] / .05, r["kl"] / .01), reverse=True)[:plan["retain_batch"] // 2]]
    return {"method": method, "exploratory": True, "stop_reason": stop, "selected_step": selected,
            "last_gate": gate, "history": history, "elapsed_seconds": time.monotonic()-started,
            "fitting_forget_seen": len(seen_f), "fitting_forget_total": len(train_f),
            "fitting_preservation_seen": len(seen_r), "fitting_preservation_total": len(train_r),
            "native_checkpoint_created": False, "final_evaluation_started": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-protocol", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)
    p = load_pilot(args.pilot_protocol)
    if Path(args.model_path).resolve() != Path(p["base_model_path"]).resolve():
        raise ValueError("Start from the original base, not the head-edited checkpoint")
    plan = p["plan"]
    output = Path(args.pilot_protocol).resolve().parent
    write_new(output / "training_started.json", {"model": str(Path(args.model_path).resolve()),
        "pilot_protocol_sha256": sha256_file(args.pilot_protocol)})
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.manual_seed(plan["seed"])
    if args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True, local_files_only=args.local_files_only)
    source = json.loads(Path(p["source_bundle"]["path"]).read_text())
    data = json.loads(Path(p["data"]["path"]).read_text())
    examples = encode_pilot(source, data, tokenizer, plan["max_length"])
    write_new(output / "encoded_development_examples.json", [asdict(e) for e in examples])
    emit(phase="load_original_base", model=args.model_path, method=METHOD, examples=len(examples))
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.float32,
        local_files_only=args.local_files_only, attn_implementation="eager").to(args.device).eval()
    model.requires_grad_(False)
    emit(phase="fingerprint_original_parameters", scope="all parameters including tied endpoints")
    original = weight_hashes(model)
    write_new(output / "original_weight_hashes.json", original)
    layer, localization = select_layer(model, examples, plan["candidate_layers"], plan["localization_examples"], plan["seed"])
    write_new(output / "localization.json", localization)
    emit(phase="mlp_layer_selected", **localization)
    references = References(output / "base_references")
    references.build(model, examples)
    width = model.model.layers[layer].mlp.down_proj.in_features
    editor = StaticEditor(model, [], [], {layer: list(range(width))}, rank=plan["rank"])
    assert not editor.rows and len(editor.downs) == 1
    with torch.no_grad():
        actual = model_logits(model, examples[0])
        with editor.base():
            expected = model_logits(model, examples[0])
        if not torch.isfinite(actual).all() or not torch.equal(actual, expected):
            raise ValueError("Zero-initialized MLP adapter changed base logits")
        del actual, expected
    emit(phase="mlp_preparation", layer=layer, rank=plan["rank"], base_logits_exact=True,
         shared_endpoints_preserved=editor.shared, embedding_and_head_trainable=False)
    config = TrainConfig(target_probability=plan["target_probability"], retain_nll_budget=.05,
        retain_kl_budget=.01, retain_nll_safety_margin=plan["fitting_nll_margin"],
        retain_kl_safety_margin=plan["fitting_kl_margin"])
    report = fit(editor, examples, references, plan, config, output)
    report.update(localization=localization, pilot_protocol_sha256=sha256_file(args.pilot_protocol))
    (output / "training_report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    if report["selected_step"] is None:
        emit(status="no_development_valid_edit", report=str(output / "training_report.json"), final_tests_touched=False)
        return 2
    manifest = {"method": METHOD, "exploratory": True, "model_path": str(Path(args.model_path).resolve()),
        "exploratory_protocol_path": str(Path(args.pilot_protocol).resolve()),
        "exploratory_protocol_sha256": sha256_file(args.pilot_protocol),
        "forget_associations": [f for f in source["facts"] if f["role"] == "forget"],
        "training_text_fingerprints": data["training_text_fingerprints"],
        "settings": {"abstention": "I don't know.", **plan}, "localization": localization}
    write_new(output / "training_manifest.json", manifest)
    locality = {}
    def reload_verified(path):
        loaded = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32,
            local_files_only=True, attn_implementation="eager").to(args.device).eval()
        locality.update(verify_locality(loaded, original, layer))
        return loaded
    emit(phase="merge_reload_strict_verification", selected_step=report["selected_step"])
    export_verified(editor, tokenizer, examples, config, output / "checkpoint", torch.float32,
        reload_verified, atol=1e-4, rtol=1e-5, manifest=manifest, numeric_slack=0., require_forgetting=True)
    report.update(native_checkpoint_created=True, locality=locality)
    (output / "training_report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    emit(status="verified_mlp_checkpoint", checkpoint=str(output / "checkpoint"), locality=locality,
         development_gate=report["last_gate"], official_eff_gen_measured=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
