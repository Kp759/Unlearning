"""Static relation-key MLP rewiring for overlap-aware factual unlearning."""
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
import json
import math
import time

import torch
from torch import nn

from freeze_static_overlap_development import normalized
from run_static_overlap_endpoint_ga import baseline_rows, compact_gate
from run_static_overlap_mlp_pilot import development_gate, emit, measure_pilot
from static_overlap_activation_protocol import METHOD
from static_overlap_core import decoder_writeouts, flat_parameters, model_logits, set_parameters, tied_weights
from static_overlap_orthogonal_rewire import build_same_subject_locality
from static_overlap_paired_ga_gd import PairSampler, objective, weighted_examples
from static_overlap_training import training_protection


class RelationKeyDown(nn.Module):
    """A fixed contextual key matrix with trainable value vectors."""
    def __init__(self, base, rank):
        super().__init__()
        if not isinstance(base, nn.Linear) or rank <= 0:
            raise ValueError("Relation-key edit needs a linear down projection and positive rank")
        self.base = base
        self.register_buffer("keys", torch.zeros(rank, base.in_features, device=base.weight.device))
        self.values = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        self.enabled = True

    def forward(self, activation):
        result = self.base(activation)
        if not self.enabled:
            return result
        coefficients = activation.float() @ self.keys.T
        correction = coefficients @ self.values.T
        return result + correction.to(result.dtype)

    def delta(self):
        return self.values @ self.keys


class ActivationRewireEditor:
    """Edit only contextual MLP value vectors; endpoints stay untied and frozen."""
    def __init__(self, model, layers, rank):
        if tied_weights(model):
            raise ValueError("Embedding and LM head must be untied before activation rewiring")
        writeouts = decoder_writeouts(model)
        layers = sorted(set(int(layer) for layer in layers))
        if not layers or any(layer not in writeouts for layer in layers):
            raise ValueError("Invalid activation-rewire layers")
        model.requires_grad_(False)
        model.eval()
        self.model, self.shared, self.merged = model, False, False
        self.downs = {}
        for layer in layers:
            edit = RelationKeyDown(writeouts[layer], rank)
            self.downs[layer] = edit
            model.model.layers[layer].mlp.down_proj = edit
        self.parameters = [edit.values for edit in self.downs.values()]
        assert {id(parameter) for parameter in model.parameters() if parameter.requires_grad} == {
            id(parameter) for parameter in self.parameters
        }

    @contextmanager
    def base(self):
        if self.merged:
            raise RuntimeError("Original base unavailable after merge")
        states = [edit.enabled for edit in self.downs.values()]
        try:
            for edit in self.downs.values():
                edit.enabled = False
            yield
        finally:
            for edit, state in zip(self.downs.values(), states):
                edit.enabled = state

    @torch.no_grad()
    def set_keys(self, keys):
        if set(keys) != set(self.downs):
            raise ValueError("Relation-key layers differ from editor layers")
        for layer, edit in self.downs.items():
            value = keys[layer]
            if (value.shape != edit.keys.shape or value.dtype != edit.keys.dtype
                    or not torch.isfinite(value).all()):
                raise ValueError(f"Invalid relation keys for layer {layer}")
            edit.keys.copy_(value)

    def artifact(self):
        return {"shared_endpoints": False, "activation_relation_rewire": {
            layer: {"keys": edit.keys.detach().cpu(), "values": edit.values.detach().cpu()}
            for layer, edit in self.downs.items()
        }}

    @torch.no_grad()
    def load_artifact(self, artifact):
        saved = artifact.get("activation_relation_rewire", {})
        if (self.merged or artifact.get("shared_endpoints") is not False
                or {int(layer) for layer in saved} != set(self.downs)):
            raise ValueError("Activation-rewire artifact has another architecture")
        for layer, edit in self.downs.items():
            state = saved[layer] if layer in saved else saved[str(layer)]
            if set(state) != {"keys", "values"}:
                raise ValueError("Invalid activation-rewire artifact fields")
            for name in ("keys", "values"):
                value, target = state[name], getattr(edit, name)
                if (not isinstance(value, torch.Tensor) or value.shape != target.shape
                        or value.dtype != target.dtype or not torch.isfinite(value).all()):
                    raise ValueError(f"Invalid activation-rewire tensor {layer}.{name}")
                target.copy_(value)

    @torch.no_grad()
    def merge(self):
        if self.merged:
            raise RuntimeError("Already merged")
        for layer, edit in self.downs.items():
            edit.base.weight.add_(edit.delta().to(edit.base.weight.dtype))
            self.model.model.layers[layer].mlp.down_proj = edit.base
        self.model.requires_grad_(False)
        self.model.config.tie_word_embeddings = False
        self.merged = True
        assert not tied_weights(self.model)
        return self.model


def labeled_prediction_positions(example, device):
    labels = torch.tensor(example.labels[1:], device=device)
    positions = torch.nonzero(labels != -100, as_tuple=False).flatten()
    if not len(positions):
        raise ValueError(f"No answer positions for {example.id}")
    return positions


@torch.no_grad()
def collect_activation_means(editor, examples):
    """Capture one answer-position activation vector per example and layer."""
    current, captured = {}, {layer: {} for layer in editor.downs}
    handles = []
    for layer, edit in editor.downs.items():
        handles.append(edit.register_forward_pre_hook(
            lambda _module, args, index=layer: current.__setitem__(index, args[0].detach())
        ))
    try:
        for example in examples:
            current.clear()
            model_logits(editor.model, example)
            for layer in editor.downs:
                if layer not in current:
                    raise RuntimeError(f"Layer {layer} activation hook did not run")
                activation = current[layer]
                positions = labeled_prediction_positions(example, activation.device)
                captured[layer][example.id] = activation[0, positions].float().mean(0).detach()
    finally:
        for handle in handles:
            handle.remove()
    return captured


def relation_keys(forget_by_fact, locality_by_fact, retain_vectors, rank,
                  relative_tolerance=1e-6, minimum_ratio=1e-4):
    """Subtract Slot-2 locality, then remove the complete retain activation span."""
    facts = sorted(forget_by_fact)
    if len(facts) > rank or set(facts) != set(locality_by_fact) or not retain_vectors:
        raise ValueError("Relation-key construction needs matched facts and retained activations")
    retained = torch.stack(retain_vectors).float()
    retained = torch.nn.functional.normalize(retained, dim=1)
    q, triangular = torch.linalg.qr(retained.T, mode="reduced")
    diagonal = triangular.diagonal().abs()
    numerical_rank = int((diagonal > relative_tolerance * diagonal.max().clamp_min(1e-30)).sum())
    # Keep the complete orthonormal QR basis for projection.  Truncating Q at
    # ``numerical_rank`` is invalid without column pivoting: a nearly dependent
    # early activation can make a small diagonal precede later independent
    # directions.  The extra QR-completion directions are conservative and the
    # activation width is much larger than the number of fitting anchors.
    keys, reports = [], []
    for fact in facts:
        positive = torch.stack(forget_by_fact[fact]).float().mean(0)
        locality = torch.stack(locality_by_fact[fact]).float().mean(0)
        contrast = positive - locality
        original_norm = contrast.norm().clamp_min(1e-30)
        residual = contrast - q @ (q.T @ contrast) if q.numel() else contrast
        if q.numel():
            residual = residual - q @ (q.T @ residual)
        ratio = float(residual.norm() / original_norm)
        if not math.isfinite(ratio) or ratio < minimum_ratio:
            raise ValueError(f"No retained-orthogonal activation direction for {fact}: {ratio}")
        keys.append(residual / residual.norm())
        reports.append({"fact_id": fact, "contrast_norm": float(original_norm),
                        "residual_ratio": ratio})
    width = retained.shape[1]
    result = retained.new_zeros((rank, width))
    result[:len(keys)] = torch.stack(keys)
    max_retain_overlap = float((retained @ result[:len(keys)].T).abs().max())
    return result, {"retain_examples": len(retained), "retain_activation_rank": numerical_rank,
                    "forget_facts": len(facts), "minimum_residual_ratio": min(
                        report["residual_ratio"] for report in reports),
                    "mean_residual_ratio": sum(report["residual_ratio"] for report in reports) / len(reports),
                    "max_normalized_retain_key_overlap": max_retain_overlap,
                    "facts": reports}


def build_and_install_relation_keys(editor, examples, synthetic_ids_by_fact, plan):
    captured = collect_activation_means(editor, examples)
    forget_examples = [example for example in examples
                       if example.split == "train" and example.role == "forget"]
    preserve_examples = [example for example in examples
                         if example.split == "train" and example.role in ("retain", "language")]
    keys, report = {}, {}
    for layer, rows in captured.items():
        forget = defaultdict(list)
        for example in forget_examples:
            forget[example.fact_id].append(rows[example.id])
        locality = {fact: [rows[eid] for eid in ids]
                    for fact, ids in synthetic_ids_by_fact.items()}
        key, detail = relation_keys(
            forget, locality, [rows[example.id] for example in preserve_examples],
            plan["activation_key_rank"], plan["activation_basis_relative_tolerance"],
            plan["minimum_relation_key_residual_ratio"],
        )
        keys[layer], report[layer] = key, detail
    editor.set_keys(keys)
    return report


def activation_step(editor, optimizer, pairs, background, references, config, plan):
    fs, rs = weighted_examples(pairs, background, plan)
    before = flat_parameters(editor.parameters).detach().clone()
    optimizer_state = deepcopy(optimizer.state_dict())
    optimizer.zero_grad(set_to_none=True)
    initial, forget_gradient = objective(editor, fs, rs, references, config, plan, backward=True)
    total_gradient = torch.cat([parameter.grad.detach().flatten() for parameter in editor.parameters])
    retain_gradient = total_gradient - forget_gradient
    cosine = float(torch.dot(forget_gradient, retain_gradient) /
                   (forget_gradient.norm() * retain_gradient.norm()).clamp_min(1e-30))
    torch.nn.utils.clip_grad_norm_(editor.parameters, 1., error_if_nonfinite=True)
    optimizer.step()
    proposal = flat_parameters(editor.parameters).detach() - before
    proposal *= min(1., plan["step_radius"] / max(float(proposal.norm()), 1e-30))
    accepted, backtracks, actual = False, None, initial
    try:
        for index in range(plan["backtracks"] + 1):
            with torch.no_grad():
                set_parameters(editor.parameters, before + proposal * (.5 ** index))
                candidate, _ = objective(editor, fs, rs, references, config, plan)
            forget_ok = (candidate["forget_gap"] < initial["forget_gap"] - 1e-6
                         if initial["forget_gap"] > 0 else candidate["forget_gap"] == 0)
            if (all(math.isfinite(value) for value in candidate.values()) and forget_ok
                    and candidate["loss"] < initial["loss"] - 1e-8
                    and candidate["retention_violation"] <= 0):
                accepted, backtracks, actual = True, index, candidate
                break
    finally:
        if not accepted:
            with torch.no_grad():
                set_parameters(editor.parameters, before)
            optimizer.load_state_dict(optimizer_state)
    return {"mode": "activation_relation_rewire", "accepted": accepted,
            "backtracks": backtracks, "before": initial, "after": actual,
            "forget_gradient_norm": float(forget_gradient.norm()),
            "retain_gradient_norm": float(retain_gradient.norm()),
            "forget_retain_gradient_cosine": cosine,
            "step_norm": float((flat_parameters(editor.parameters).detach() - before).norm())}


def fit(editor, examples, references, config, plan, output, *, source, data, tokenizer):
    synthetic, locality_audit = build_same_subject_locality(source, data, tokenizer, plan)
    examples.extend(synthetic)
    data["training_text_fingerprints"] = sorted(set(data["training_text_fingerprints"]) | {
        normalized(example.prompt) for example in synthetic
    } | {normalized(example.prompt + example.completion) for example in synthetic})
    references.build(editor.model, synthetic)
    (output / "same_subject_locality_manifest.json").write_text(json.dumps({
        "method": METHOD, "rows": locality_audit, "verified_fact_count": 0,
        "synthetic_query_count": len(synthetic), "invented_answers_used": False,
    }, indent=2) + "\n")
    synthetic_ids_by_fact = {fact: [row["id"] for row in rows]
                             for fact, rows in locality_audit.items()}
    key_report = build_and_install_relation_keys(editor, examples, synthetic_ids_by_fact, plan)
    (output / "activation_relation_keys.json").write_text(json.dumps({
        "method": METHOD, "layers": key_report, "invented_answers_used": False,
        "keys_are_runtime_router": False,
    }, indent=2) + "\n")
    emit(phase="activation_relation_keys_ready", layers={str(layer): {
        key: value for key, value in detail.items() if key != "facts"
    } for layer, detail in key_report.items()})
    sampler = PairSampler(examples, source["facts"] + data.get("facts", []), plan["seed"])
    pairing = sampler.manifest()
    pairing.update(method=METHOD, synthetic_same_subject_locality_queries=len(synthetic),
                   same_subject_supervision="base_distribution_distillation_without_answer_claim")
    (output / "pair_manifest.json").write_text(json.dumps(pairing, indent=2) + "\n")
    emit(phase="association_pairs_ready", forget_facts=len(sampler.order),
         coverage=pairing["coverage_forget_facts"], exact_mixed_pairs=len(pairing["exact_mixed_companions"]))
    initial = baseline_rows(examples, references)
    baseline = development_gate(initial, config)
    (output / "baseline_metrics.json").write_text(json.dumps({"gate": baseline, "rows": initial}) + "\n")
    emit(phase="activation_rewire_baseline", **compact_gate(baseline))
    optimizer = torch.optim.Adam(editor.parameters, lr=plan["learning_rate"])
    started = time.monotonic()
    history, gates, hard_f, hard_r = [], [], [], []
    seen_f, seen_r = set(), set()
    safe_delta = flat_parameters(editor.parameters).detach().clone()
    safe_rows, safe_step = initial, 0
    last_rows, last_gate = initial, baseline
    rejected = stale = failed_preservation_gates = step = 0
    selected, stop = None, "step_budget"
    previous_mean = sum(row["nll"] for row in initial
                        if row["split"] == "train" and row["role"] == "forget") / sum(
        row["split"] == "train" and row["role"] == "forget" for row in initial)

    def check_gate():
        nonlocal safe_delta, safe_rows, safe_step, last_rows, last_gate, hard_f, hard_r
        nonlocal selected, stop, previous_mean, stale, failed_preservation_gates
        emit(phase="activation_rewire_development_gate_start", step=step, examples=len(examples))
        observed = measure_pilot(editor.model, examples, references)
        gate = development_gate(observed, config)
        train = [row for row in observed if row["split"] == "train"]
        retention_ok = training_protection(train, config, internal=True)[1]["retention_passed"]
        hard_f = [row["id"] for row in sorted(
            (row for row in train if row["role"] == "forget"), key=lambda row: row["nll"])]
        hard_r = [row["id"] for row in sorted(
            (row for row in train if row["role"] != "forget"),
            key=lambda row: max(row["nll_increase"] / config.training_nll_budget,
                                row["kl"] / config.training_kl_budget), reverse=True)]
        if retention_ok:
            safe_delta = flat_parameters(editor.parameters).detach().clone()
            safe_rows, safe_step = observed, step
            last_rows, last_gate = observed, gate
        else:
            failed_preservation_gates += 1
            torch.save(editor.artifact(), output / "last_rejected_block_delta.pt")
            with torch.no_grad():
                set_parameters(editor.parameters, safe_delta)
            optimizer.state.clear()
            last_rows, last_gate = safe_rows, development_gate(safe_rows, config)
        mean = sum(row["nll"] for row in last_rows
                   if row["split"] == "train" and row["role"] == "forget") / sum(
            row["split"] == "train" and row["role"] == "forget" for row in last_rows)
        gain = mean - previous_mean
        stale = (stale + 1 if gain < plan["min_gate_nll_gain"] else 0) if retention_ok else 0
        previous_mean = mean
        record = {"step": step, "gate": gate, "training_preservation_passed": retention_ok,
                  "rolled_back": not retention_ok, "retained_state_step": safe_step,
                  "failed_preservation_gates": failed_preservation_gates,
                  "training_mean_nll_gain_since_gate": gain}
        gates.append(record)
        (output / f"gate_{step}_metrics.json").write_text(json.dumps(observed) + "\n")
        (output / "last_metrics.json").write_text(json.dumps(last_rows) + "\n")
        torch.save(editor.artifact(), output / "last_endpoint_delta.pt")
        emit(phase="activation_rewire_development_gate",
             **{key: value for key, value in record.items() if key != "gate"}, **compact_gate(gate))
        if retention_ok and gate["passed"]:
            selected, stop = step, "development_gate_passed"
            torch.save(editor.artifact(), output / "training_factors.pt")

    def report():
        return {"method": METHOD, "exploratory": True, "stop_reason": stop,
                "selected_step": selected, "last_gate": last_gate,
                "last_state_step": safe_step, "baseline_gate": baseline,
                "history": history, "gates": gates,
                "elapsed_seconds": time.monotonic() - started,
                "pair_manifest": str(output / "pair_manifest.json"),
                "activation_relation_keys": str(output / "activation_relation_keys.json"),
                "fitting_forget_seen": len(seen_f), "fitting_preservation_seen": len(seen_r),
                "development_used_for_gradients": False, "native_checkpoint_created": False,
                "final_tests_touched": False, "inference_router": False,
                "embedding_and_lm_head_untied_but_frozen": True,
                "relation_keys_fixed_before_value_optimization": True}

    for step in range(1, plan["steps"] + 1):
        if time.monotonic() - started >= plan["max_training_seconds"]:
            step -= 1
            stop = "wall_time_budget"
            break
        pairs, background = sampler.batch(plan, hard_f, hard_r)
        result = activation_step(editor, optimizer, pairs, background, references, config, plan)
        result.update(step=step, elapsed_seconds=time.monotonic() - started)
        history.append(result)
        seen_f.update(forget.id for forget, _ in pairs)
        seen_r.update(retain.id for _, retains in pairs for retain in retains)
        seen_r.update(example.id for example in background)
        rejected = 0 if result["accepted"] else rejected + 1
        emit(phase="activation_rewire_step", **result)
        with (output / "training.jsonl").open("a") as handle:
            handle.write(json.dumps(result) + "\n")
        if step % plan["check_every"] == 0 or step == plan["steps"]:
            check_gate()
            (output / "training_report.json").write_text(json.dumps(report(), indent=2) + "\n")
            if selected is not None:
                break
            if stale >= plan["stalled_gates"]:
                stop = "insufficient_safe_forgetting_progress"
                break
            if failed_preservation_gates >= plan["max_failed_preservation_gates"]:
                stop = "activation_rewire_could_not_satisfy_full_training_gate"
                break
        if rejected >= plan["max_stalled_steps"]:
            stop = "consecutive_rejected_activation_steps"
            break
    if not gates or gates[-1]["step"] != step:
        check_gate()
    return report()
