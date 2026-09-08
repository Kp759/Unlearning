"""Joint bounded GA / retain GD / exact forward KL with finite-anchor protection."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import tempfile

import torch

from static_overlap_core import (
    answer_nll, bounded_forget, constrained_step, flat_gradient, forward_kl,
    model_logits, selected_logits, tied_weights,
)


@dataclass
class TrainConfig:
    steps: int = 200
    batch_size: int = 4
    protected_batch_size: int = 8
    learning_rate: float = 1e-3
    forget_increase: float = 2.0
    lambda_forget: float = 1.0
    lambda_abstain: float = 1.0
    lambda_retain: float = 1.0
    lambda_kl: float = 1.0
    lambda_delta: float = 1e-4
    epsilon: float = 1e-4
    step_radius: float = 0.01
    retain_nll_budget: float = 0.05
    retain_kl_budget: float = 0.01
    backtracks: int = 10
    max_stalled_steps: int = 10
    seed: int = 1

    def validate(self):
        for key, value in asdict(self).items():
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Non-finite/non-numeric config: {key}")
            if key != "seed" and value < 0:
                raise ValueError(f"Negative config: {key}")
        for key in ("steps", "batch_size", "protected_batch_size", "max_stalled_steps"):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if type(self.backtracks) is not int or type(self.seed) is not int:
            raise ValueError("backtracks and seed must be integers")
        for key in ("learning_rate", "forget_increase", "step_radius", "lambda_forget", "lambda_retain", "lambda_kl"):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")


@torch.no_grad()
def measure(editor, examples):
    rows = []
    for example in examples:
        with editor.base():
            base = model_logits(editor.model, example)
        edited = model_logits(editor.model, example)
        base_nll = answer_nll(base, example).item()
        nll = answer_nll(edited, example).item()
        rows.append({"id": example.id, "split": example.split, "role": example.role,
                     "base_nll": base_nll, "nll": nll, "nll_increase": nll - base_nll,
                     "kl": forward_kl(base, edited, example).item()})
    return rows


def within_budgets(rows, config):
    protected = [r for r in rows if r["role"] in ("retain", "language")]
    if not protected:
        raise ValueError("Protection cannot pass with an empty anchor set")
    finite = all(math.isfinite(r[k]) for r in protected for k in ("nll", "base_nll", "kl"))
    max_nll = max(r["nll_increase"] for r in protected)
    max_kl = max(r["kl"] for r in protected)
    passed = finite and max_nll <= config.retain_nll_budget and max_kl <= config.retain_kl_budget
    return passed, {"max_retained_nll_increase": max_nll, "max_retained_kl": max_kl}


def train(editor, examples, config, log_path=None):
    config.validate()
    fitting = [e for e in examples if e.split == "train"]
    forget = [e for e in fitting if e.role == "forget"]
    retain = [e for e in fitting if e.role == "retain"]
    anchors = [e for e in fitting if e.role in ("retain", "language")]
    abstain = [e for e in fitting if e.role == "abstain"]
    language = [e for e in anchors if e.role == "language"]
    if not forget or not retain or not language or (config.lambda_abstain and not abstain):
        raise ValueError("Missing a required training objective role")
    # Fixed per-example targets, set once against the unmodified model.
    baseline = measure(editor, fitting)
    if any(abs(row["nll_increase"]) > 1e-7 or row["kl"] > 1e-7 for row in baseline):
        raise ValueError("Training must start from zero effective deltas")
    base_nll = {row["id"]: row["base_nll"] for row in baseline}
    optimizer = torch.optim.Adam(editor.parameters, lr=config.learning_rate)
    rng, history, stalls = random.Random(config.seed), [], 0
    order = list(anchors)
    rng.shuffle(order)
    cursor = 0

    def sample(pool):
        return rng.sample(pool, min(config.batch_size, len(pool)))

    for step in range(config.steps):
        protected = [order[(cursor + j) % len(order)]
                     for j in range(min(config.protected_batch_size, len(order)))]
        cursor = (cursor + len(protected)) % len(order)
        gradients = torch.stack([flat_gradient(answer_nll(model_logits(editor.model, e), e),
                                               editor.parameters) for e in protected])
        # Accumulate each micro-example separately to bound activation memory.
        # A detached leaf surrogate delivers this aggregate gradient to Adam.
        aggregate = torch.zeros_like(gradients[0])
        components = {"forget": 0.0, "abstain": 0.0, "retain": 0.0, "kl": 0.0, "delta": 0.0}

        def add(name, loss):
            nonlocal aggregate
            components[name] += loss.detach().item()
            aggregate += flat_gradient(loss, editor.parameters)

        batch = sample(forget)
        for e in batch:
            nll = answer_nll(model_logits(editor.model, e), e)
            add("forget", config.lambda_forget * bounded_forget(nll, base_nll[e.id], config.forget_increase) / len(batch))
        if config.lambda_abstain:
            batch = sample(abstain)
            for e in batch:
                add("abstain", config.lambda_abstain * answer_nll(model_logits(editor.model, e), e) / len(batch))
        batch = sample(retain)
        for e in batch:
            add("retain", config.lambda_retain * answer_nll(model_logits(editor.model, e), e) / len(batch))
        # Guarantee both answer-prefix and general-language KL in every step.
        batch = sample(retain) + sample(language)
        for e in batch:
            with editor.base(), torch.no_grad():
                base = model_logits(editor.model, e)
            add("kl", config.lambda_kl * forward_kl(base, model_logits(editor.model, e), e) / len(batch))
        add("delta", config.lambda_delta * editor.norm_sq())
        parameters = torch.cat([p.flatten() for p in editor.parameters])
        surrogate = (parameters * aggregate).sum()

        def check():
            # All training anchors, including mixed companion spans, checked
            # against BASE budgets after each proposal/backtrack. No ratcheting.
            return within_budgets(measure(editor, anchors), config)

        record = constrained_step(optimizer, editor.parameters, surrogate, gradients, check,
                                  epsilon=config.epsilon, radius=config.step_radius,
                                  backtracks=config.backtracks)
        record.update(step=step + 1, objective=sum(components.values()), components=components)
        history.append(record)
        if log_path:
            with Path(log_path).open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
        print(json.dumps({k: record[k] for k in ("step", "accepted", "objective", "step_norm")}), flush=True)
        stalls = 0 if record["accepted"] else stalls + 1
        if stalls >= config.max_stalled_steps:
            break
    report = {"config": asdict(config), "history": history,
              "stop_reason": "no_useful_feasible_step" if stalls >= config.max_stalled_steps else "step_budget",
              "accepted_steps": sum(row["accepted"] for row in history),
              "validation": measure(editor, [e for e in examples if e.split == "validation"])}
    return report


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def export_verified(editor, tokenizer, examples, config, output, deployment_dtype,
                    reload_model, atol=0.05, rtol=0.01, manifest=None):
    """Stream finite-anchor references to disk, merge/cast, reload native HF model.

    The success marker is written only after reload parity AND actual base
    retention budgets pass on fitting and validation anchors in deployment dtype.
    Full-vocabulary references can require substantial temporary disk space.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if manifest is not None:
        (output / "training_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    shared = editor.shared
    with tempfile.TemporaryDirectory(prefix="static-edit-reference-", dir=output.parent) as temporary:
        temporary = Path(temporary)
        for i, e in enumerate(examples):
            factor, labels = selected_logits(model_logits(editor.model, e), e)
            with editor.base():
                base, _ = selected_logits(model_logits(editor.model, e), e)
            torch.save({"factor": factor.cpu(), "base": base.cpu(), "labels": labels.cpu()}, temporary / f"{i}.pt")
        editor.merge()
        editor.model.to(dtype=deployment_dtype)
        if tied_weights(editor.model) != shared:
            raise RuntimeError("Casting lost endpoint weight sharing")

        def verify(model):
            rows, max_error = [], 0.0
            for i, e in enumerate(examples):
                reference = torch.load(temporary / f"{i}.pt", weights_only=True)
                values, labels = selected_logits(model_logits(model, e), e)
                values, labels = values.cpu(), labels.cpu()
                if not torch.isfinite(values).all() or not torch.allclose(values, reference["factor"], atol=atol, rtol=rtol):
                    raise RuntimeError(f"Merged/cast/reloaded parity failed for {e.id}")
                max_error = max(max_error, (values - reference["factor"]).abs().max().item())
                base_logp, logp = reference["base"].log_softmax(-1), values.log_softmax(-1)
                nll = -logp.gather(-1, labels[:, None]).mean().item()
                base_nll = -base_logp.gather(-1, labels[:, None]).mean().item()
                rows.append({"id": e.id, "split": e.split, "role": e.role, "nll": nll,
                             "base_nll": base_nll, "nll_increase": nll - base_nll,
                             "kl": (base_logp.exp() * (base_logp - logp)).sum(-1).mean().clamp_min(0).item()})
            passed, protection = within_budgets(rows, config)
            if not passed:
                raise RuntimeError(f"Deployment retention budgets failed: {protection}")
            return {"max_selected_logit_error": max_error, "protection": protection, "metrics": rows}

        merged_report = verify(editor.model)
        # Do not carry a source checkpoint's generation penalties or hard masks
        # into the native artifact. EOS/BOS/PAD retain their ordinary semantics.
        from transformers import GenerationConfig
        original_generation = editor.model.generation_config
        editor.model.generation_config = GenerationConfig(
            bos_token_id=original_generation.bos_token_id,
            eos_token_id=original_generation.eos_token_id,
            pad_token_id=original_generation.pad_token_id)
        editor.model.save_pretrained(output, safe_serialization=True)
        tokenizer.save_pretrained(output)
        # Release accelerator residency before loading a second checkpoint.
        editor.model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        reloaded = reload_model(output)
        reloaded.eval()
        if tied_weights(reloaded) != shared:
            raise RuntimeError("Reload lost endpoint weight sharing")
        reloaded_report = verify(reloaded)
    files = {p.name: sha256_file(p) for p in output.iterdir() if p.is_file()}
    report = {"verified": True, "runtime_router": False, "runtime_guard": False,
              "shared_endpoints": shared, "deployment_dtype": str(deployment_dtype),
              "parity_atol": atol, "parity_rtol": rtol,
              "merged": merged_report, "reloaded": reloaded_report, "file_sha256": files}
    (output / "static_edit_export.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
