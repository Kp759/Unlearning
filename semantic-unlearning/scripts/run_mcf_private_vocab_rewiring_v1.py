#!/usr/bin/env python3
"""Train materialized private-vocabulary factual unlearning on MCF.

V1 repurposes pre-existing Llama reserved tokenizer slots as private lexical
clones of the forget subjects. The original tokenizer vocabulary size, model
shape, Transformer, and LM head stay fixed. During optimization only one compact
vector per unique forget subject is trainable. At completion those vectors are
materialized into the selected reserved input-embedding rows and an ordinary
checkpoint + tokenizer are saved.

This is behavioral rerouting, not latent knowledge deletion: the frozen
Transformer may still contain the original fact, and restoring the original
subject tokenization can bypass the reroute.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any, Dict, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import mcf_private_vocab_rewiring_v1_core as core


FILES = {
    "forget": "training_visible_forget_direct.json",
    "protection_fit": "training_visible_protection_fit_direct.json",
    "protection_development": "training_visible_protection_development_direct.json",
    "protection_certification": "training_visible_protection_certification_direct.json",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--protocol-dir", required=True)
    p.add_argument("--experiment-registry", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--clone-init-steps", type=int, default=150)
    p.add_argument("--clone-init-batch-size", type=int, default=16)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--forget-batch-size", type=int, default=8)
    p.add_argument("--retain-batch-size", type=int, default=16)
    p.add_argument("--check-every", type=int, default=25)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--minimum-forget-margin", type=float, default=0.1)
    p.add_argument("--train-margin-target", type=float, default=0.1)
    p.add_argument("--retain-kl-weight", type=float, default=20.0)
    p.add_argument("--anchor-weight", type=float, default=1e-3)
    p.add_argument("--relative-row-cap", type=float, default=0.5)
    p.add_argument("--topk", type=int, default=64)
    p.add_argument("--clone-init-kl-max", type=float, default=5e-3)
    p.add_argument("--retain-kl-mean-max", type=float, default=1e-4)
    p.add_argument("--nonclone-certification-prompts", type=int, default=64)
    p.add_argument("--save-model", action="store_true")
    args = p.parse_args(list(argv) if argv is not None else None)
    if args.seed != 1 or args.forget_num != 50:
        p.error("V1 is locked to consumed seed 1 / 50 forget facts")
    if args.train_margin_target < args.minimum_forget_margin:
        p.error("training margin target cannot be below final success margin")
    if args.clone_init_steps < 0 or args.steps <= 0:
        p.error("invalid step count")
    if args.check_every <= 0:
        p.error("check interval must be positive")
    if args.relative_row_cap <= 0:
        p.error("relative row cap must be positive")
    return args


def dtype_from_name(name: str) -> torch.dtype:
    return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[name]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_protocol(protocol_dir: Path, forget_num: int) -> Dict[str, list[Dict[str, Any]]]:
    manifest = load_json(protocol_dir / "split_manifest.json")
    if manifest.get("official_retain_text_serialized") is not False:
        raise RuntimeError("official retain text must remain sealed")
    if int(manifest["partition_counts"]["forget"]) != int(forget_num):
        raise RuntimeError("forget partition count mismatch")
    out: Dict[str, list[Dict[str, Any]]] = {}
    for role, filename in FILES.items():
        rows = load_json(protocol_dir / filename)
        if not isinstance(rows, list):
            raise RuntimeError(f"protocol role {role} is not a list")
        out[role] = rows
    return out


def validate_registry(registry: Mapping[str, Any]) -> None:
    arch = registry.get("architecture", {})
    if (
        registry.get("protocol") != core.PROTOCOL
        or arch.get("trainable_parameter_families") != ["private_subject_embedding_rows"]
        or arch.get("transformer_frozen") is not True
        or arch.get("lm_head_frozen_bit_identical") is not True
        or arch.get("original_input_embedding_rows_frozen") is not True
        or arch.get("reserved_vocab_slots_repurposed") is not True
        or arch.get("runtime_matcher") is not False
        or arch.get("external_sidecar") is not False
    ):
        raise RuntimeError("private-vocabulary registry contract mismatch")


def ensure_untied_input_embedding(model: Any) -> nn.Module:
    inp = model.get_input_embeddings()
    out = model.get_output_embeddings()
    if out is not None and inp.weight.data_ptr() == out.weight.data_ptr():
        replacement = nn.Embedding(
            inp.num_embeddings,
            inp.embedding_dim,
            padding_idx=getattr(inp, "padding_idx", None),
            device=inp.weight.device,
            dtype=inp.weight.dtype,
        )
        with torch.no_grad():
            replacement.weight.copy_(inp.weight)
        model.set_input_embeddings(replacement)
        if hasattr(model.config, "tie_word_embeddings"):
            model.config.tie_word_embeddings = False
        inp = replacement
    return inp


def render_prompt(record: Mapping[str, Any]) -> str:
    rr = record["requested_rewrite"]
    return str(rr["prompt"]).format(str(rr["subject"]))


def answer_ids(tokenizer: Any, answer: str) -> list[int]:
    text = str(answer).strip()
    if not text:
        raise ValueError("empty target answer")
    ids = tokenizer(
        " " + text,
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]
    if not ids:
        raise ValueError(f"target answer tokenizes empty: {answer!r}")
    return [int(value) for value in ids]


def sequence_logprob_batch(
    model: Any,
    tokenizer: Any,
    prompts: Sequence[str],
    answers: Sequence[str],
    *,
    device: torch.device,
) -> torch.Tensor:
    rows: list[list[int]] = []
    starts: list[int] = []
    targets: list[list[int]] = []
    bos = getattr(tokenizer, "bos_token_id", None)
    for prompt, answer in zip(prompts, answers):
        pids = tokenizer(
            prompt,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
        pids = [int(v) for v in pids]
        if bos is not None:
            pids = [int(bos)] + pids
        aids = answer_ids(tokenizer, answer)
        if len(pids) < 1:
            raise RuntimeError("prompt requires at least one prediction position")
        rows.append(pids + aids)
        starts.append(len(pids))
        targets.append(aids)
    max_len = max(len(row) for row in rows)
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is None:
        pad = getattr(tokenizer, "eos_token_id", None)
    if pad is None:
        pad = 0
    input_ids = torch.full(
        (len(rows), max_len), int(pad), device=device, dtype=torch.long
    )
    attention = torch.zeros_like(input_ids)
    for i, row in enumerate(rows):
        input_ids[i, : len(row)] = torch.tensor(row, device=device, dtype=torch.long)
        attention[i, : len(row)] = 1
    logits = model(input_ids=input_ids, attention_mask=attention).logits
    log_probs = F.log_softmax(logits.float(), dim=-1)
    values = []
    for i, (start, aids) in enumerate(zip(starts, targets)):
        positions = torch.arange(
            start - 1, start - 1 + len(aids), device=device, dtype=torch.long
        )
        token_ids = torch.tensor(aids, device=device, dtype=torch.long)
        selected = log_probs[i, positions, token_ids]
        values.append(selected.mean())
    return torch.stack(values)


def margin_batch(
    model: Any,
    tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
) -> torch.Tensor:
    prompts = [render_prompt(record) for record in records]
    new_answers = [str(record["requested_rewrite"]["target_new"]["str"]) for record in records]
    true_answers = [str(record["requested_rewrite"]["target_true"]["str"]) for record in records]
    new_lp = sequence_logprob_batch(model, tokenizer, prompts, new_answers, device=device)
    true_lp = sequence_logprob_batch(model, tokenizer, prompts, true_answers, device=device)
    return new_lp - true_lp


def final_logits_batch(
    model: Any,
    tokenizer: Any,
    texts: Sequence[str],
    *,
    device: torch.device,
) -> torch.Tensor:
    bos = getattr(tokenizer, "bos_token_id", None)
    encoded = []
    for text in texts:
        ids = tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]
        ids = [int(v) for v in ids]
        if bos is not None:
            ids = [int(bos)] + ids
        if not ids:
            raise RuntimeError("empty context")
        encoded.append(ids)
    max_len = max(map(len, encoded))
    pad = getattr(tokenizer, "pad_token_id", None)
    if pad is None:
        pad = getattr(tokenizer, "eos_token_id", 0)
    input_ids = torch.full((len(encoded), max_len), int(pad), device=device, dtype=torch.long)
    attention = torch.zeros_like(input_ids)
    lengths = []
    for i, ids in enumerate(encoded):
        input_ids[i, : len(ids)] = torch.tensor(ids, device=device, dtype=torch.long)
        attention[i, : len(ids)] = 1
        lengths.append(len(ids))
    logits = model(input_ids=input_ids, attention_mask=attention).logits.float()
    index = torch.tensor([length - 1 for length in lengths], device=device, dtype=torch.long)
    batch = torch.arange(len(encoded), device=device, dtype=torch.long)
    return logits[batch, index]


def topk_teacher_kl(
    model: Any,
    base_tokenizer: Any,
    clone_tokenizer: Any,
    texts: Sequence[str],
    *,
    device: torch.device,
    topk: int,
) -> torch.Tensor:
    if not texts:
        return torch.zeros((), device=device, dtype=torch.float32)
    with torch.no_grad():
        teacher = final_logits_batch(model, base_tokenizer, texts, device=device)
        k = min(int(topk), int(teacher.shape[-1]))
        _, indices = torch.topk(teacher, k=k, dim=-1)
        teacher_selected = teacher.gather(1, indices)
        teacher_prob = F.softmax(teacher_selected, dim=-1)
    student = final_logits_batch(model, clone_tokenizer, texts, device=device)
    student_selected = student.gather(1, indices)
    student_logprob = F.log_softmax(student_selected, dim=-1)
    return F.kl_div(student_logprob, teacher_prob, reduction="batchmean")


def make_retain_contexts(
    forget_records: Sequence[Mapping[str, Any]],
    protection_fit: Sequence[Mapping[str, Any]],
) -> tuple[list[str], Dict[str, int]]:
    subjects = core.unique_subjects(forget_records)
    contexts: list[str] = []
    for subject in subjects:
        contexts.extend(core.generic_subject_contexts(subject))
    subject_set = set(subjects)
    overlaps = []
    for record in protection_fit:
        rr = record["requested_rewrite"]
        if str(rr["subject"]) in subject_set:
            overlaps.append(render_prompt(record))
    contexts.extend(overlaps)
    contexts = list(dict.fromkeys(contexts))
    return contexts, {
        "generic_subject_contexts": 4 * len(subjects),
        "same_subject_protection_fit_contexts": len(overlaps),
        "total_unique_contexts": len(contexts),
    }


def clone_equivalence_contexts(forget_records: Sequence[Mapping[str, Any]]) -> list[str]:
    contexts: list[str] = []
    for record in forget_records:
        subject = str(record["requested_rewrite"]["subject"])
        contexts.extend(core.generic_subject_contexts(subject))
        contexts.append(render_prompt(record))
    return list(dict.fromkeys(contexts))


def mean_kl_over_contexts(
    model: Any,
    base_tokenizer: Any,
    clone_tokenizer: Any,
    contexts: Sequence[str],
    *,
    device: torch.device,
    topk: int,
    batch_size: int = 16,
) -> float:
    if not contexts:
        return 0.0
    total = 0.0
    count = 0
    for start in range(0, len(contexts), batch_size):
        batch = contexts[start : start + batch_size]
        value = topk_teacher_kl(
            model,
            base_tokenizer,
            clone_tokenizer,
            batch,
            device=device,
            topk=topk,
        )
        total += float(value.detach().item()) * len(batch)
        count += len(batch)
    return total / max(count, 1)


def evaluate_margins(
    model: Any,
    tokenizer: Any,
    records: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    batch_size: int = 8,
) -> list[float]:
    out: list[float] = []
    with torch.no_grad():
        for start in range(0, len(records), batch_size):
            batch = records[start : start + batch_size]
            out.extend(
                float(v)
                for v in margin_batch(
                    model, tokenizer, batch, device=device
                ).detach().cpu().tolist()
            )
    return out


def margin_summary(values: Sequence[float], threshold: float) -> Dict[str, Any]:
    return {
        "count": len(values),
        "minimum": min(values) if values else None,
        "mean": sum(values) / len(values) if values else None,
        "passed": sum(float(v) >= float(threshold) for v in values),
        "failures": sum(float(v) < float(threshold) for v in values),
        "threshold": float(threshold),
    }


def nonclone_tokenization_certification(
    base_tokenizer: Any,
    clone_tokenizer: Any,
    protection_fit: Sequence[Mapping[str, Any]],
    forget_subjects: Sequence[str],
    *,
    maximum: int,
) -> Dict[str, Any]:
    subject_set = set(forget_subjects)
    checked = 0
    mismatches: list[Dict[str, Any]] = []
    for record in protection_fit:
        if str(record["requested_rewrite"]["subject"]) in subject_set:
            continue
        text = render_prompt(record)
        base_ids = base_tokenizer(text, add_special_tokens=False)["input_ids"]
        clone_ids = clone_tokenizer(text, add_special_tokens=False)["input_ids"]
        if base_ids != clone_ids:
            mismatches.append({"case_id": int(record["case_id"]), "text": text})
        checked += 1
        if checked >= int(maximum):
            break
    return {
        "checked": checked,
        "tokenization_mismatches": len(mismatches),
        "mismatch_examples": mismatches[:5],
        "passed": checked > 0 and not mismatches,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "method").mkdir()

    registry = load_json(Path(args.experiment_registry))
    validate_registry(registry)
    protocol = load_protocol(Path(args.protocol_dir), int(args.forget_num))
    forget = protocol["forget"]
    protection_fit = protocol["protection_fit"]

    dtype = dtype_from_name(args.dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    input_embedding = ensure_untied_input_embedding(model)
    output_embedding = model.get_output_embeddings()
    if output_embedding is None:
        raise RuntimeError("causal LM lacks an output embedding / LM head")
    if input_embedding.weight.data_ptr() == output_embedding.weight.data_ptr():
        raise RuntimeError("input embedding and LM head remain tied")

    base_tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    subjects = core.unique_subjects(forget)
    mapping = core.build_subject_slot_mapping(base_tokenizer, subjects)
    private_ids = [int(item["private_token_id"]) for item in mapping]

    tokenizer_dir = output / "private_tokenizer"
    base_tokenizer.save_pretrained(tokenizer_dir)
    core.patch_saved_tokenizer_reserved_slots(tokenizer_dir, mapping)
    clone_tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, use_fast=True)
    if len(clone_tokenizer) != len(base_tokenizer):
        raise RuntimeError("private tokenizer changed vocabulary size")
    core.validate_subject_routing(clone_tokenizer, mapping)

    tokenization_cert = nonclone_tokenization_certification(
        base_tokenizer,
        clone_tokenizer,
        protection_fit,
        subjects,
        maximum=int(args.nonclone_certification_prompts),
    )
    if not tokenization_cert["passed"]:
        raise RuntimeError("private tokenizer changed a non-forget protection prompt")

    lm_head_hash_before = core.sha256_tensor(output_embedding.weight)
    nonprivate_embedding_hash_before = core.non_private_row_hash(
        input_embedding.weight, private_ids
    )

    initial_rows = core.initialize_clone_rows(input_embedding.weight, mapping)
    controller = core.PrivateRowController(private_ids, initial_rows).to(device)
    hook = core.EmbeddingHook.install(input_embedding, controller)
    optimizer = torch.optim.AdamW(
        [controller.rows], lr=float(args.learning_rate), weight_decay=0.0
    )

    retain_contexts, retain_context_stats = make_retain_contexts(forget, protection_fit)
    init_contexts = clone_equivalence_contexts(forget)
    rng = random.Random(int(args.seed) + 941)

    base_margins = evaluate_margins(
        model, base_tokenizer, forget, device=device, batch_size=int(args.forget_batch_size)
    )
    initial_clone_margins = evaluate_margins(
        model, clone_tokenizer, forget, device=device, batch_size=int(args.forget_batch_size)
    )
    initial_clone_kl = mean_kl_over_contexts(
        model,
        base_tokenizer,
        clone_tokenizer,
        init_contexts,
        device=device,
        topk=int(args.topk),
    )

    init_log = []
    for step in range(1, int(args.clone_init_steps) + 1):
        batch = rng.sample(init_contexts, min(int(args.clone_init_batch_size), len(init_contexts)))
        optimizer.zero_grad(set_to_none=True)
        loss = topk_teacher_kl(
            model,
            base_tokenizer,
            clone_tokenizer,
            batch,
            device=device,
            topk=int(args.topk),
        )
        loss.backward()
        optimizer.step()
        cap = controller.enforce_relative_cap(float(args.relative_row_cap))
        if step == 1 or step % int(args.check_every) == 0 or step == int(args.clone_init_steps):
            mean_kl = mean_kl_over_contexts(
                model,
                base_tokenizer,
                clone_tokenizer,
                init_contexts,
                device=device,
                topk=int(args.topk),
            )
            row = {"step": step, "batch_kl": float(loss.detach().item()), "mean_kl": mean_kl, **cap}
            init_log.append(row)
            print(
                f"clone-init step {step:4d}: batch KL={row['batch_kl']:.6g}, "
                f"mean KL={mean_kl:.6g}, max rel delta={cap['max_relative_delta']:.4f}",
                flush=True,
            )

    post_init_kl = mean_kl_over_contexts(
        model,
        base_tokenizer,
        clone_tokenizer,
        init_contexts,
        device=device,
        topk=int(args.topk),
    )
    post_init_margins = evaluate_margins(
        model, clone_tokenizer, forget, device=device, batch_size=int(args.forget_batch_size)
    )
    if post_init_kl > float(args.clone_init_kl_max):
        raise RuntimeError(
            f"clone equivalence preflight failed: KL {post_init_kl:.6g} > {args.clone_init_kl_max}"
        )

    with torch.no_grad():
        controller.initial_rows.copy_(controller.rows.detach())
    optimizer = torch.optim.AdamW(
        [controller.rows], lr=float(args.learning_rate), weight_decay=0.0
    )

    training_log: list[Dict[str, Any]] = []
    best_state = controller.rows.detach().clone()
    best_score = float("inf")
    for step in range(1, int(args.steps) + 1):
        forget_batch = rng.sample(forget, min(int(args.forget_batch_size), len(forget)))
        retain_batch = rng.sample(
            retain_contexts, min(int(args.retain_batch_size), len(retain_contexts))
        )
        optimizer.zero_grad(set_to_none=True)
        margins = margin_batch(model, clone_tokenizer, forget_batch, device=device)
        forget_loss = F.relu(float(args.train_margin_target) - margins).square().mean()
        retain_kl = topk_teacher_kl(
            model,
            base_tokenizer,
            clone_tokenizer,
            retain_batch,
            device=device,
            topk=int(args.topk),
        )
        anchor = (controller.rows - controller.initial_rows).float().pow(2).mean()
        loss = (
            forget_loss
            + float(args.retain_kl_weight) * retain_kl
            + float(args.anchor_weight) * anchor
        )
        loss.backward()
        optimizer.step()
        cap = controller.enforce_relative_cap(float(args.relative_row_cap))

        if step == 1 or step % int(args.check_every) == 0 or step == int(args.steps):
            all_margins = evaluate_margins(
                model,
                clone_tokenizer,
                forget,
                device=device,
                batch_size=int(args.forget_batch_size),
            )
            summary = margin_summary(all_margins, float(args.minimum_forget_margin))
            retain_mean_kl = mean_kl_over_contexts(
                model,
                base_tokenizer,
                clone_tokenizer,
                retain_contexts,
                device=device,
                topk=int(args.topk),
            )
            score = float(summary["failures"]) * 1000.0 + retain_mean_kl
            if score < best_score:
                best_score = score
                best_state = controller.rows.detach().clone()
            row = {
                "step": step,
                "loss": float(loss.detach().item()),
                "forget_loss": float(forget_loss.detach().item()),
                "retain_batch_kl": float(retain_kl.detach().item()),
                "anchor": float(anchor.detach().item()),
                "direct": summary,
                "retain_mean_kl": retain_mean_kl,
                **cap,
            }
            training_log.append(row)
            print(
                f"step {step:4d}: direct fail={summary['failures']}, "
                f"min margin={summary['minimum']:.4f}, retain KL={retain_mean_kl:.6g}, "
                f"max rel delta={cap['max_relative_delta']:.4f}",
                flush=True,
            )
            if summary["failures"] == 0 and retain_mean_kl <= float(args.retain_kl_mean_max):
                print("registered training gate reached; stopping early", flush=True)
                break

    with torch.no_grad():
        controller.rows.copy_(best_state)
    final_margins = evaluate_margins(
        model, clone_tokenizer, forget, device=device, batch_size=int(args.forget_batch_size)
    )
    final_summary = margin_summary(final_margins, float(args.minimum_forget_margin))
    final_retain_kl = mean_kl_over_contexts(
        model,
        base_tokenizer,
        clone_tokenizer,
        retain_contexts,
        device=device,
        topk=int(args.topk),
    )

    core.materialize_private_rows(input_embedding.weight, controller)
    hook.remove()
    lm_head_hash_after = core.sha256_tensor(output_embedding.weight)
    nonprivate_embedding_hash_after = core.non_private_row_hash(
        input_embedding.weight, private_ids
    )
    if lm_head_hash_before != lm_head_hash_after:
        raise RuntimeError("LM head changed during private-vocabulary training")
    if nonprivate_embedding_hash_before != nonprivate_embedding_hash_after:
        raise RuntimeError("a non-private input embedding row changed")

    method = {
        "protocol": core.PROTOCOL,
        "seed": int(args.seed),
        "unique_forget_subjects": len(subjects),
        "subject_mapping": mapping,
        "tokenizer_vocab_size_unchanged": len(clone_tokenizer) == len(base_tokenizer),
        "nonclone_tokenization_certification": tokenization_cert,
        "retain_contexts": retain_context_stats,
        "clone_initialization": {
            "initial_mean_kl": initial_clone_kl,
            "post_init_mean_kl": post_init_kl,
            "registered_max": float(args.clone_init_kl_max),
            "log": init_log,
        },
        "margins": {
            "base": margin_summary(base_margins, float(args.minimum_forget_margin)),
            "raw_clone": margin_summary(initial_clone_margins, float(args.minimum_forget_margin)),
            "base_equivalent_clone": margin_summary(post_init_margins, float(args.minimum_forget_margin)),
            "final": final_summary,
            "final_per_case": [
                {"case_id": int(record["case_id"]), "margin": float(margin)}
                for record, margin in zip(forget, final_margins)
            ],
        },
        "final_retain_mean_kl": final_retain_kl,
        "integrity": {
            "lm_head_sha256_before": lm_head_hash_before,
            "lm_head_sha256_after": lm_head_hash_after,
            "lm_head_bit_identical": lm_head_hash_before == lm_head_hash_after,
            "nonprivate_input_rows_sha256_before": nonprivate_embedding_hash_before,
            "nonprivate_input_rows_sha256_after": nonprivate_embedding_hash_after,
            "nonprivate_input_rows_bit_identical": nonprivate_embedding_hash_before
            == nonprivate_embedding_hash_after,
            "materialized_private_row_ids": private_ids,
        },
        "claim_boundary": {
            "behavioral_unlearning": True,
            "latent_erasure_claimed": False,
            "restoring_original_tokenizer_can_bypass": True,
            "runtime_matcher": False,
            "external_sidecar": False,
        },
        "training_log": training_log,
    }
    (output / "method" / "private_vocab_rewiring.json").write_text(
        json.dumps(method, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    torch.save(
        {
            "private_token_ids": private_ids,
            "private_rows": controller.rows.detach().cpu(),
            "subject_mapping": mapping,
            "base_model": str(args.model_path),
        },
        output / "method" / "private_rows.pt",
    )

    if args.save_model:
        model_dir = output / "model"
        model.save_pretrained(model_dir)
        clone_tokenizer.save_pretrained(model_dir)
        reloaded = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
        core.validate_subject_routing(reloaded, mapping)

    completion = {
        "direct": final_summary,
        "retain_mean_kl": final_retain_kl,
        "retain_gate": final_retain_kl <= float(args.retain_kl_mean_max),
        "integrity_passed": True,
        "checkpoint_saved": bool(args.save_model),
    }
    (output / "method" / "completion.json").write_text(
        json.dumps(completion, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(completion, indent=2), flush=True)


if __name__ == "__main__":
    main()
