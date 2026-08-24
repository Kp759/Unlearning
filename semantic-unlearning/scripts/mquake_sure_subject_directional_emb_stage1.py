#!/usr/bin/env python3
"""SURE-MQuAKE subject-keyed directional GA on input embeddings only.

Port of ``mcf_sure_subject_directional_emb_stage1.py``. The architecture is
unchanged -- train only the subject's input-embedding rows, keep every
transformer block and the LM head frozen and untied -- but MQuAKE's objective
differs from MCF's in one structural way that the loss has to follow.

MCF supplies both ``target_true`` (sensitive) and ``target_new`` (a neutral
reference), so forgetting is a *pairwise* margin: prefer target_new over
target_true. MQuAKE's locked training view carries ``target_true`` only (see
``build_mquake_sure_canonical_split.py``), and its official Eff metric is
different in kind -- ``mquake_zero_unlearn_official_eval`` teacher-forces the
sensitive answer and scores *next-token argmax accuracy*, where "0.00 means no
evaluated sensitive token remains argmax".

So the reference here is not a supplied replacement answer but **whatever the
model would otherwise say**: the highest-logit non-sensitive token, cached once
on the base model. Forgetting means driving the sensitive token below that
competitor by a margin.

    L = margin_weight * relu( train_margin
                              - [ logp(competitor) - logp(target_true) ] )
      + delta_l2 * ||Delta||^2

The competitor is a *stable* reference in this architecture for free: the only
edited parameters are input-embedding rows, so no output row is ever modified
and the competitor token's LM-head row is identical to base throughout. That is
not true of LM-head editing methods, which have to explicitly exclude edited
rows when choosing a competitor (compare
``mquake_exact_top2_sparse_repair_v2.cache_fixed_competitors``, which calls
``logits.index_fill_(1, edited, -inf)`` for exactly that reason).

Everything else is inherited from the MCF script: subject-token row selection
with mandatory direct-prompt liveness, full-coverage row selection, the scale
sweep, and materialization into the embedding weight.

Data firewall: only ``requested_rewrite`` prompt/subject/target_true of the
locked forget split are read. MQuAKE's atomic ``question`` field (AtomicGen),
the instance-level multi-hop questions, retain facts and PPL text are never
loaded here.

Scope note: this edits the *subject* of an atomic fact, so it fires on any
prompt that mentions that subject -- the atomic rewrite prompt (Eff) and the
natural-language single-hop question (AtomicGen). Multi-hop questions that
reference the subject only descriptively, via an earlier hop, contain none of
the edited rows and are therefore out of scope by construction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F

import gagd_compare as gagd
import mcf_sure_directional_emb_lm_stage1 as base1
import mcf_sure_subject_directional_emb_stage1 as subj
import sure_canonical_core as core
import sure_context_projection as context
import sure_stage2_sparse_repair as stage2

METHOD = "SURE-MQuAKE-subject-keyed-directional-embedding-stage1"
PROTOCOL = "mquake_target_true_sensitive_subject_keyed_embedding_ga_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--training-visible-path", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument(
        "--forget-num",
        type=int,
        default=50,
        help=(
            "FLATTENED atomic-fact count, not the official instance count. "
            "MQuAKE is sampled at the instance level and requested_rewrite "
            "facts are flattened only afterwards, so the training-visible file "
            "holds more records than instances sampled. load_locked checks this "
            "against sampling.forget_num in the manifest, which is the flattened "
            "count; sampling.forget_num_instances is the instance count that the "
            "split builder and the official eval both take."
        ),
    )

    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--cache-batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--margin-weight", type=float, default=100.0)
    p.add_argument(
        "--train-margin",
        type=float,
        default=6.0,
        help=(
            "Training hinge target on logp(competitor) - logp(target_true). "
            "6.0 is the knee of the Gen/PPL frontier on MCF; carried over as a "
            "starting point, not as a tuned MQuAKE value."
        ),
    )
    p.add_argument("--delta-l2", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument(
        "--stage1-constraint-margin",
        type=float,
        default=0.05,
        help="Reported failure threshold on the competitor margin.",
    )
    p.add_argument(
        "--max-subject-token-frequency",
        type=int,
        default=10**9,
        help=(
            "Effectively off, matching the MCF finding: filtering starved rows "
            "(~1.5 of 3-5 subject tokens edited) and the model re-identified the "
            "entity from the untouched remainder."
        ),
    )
    p.add_argument("--wikidata-dir", default="data/wikidata")
    p.add_argument("--frequency-docs", type=int, default=5000)
    p.add_argument("--frequency-doc-start", type=int, default=20)
    p.add_argument("--row-norm-cap", type=float, default=0.0)
    p.add_argument("--row-norm-cap-frequency-alpha", type=float, default=0.0)
    p.add_argument(
        "--candidate-scales",
        default="1,.875,.75,.625,.5,.375,.25,.1875,.125,.09375,.0625,.046875,.03125,.015625,.0078125,0",
    )
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--device-map", default="single")
    a = p.parse_args(argv)
    if int(a.frequency_docs) > 0 and int(a.frequency_doc_start) < 20:
        p.error("--frequency-doc-start must be >= 20 to stay clear of official PPL docs")
    if float(a.train_margin) <= 0:
        p.error("--train-margin must be positive")
    return a


def validate_locked_mquake(
    visible_path: Path, manifest_path: Path, seed: int, forget_num: int
) -> Tuple[List[Mapping[str, Any]], Mapping[str, Any]]:
    records, manifest = stage2.load_locked(
        "mquake", visible_path, manifest_path, seed, forget_num
    )
    for index, record in enumerate(records):
        rewrite = record.get("requested_rewrite")
        if not isinstance(rewrite, Mapping):
            raise RuntimeError(f"record {index} lacks requested_rewrite")
        if not str(rewrite.get("subject", "")).strip():
            raise RuntimeError(f"record {index} lacks a subject")
        if not str((rewrite.get("target_true") or {}).get("str", "")).strip():
            raise RuntimeError(f"record {index} lacks target_true")
        # MQuAKE's atomic question is the AtomicGen probe and must stay unseen.
        if record.get("atomic_gen_prompt") or record.get("questions"):
            raise RuntimeError(f"record {index} exposes held-out MQuAKE probes")
    return records, manifest


@torch.no_grad()
def cache_base_competitors(
    model: torch.nn.Module,
    tok: Any,
    cases: Sequence[core.SensitivePredictionCase],
    device: torch.device,
    batch_size: int,
    llama_like: bool,
) -> torch.Tensor:
    """Highest-logit non-sensitive token per case, from the BASE model.

    MQuAKE supplies no ``target_new``, so the reference is whatever the model
    would otherwise answer. Cached once before any delta is active, so the
    target is fixed rather than chasing the edit.
    """
    picks: List[torch.Tensor] = []
    for start in range(0, len(cases), batch_size):
        batch = list(cases[start : start + batch_size])
        logits = core.forward_last_logits(model, tok, batch, device).float()
        tids = core.official_target_ids(
            tok, batch, llama_like=llama_like, device=device
        )
        rows = torch.arange(logits.shape[0], device=logits.device)
        # Mask only the sensitive token itself; every other row is untouched by
        # this architecture, so no further exclusion is needed.
        logits[rows, tids] = float("-inf")
        picks.append(logits.argmax(dim=-1).detach().cpu())
    return torch.cat(picks) if picks else torch.empty(0, dtype=torch.long)


@torch.no_grad()
def competitor_margins(
    model: torch.nn.Module,
    tok: Any,
    cases: Sequence[core.SensitivePredictionCase],
    competitors: torch.Tensor,
    device: torch.device,
    batch_size: int,
    llama_like: bool,
) -> torch.Tensor:
    """logp(competitor) - logp(target_true) per case. Positive = forgotten."""
    out: List[torch.Tensor] = []
    for start in range(0, len(cases), batch_size):
        batch = list(cases[start : start + batch_size])
        logits = core.forward_last_logits(model, tok, batch, device).float()
        logp = F.log_softmax(logits, dim=-1)
        tids = core.official_target_ids(
            tok, batch, llama_like=llama_like, device=device
        )
        comp = competitors[start : start + len(batch)].to(logp.device)
        rows = torch.arange(logp.shape[0], device=logp.device)
        out.append((logp[rows, comp] - logp[rows, tids]).detach().cpu())
    return torch.cat(out) if out else torch.empty(0)


def run_competitor_stage1(
    a: argparse.Namespace,
    records: Sequence[Mapping[str, Any]],
    manifest: Mapping[str, Any],
    *,
    dataset: str,
    method: str,
    protocol: str,
    extra_config: Mapping[str, Any] | None = None,
) -> None:
    """Shared Stage 1 for the datasets whose locked view has no target_new.

    MQuAKE and canonical ZsRE both supply target_true only -- ZsRE's loader
    even raises if target_new is present ("Canonical ZsRE Stage 2 forbids
    target_new/neutral targets") -- and both score forgetting by accuracy on
    the sensitive answer rather than by a pairwise preference. The objective is
    therefore identical for the two, so it lives here once rather than being
    copied per dataset.
    """
    ns = argparse.Namespace(
        model_path=a.model_path,
        dtype=a.dtype,
        device_map=a.device_map,
        gradient_checkpointing=False,
    )
    model, tok = gagd.load_model_and_tokenizer(ns, for_training=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    output_layer = core.untie_and_freeze_output_head(model)
    input_layer = model.get_input_embeddings()
    if input_layer is None:
        raise RuntimeError("model has no input embedding layer")
    if input_layer.weight.data_ptr() == output_layer.weight.data_ptr():
        raise RuntimeError("embedding and LM head must be untied before Stage 1")
    output_layer.weight.requires_grad_(False)
    device = gagd.first_device(model)
    llama_like = core.is_llama_like(model, tok)

    vocab_size = int(input_layer.weight.shape[0])
    frequency_documents = subj.load_frequency_documents(
        a.wikidata_dir, int(a.frequency_doc_start), int(a.frequency_docs)
    )
    counts = subj.token_frequency_counts(tok, frequency_documents, vocab_size)

    # MQuAKE's locked view has no target_new and no relation bank coverage, so
    # the direct prompt is the only training prompt. Row liveness is therefore
    # measured against it alone.
    direct_live_ids = subj.live_prompt_token_ids(records, tok)
    selected_ids, subject_reports = subj.select_subject_rows(
        records,
        tok,
        llama_like=llama_like,
        counts=counts,
        max_frequency=int(a.max_subject_token_frequency),
        direct_live_ids=direct_live_ids,
        paraphrase_live_ids=None,
    )
    if not selected_ids:
        raise RuntimeError("no subject embedding rows selected")
    threshold_override = [
        r for r in subject_reports if r["direct_row_above_frequency_threshold"]
    ]
    print(
        f"Selected {len(selected_ids)} subject embedding rows across "
        f"{len(records)} MQuAKE atomic facts; {len(threshold_override)} kept a "
        "direct-prompt row above the frequency threshold."
    )

    # Use the shared expansion rather than a local copy. A local version here
    # set target_text to the WHOLE answer at every token position, while
    # core.official_target_ids takes one token column from it -- so every case
    # resolved to the answer's FIRST token. Single-token answers were
    # coincidentally right; multi-token answers optimized positions 1..n
    # against token 0, and Stage 1 was not scoring the decisions official Eff
    # scores. expand_answer_field_cases sets target_text=tok.decode([token_id])
    # per position, matching the evaluator.
    cases = context.expand_answer_field_cases(
        records, tok, field="target_true", llama_like=llama_like
    )
    competitors = cache_base_competitors(
        model, tok, cases, device, int(a.cache_batch_size), llama_like
    )
    base_margins = competitor_margins(
        model, tok, cases, competitors, device, int(a.cache_batch_size), llama_like
    )
    print(
        f"{len(cases)} teacher-forced cases; base competitor margin "
        f"mean={float(base_margins.mean()):.4f} "
        f"min={float(base_margins.min()):.4f} "
        f"(negative = sensitive token still wins)"
    )

    cap_alpha = float(a.row_norm_cap_frequency_alpha)
    base_cap = float(a.row_norm_cap)
    if base_cap > 0 and counts.numel():
        cap_freq = torch.tensor(
            [float(counts[i].item()) for i in selected_ids], dtype=torch.float32
        )
    else:
        cap_freq = torch.zeros(len(selected_ids), dtype=torch.float32)
    row_norm_caps = (
        base_cap / (1.0 + cap_freq).pow(cap_alpha)
        if base_cap > 0
        else torch.zeros(len(selected_ids), dtype=torch.float32)
    )

    hidden_size = int(input_layer.weight.shape[1])
    delta_module = core.SelectedRowDelta(
        len(selected_ids), hidden_size, direction_basis=None,
        device=input_layer.weight.device,
    )
    parameters = list(delta_module.parameters())
    opt = torch.optim.AdamW(parameters, lr=float(a.lr), weight_decay=0.0)
    sampler = core.IndexSampler(len(cases), int(a.batch_size), int(a.seed))

    out_dir = gagd.resolve_output_path(a.output_dir)
    ckpt = out_dir / "checkpoint"
    out_dir.mkdir(parents=True, exist_ok=True)

    emb_hook = base1.register_input_embedding_delta_hook(
        input_layer, selected_ids, delta_module.effective_delta
    )
    rows_touched_ever = torch.zeros(len(selected_ids), dtype=torch.bool)
    try:
        model.eval()
        with (out_dir / "train_log.jsonl").open("w", encoding="utf-8") as log_f:
            for step in range(1, int(a.steps) + 1):
                idx = sampler.next()
                batch = [cases[i] for i in idx]
                opt.zero_grad(set_to_none=True)

                logits = core.forward_last_logits(model, tok, batch, device)
                logp = F.log_softmax(logits.float(), dim=-1)
                rows = torch.arange(logp.shape[0], device=logp.device)
                tids = core.official_target_ids(
                    tok, batch, llama_like=llama_like, device=device
                )
                comp = competitors[torch.tensor(idx, dtype=torch.long)].to(logp.device)

                margin = logp[rows, comp] - logp[rows, tids]
                margin_hinge = F.relu(float(a.train_margin) - margin).mean()
                delta_now = delta_module.effective_delta()
                l2 = delta_now.square().mean()
                total = (
                    float(a.margin_weight) * margin_hinge + float(a.delta_l2) * l2
                )
                if not torch.isfinite(total):
                    raise FloatingPointError(f"Non-finite loss at step {step}")
                total.backward()
                grad_norm = None
                if float(a.grad_clip) > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        parameters, float(a.grad_clip)
                    )
                opt.step()

                if base_cap > 0:
                    with torch.no_grad():
                        raw_p = delta_module.raw_delta
                        if raw_p is not None:
                            caps = row_norm_caps.to(raw_p.device)
                            norms = raw_p.data.norm(dim=-1)
                            scale = (caps / norms.clamp_min(1e-12)).clamp(max=1.0)
                            raw_p.data.mul_(scale.unsqueeze(-1))

                if step == 1 or step % 25 == 0 or step == int(a.steps):
                    raw = delta_module.raw_delta
                    live_rows = 0
                    if raw is not None and raw.grad is not None:
                        touched = raw.grad.detach().abs().sum(dim=-1) > 0
                        live_rows = int(touched.sum().item())
                        rows_touched_ever |= touched.cpu()
                    row = {
                        "step": int(step),
                        "total_loss": float(total.detach().cpu()),
                        "margin_hinge": float(margin_hinge.detach().cpu()),
                        "batch_mean_margin": float(margin.mean().detach().cpu()),
                        "batch_min_margin": float(margin.min().detach().cpu()),
                        "rows_with_nonzero_grad_this_step": live_rows,
                        "rows_touched_so_far": int(rows_touched_ever.sum().item()),
                        "embedding_delta_norm": float(delta_now.detach().norm().cpu()),
                        "lm_head_delta_norm": 0.0,
                        "lora_used": False,
                    }
                    if grad_norm is not None:
                        row["grad_norm"] = float(grad_norm.detach().cpu())
                    log_f.write(json.dumps(row) + "\n")
                    log_f.flush()
    finally:
        emb_hook.remove()
    del opt

    trained_delta = delta_module.effective_delta().detach().clone()

    scale_reports: List[Dict[str, Any]] = []
    for scale in core.parse_scales(a.candidate_scales):
        handle = base1.register_input_embedding_delta_hook(
            input_layer, selected_ids, lambda scale=scale: trained_delta * float(scale)
        )
        try:
            margins = competitor_margins(
                model, tok, cases, competitors, device,
                int(a.cache_batch_size), llama_like,
            )
        finally:
            handle.remove()
        scale_reports.append(
            {
                "scale": float(scale),
                "direct_failures": int(
                    (margins < float(a.stage1_constraint_margin)).sum().item()
                ),
                "direct_only_failures": int(
                    (margins < float(a.stage1_constraint_margin)).sum().item()
                ),
                "minimum_margin": float(margins.min()),
                "embedding_delta_norm": float(trained_delta.norm().cpu() * scale),
            }
        )

    # Invariant that would have caught the target-token bug immediately: at
    # scale 0 the materialized model is exactly Base, so the sensitive answers
    # must still win. If Base already satisfies the objective, Stage 1 is not
    # measuring the decisions the official evaluator measures.
    zero = next((r for r in scale_reports if float(r["scale"]) == 0.0), None)
    if zero is not None and int(zero["direct_failures"]) == 0:
        raise RuntimeError(
            "scale=0 (the Base model) reports 0 competitor-margin failures with "
            f"minimum_margin={zero['minimum_margin']}. Base cannot already be "
            "forgetting, so Stage 1's (prompt, target token) pairs do not match "
            "the official evaluator's. Check the case expansion before trusting "
            "any number from this run."
        )

    selected_scale = base1.select_stage1_scale(scale_reports)
    final_delta = trained_delta * float(selected_scale)
    base1.materialize_input_delta(input_layer, selected_ids, final_delta)

    final_margins = competitor_margins(
        model, tok, cases, competitors, device, int(a.cache_batch_size), llama_like
    )
    failures = [
        int(i)
        for i, v in enumerate(final_margins.tolist())
        if float(v) < float(a.stage1_constraint_margin)
    ]

    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)

    config: Dict[str, Any] = {
        "schema_version": 1,
        "method": method,
        "protocol": protocol,
        "dataset": dataset,
        "source_protocol": manifest.get("protocol"),
        "seed": int(a.seed),
        "forget_num": int(a.forget_num),
        "objective": (
            "relu(train_margin - [logp(competitor) - logp(target_true)]). MQuAKE "
            "supplies no target_new and scores Eff by next-token argmax accuracy "
            "on the sensitive answer, so the reference is the highest-logit "
            "non-sensitive token cached on the BASE model, not a supplied "
            "replacement answer as in MCF"
        ),
        "competitor_reference_is_stable": (
            "only input-embedding rows are edited, so no output row is ever "
            "modified and the competitor's LM-head row stays identical to base; "
            "LM-head editing methods must explicitly exclude edited rows when "
            "picking a competitor"
        ),
        "lm_head_untied_before_training": True,
        "lm_head_trainable_parameters": 0,
        "lm_head_edited": False,
        "transformer_trainable_parameters": 0,
        "editable_embedding_rows": "subject_token_rows_only",
        "selected_token_ids": selected_ids,
        "selected_row_count": len(selected_ids),
        "rows_ever_touched_by_gradient": int(rows_touched_ever.sum().item()),
        "subject_row_selection": subject_reports,
        "records_with_direct_row_above_threshold": len(threshold_override),
        "max_subject_token_frequency": int(a.max_subject_token_frequency),
        "frequency_documents_used": len(frequency_documents),
        "frequency_doc_range": [
            int(a.frequency_doc_start),
            int(a.frequency_doc_start) + int(a.frequency_docs),
        ],
        "row_norm_cap": base_cap,
        "row_norm_cap_frequency_alpha": cap_alpha,
        "margin_weight": float(a.margin_weight),
        "train_margin": float(a.train_margin),
        "delta_l2": float(a.delta_l2),
        "steps": int(a.steps),
        "batch_size": int(a.batch_size),
        "lr": float(a.lr),
        "case_count": len(cases),
        "base_competitor_margin_mean": float(base_margins.mean()),
        "base_competitor_margin_min": float(base_margins.min()),
        "selected_scale": float(selected_scale),
        "scale_reports": scale_reports,
        "stage1_constraint_margin": float(a.stage1_constraint_margin),
        "stage1_direct_failures": len(failures),
        "stage1_failing_positions": failures,
        "stage1_minimum_margin": float(final_margins.min()),
        "final_embedding_delta_norm": float(final_delta.norm().cpu()),
        "final_lm_head_delta_norm": 0.0,
        "lora_used": False,
        "benchmark_retain_seen": 0,
        "ppl_eval_text_seen": 0,
    }
    if extra_config:
        config.update(dict(extra_config))
    core.write_json(out_dir / "stage1_config.json", config)
    print(json.dumps(config, indent=2))
    print(f"Stage-1 checkpoint: {ckpt}")
    print(
        f"Stage-1 competitor-margin failures: {len(failures)}/{len(cases)}; "
        f"min margin={config['stage1_minimum_margin']:.6f}"
    )


def main(argv: Sequence[str] | None = None) -> None:
    a = parse_args(argv)
    gagd.set_seed(a.seed)
    if a.device_map == "single":
        gagd.require_cuda_if_needed(a.device_map)
    records, manifest = validate_locked_mquake(
        Path(a.training_visible_path).resolve(),
        Path(a.split_manifest).resolve(),
        int(a.seed),
        int(a.forget_num),
    )
    run_competitor_stage1(
        a, records, manifest,
        dataset="mquake", method=METHOD, protocol=PROTOCOL,
        extra_config={
            "multihop_scope_note": (
                "edits the atomic fact's subject, so it fires on any prompt "
                "naming that subject (atomic rewrite prompt = Eff, single-hop "
                "question = AtomicGen). Multi-hop questions that reach the "
                "subject only through an earlier hop contain none of the edited "
                "rows and are out of scope by construction"
            ),
            "atomic_gen_seen": 0,
            "multihop_questions_seen": 0,
        },
    )


if __name__ == "__main__":
    main()
