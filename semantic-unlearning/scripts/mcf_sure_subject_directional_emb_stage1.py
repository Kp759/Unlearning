#!/usr/bin/env python3
"""SURE-MCF subject-keyed directional GA/GD on input embeddings only.

Why this exists (the structural finding that motivates it)
---------------------------------------------------------
Every earlier MCF SURE stage edited LM-head rows selected from
``requested_rewrite.target_true`` (see
``mcf_sure_directional_emb_lm_stage1.py`` and
``mcf_sure_fullrow_failure_repair.py``, both of which call
``sensitive_field="target_true"``).  On CounterFact that choice cannot
preserve locality, and the reason is visible in the data rather than in
any hyperparameter:

    subject      : Danielle Darrieux        target_true : "French"
    neighborhood : "The mother tongue of Leon Blum is"      -> French
                   "The native language of Montesquieu is"  -> French

The neighborhood prompts that define Spe have *different subjects* but
the *same* correct answer as the record being forgotten.  A LM-head row
edit changes ``logit_t(h) = W[t] . h`` for every context at once, so
suppressing "French" for Darrieux necessarily suppresses it for Blum and
Montesquieu.  Eff and Spe are coupled through one shared parameter, which
is why ~20 real runs traced a strict trade-off curve (Eff 0 -> Spe 0.16;
Spe 8.6 -> Eff 82) and never reached both corners.  The repository had
already recorded the same conclusion for MQuAKE in
``mquake_sure_contextual_mlp_v80.py``: "globally changing LM-head token
rows remains utility limited even when the update is sparse and prompt
protected".

This script moves the edit to the *subject* side of the input embedding
and leaves the LM head frozen.

Architecture
------------
Trainable parameters: input-embedding rows for the forget records'
subject tokens, and nothing else.  The LM head is untied and frozen
(untying is mandatory -- with tied weights an input-embedding edit would
silently also edit the output row for that token).

Per forget prompt x (locked direct prompts plus the hand-authored
synthetic paraphrase bank, all of which contain the subject):

    u_s        = normalize(W[target_true token])   closed-form sensitive
                                                   readout direction
    dh         = h(x) - h_base(x)
    L_margin   = relu(m - [logp(target_new) - logp(target_true)])
    L_surgical = || dh - (dh . u_s) u_s ||^2
    L_budget   = || E[S] - E_base[S] ||^2

``L_margin`` is gradient ascent on the sensitive token and gradient
descent on the non-sensitive reference in one hinged term, so it stops
pushing once the margin is met instead of running away.

``L_surgical`` is the geometric constraint.  ``u_s`` does not have to be
estimated: the LM-head row for ``target_true`` *is* the direction that
reads "French" out of the final hidden state, so "forget only the
sensitive fact" becomes "let h move along u_s and nowhere else".  It
stops embedding GA from scrambling the subject's whole representation,
which is the catastrophic mode a raw embedding update would otherwise
hit.

Note that ``L_surgical`` is doing representation hygiene, *not* locality.
Locality here is combinatorial rather than geometric: a neighborhood
prompt about Leon Blum does not contain Darrieux's subject tokens, so no
gradient path to it exists and its forward pass stays bitwise identical
to Base.  This is why the earlier soft-weight tuning failures do not
recur -- there, ``pass_guard_weight``/``distribution_kl_weight`` were
being asked to *buy* locality and fought the hinge directly (weight 1.0
collapsed Spe, weight 10.0 broke Eff).  Here locality is free and the
penalty only has to keep the edit clean.

Row selection is frequency-filtered: a subject token that is common in
ordinary text (``the``, ``university``, ``robert``) would reintroduce
collateral damage, so rows are kept only below
``--max-subject-token-frequency`` as counted on a Wikipedia slice that is
disjoint from the official PPL documents.  Each record keeps at least its
rarest subject token so no record is left with nothing to train.

Data firewall: only ``requested_rewrite`` fields of the locked forget
split are read (``subject`` is an allowed key of the locked
training-visible view -- see
``build_mcf_sure_target_aware_direct_split.py``).  Official paraphrase,
neighborhood, retain, and PPL evaluation data are never loaded here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch
import torch.nn.functional as F
from datasets import load_from_disk

import gagd_compare as gagd
import mcf_sure_directional_emb_lm_stage1 as base1
import mcf_synthetic_paraphrase_templates as synth
import sure_canonical_core as core
import sure_context_projection as context

METHOD = "SURE-MCF-subject-keyed-directional-embedding-stage1"
PROTOCOL = "mcf_target_true_sensitive_subject_keyed_embedding_ga_gd_v1"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--training-visible-path", required=True)
    p.add_argument("--split-manifest", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--forget-num", type=int, default=50)

    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--cache-batch-size", type=int, default=8)
    p.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help=(
            "Higher than the 1e-4 used by mcf_sure_directional_emb_lm_stage1.py: "
            "that script spread its update over target_true rows in both the "
            "embedding and the LM head, while this one trains only a handful of "
            "subject embedding rows and has to move the hidden state through 28 "
            "frozen blocks."
        ),
    )
    p.add_argument(
        "--margin-weight",
        type=float,
        default=100.0,
        help="Weight on the GA/GD hinge; dominant term, matching v8's 100x hinge.",
    )
    p.add_argument(
        "--train-margin",
        type=float,
        default=1.0,
        help=(
            "Training hinge target on logp(target_new) - logp(target_true). "
            "Deliberately larger than --stage1-constraint-margin so BF16 "
            "materialization has headroom; it does not redefine the reported "
            "failure rule."
        ),
    )
    p.add_argument(
        "--surgical-weight",
        type=float,
        default=1.0,
        help=(
            "Weight on || dh orthogonal to u_s ||^2. Keeps the hidden-state "
            "change confined to the closed-form sensitive readout direction. "
            "Set 0 to ablate (raw embedding GA/GD with no direction constraint)."
        ),
    )
    p.add_argument("--delta-l2", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--stage1-constraint-margin", type=float, default=0.05)
    p.add_argument("--synthetic-paraphrases-per-record", type=int, default=3)

    p.add_argument(
        "--max-subject-token-frequency",
        type=int,
        default=100,
        help=(
            "Keep a subject token row only if it occurs at most this many times "
            "in the frequency corpus. Bounds Spe/PPL collateral: a subject token "
            "that is also ordinary vocabulary ('the', 'university', 'robert') "
            "would reintroduce exactly the shared-parameter coupling this "
            "architecture exists to avoid. Each record still keeps its rarest "
            "subject token if every one of them exceeds the threshold."
        ),
    )
    p.add_argument("--wikidata-dir", default="data/wikidata")
    p.add_argument(
        "--frequency-docs",
        type=int,
        default=5000,
        help="Documents used to count subject-token frequency.",
    )
    p.add_argument(
        "--frequency-doc-start",
        type=int,
        default=20,
        help=(
            "Must stay >= 20: mcf_zero_unlearn_official_eval.load_official_ppl_text "
            "is hardcoded to documents [:20], and counting frequencies on the exact "
            "text official PPL is later scored against would contaminate the result."
        ),
    )

    p.add_argument(
        "--candidate-scales",
        default="1,.875,.75,.625,.5,.375,.25,.1875,.125,.09375,.0625,.046875,.03125,.015625,.0078125,0",
    )
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--device-map", default="single")
    a = p.parse_args(argv)
    if int(a.frequency_docs) > 0 and int(a.frequency_doc_start) < 20:
        p.error(
            "--frequency-doc-start must be >= 20 so the frequency corpus stays "
            "disjoint from official PPL's hardcoded [:20] documents"
        )
    if float(a.train_margin) <= 0:
        p.error("--train-margin must be positive")
    return a


def subject_token_ids(tok: Any, subject: str, *, llama_like: bool) -> List[int]:
    """Token ids for a subject in both mid-sentence and sentence-initial form.

    The locked direct prompt puts the subject after a space ("The mother
    tongue of {} is"), but synthetic and official paraphrases also place it
    sentence-initially ("Danielle Darrieux, a native"). Those tokenize
    differently, and Gen depends on the sentence-initial form firing too, so
    both variants are unioned.
    """
    found: set[int] = set()
    text = str(subject).strip()
    if not text:
        return []
    for variant in (" " + text, text):
        ids = core.flat_ids(tok, variant)
        ids = [int(x) for x in ids]
        if llama_like and ids:
            ids = ids[1:]
        found.update(ids)
    return sorted(found - set(int(x) for x in gagd.special_token_ids(tok)))


def load_frequency_documents(
    wikidata_dir: str, doc_start: int, num_docs: int
) -> List[str]:
    if num_docs <= 0:
        return []
    path = Path(wikidata_dir)
    if not path.exists():
        return []
    raw_ds = load_from_disk(str(path))
    texts = raw_ds["train"]["text"][doc_start : doc_start + num_docs]
    return [t for t in texts if t and t.strip()]


def token_frequency_counts(
    tok: Any, documents: Sequence[str], vocab_size: int
) -> torch.Tensor:
    counts = torch.zeros(vocab_size, dtype=torch.long)
    for doc in documents:
        ids = tok(str(doc), add_special_tokens=False)["input_ids"]
        if not ids:
            continue
        flat = torch.tensor([int(x) for x in ids], dtype=torch.long)
        flat = flat[flat < vocab_size]
        if flat.numel():
            counts += torch.bincount(flat, minlength=vocab_size)
    return counts


def select_subject_rows(
    records: Sequence[Mapping[str, Any]],
    tok: Any,
    *,
    llama_like: bool,
    counts: torch.Tensor,
    max_frequency: int,
) -> Tuple[List[int], List[Dict[str, Any]]]:
    """Frequency-filtered subject embedding rows, with a per-record report."""
    selected: set[int] = set()
    reports: List[Dict[str, Any]] = []
    have_counts = bool(counts.numel())
    for position, record in enumerate(records):
        rewrite = record.get("requested_rewrite")
        if not isinstance(rewrite, Mapping):
            raise ValueError(f"record {position} lacks requested_rewrite")
        subject = str(rewrite.get("subject", ""))
        ids = subject_token_ids(tok, subject, llama_like=llama_like)
        if not ids:
            raise RuntimeError(
                f"record {position} subject {subject!r} produced no embedding rows"
            )
        if have_counts:
            freqs = {i: int(counts[i].item()) for i in ids}
            kept = [i for i in ids if freqs[i] <= int(max_frequency)]
            fallback = False
            if not kept:
                # Never leave a record with nothing trainable; take its rarest
                # token and record that this record is expected to carry more
                # collateral than the clean-anchor majority.
                kept = [min(ids, key=lambda i: freqs[i])]
                fallback = True
        else:
            freqs = {i: -1 for i in ids}
            kept = list(ids)
            fallback = False
        selected.update(kept)
        reports.append(
            {
                "record_position": int(position),
                "case_id": int(record.get("case_id", position)),
                "subject": subject,
                "subject_token_ids": ids,
                "kept_token_ids": sorted(int(x) for x in kept),
                "kept_token_frequencies": [freqs[i] for i in sorted(kept)],
                "dropped_token_ids": sorted(int(x) for x in set(ids) - set(kept)),
                "rarest_token_fallback": bool(fallback),
            }
        )
    return sorted(int(x) for x in selected), reports


def forward_last_logits_and_hidden(
    model: torch.nn.Module,
    tok: Any,
    cases: Sequence[core.SensitivePredictionCase],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Differentiable twin of core.forward_last_hidden (which is no_grad)."""
    encoded = tok([c.prompt for c in cases], padding=True, return_tensors="pt").to(device)
    output = model(**encoded, output_hidden_states=True, use_cache=False)
    positions = encoded["attention_mask"].sum(dim=1) - 1
    rows = torch.arange(len(cases), device=device)
    logits = output.logits[rows, positions, :]
    hidden = output.hidden_states[-1][rows, positions, :]
    return logits, hidden


def sensitive_readout_directions(
    output_layer: torch.nn.Module, tids: torch.Tensor
) -> torch.Tensor:
    """u_s per case: the frozen LM-head row that reads the sensitive answer."""
    rows = output_layer.weight.index_select(0, tids.to(output_layer.weight.device))
    rows = rows.float().detach()
    return F.normalize(rows, dim=-1, eps=1e-6)


def surgical_off_direction_penalty(
    hidden: torch.Tensor, base_hidden: torch.Tensor, directions: torch.Tensor
) -> torch.Tensor:
    """|| dh - (dh . u_s) u_s ||^2 -- the component of the hidden-state change
    that is NOT along the sensitive readout direction."""
    dh = hidden.float() - base_hidden.to(device=hidden.device, dtype=torch.float32)
    u = directions.to(device=dh.device, dtype=torch.float32)
    parallel = (dh * u).sum(dim=-1, keepdim=True) * u
    return (dh - parallel).square().mean()


@torch.no_grad()
def sensitive_readout_scores(
    model: torch.nn.Module,
    tok: Any,
    output_layer: torch.nn.Module,
    cases: Sequence[core.SensitivePredictionCase],
    device: torch.device,
    batch_size: int,
    llama_like: bool,
) -> float:
    """Mean h . u_s -- mechanistic forgetting evidence, independent of NLL."""
    if not cases:
        return 0.0
    totals: List[torch.Tensor] = []
    for start in range(0, len(cases), batch_size):
        batch = list(cases[start : start + batch_size])
        hidden = core.forward_last_hidden(model, tok, batch, device, len(batch))
        tids = core.official_target_ids(
            tok, batch, llama_like=llama_like, device=device
        )
        u = sensitive_readout_directions(output_layer, tids)
        totals.append((hidden.float() * u.to(hidden.device)).sum(dim=-1).cpu())
    return float(torch.cat(totals).mean())


def main(argv: Sequence[str] | None = None) -> None:
    a = parse_args(argv)
    gagd.set_seed(a.seed)
    if a.device_map == "single":
        gagd.require_cuda_if_needed(a.device_map)

    records, manifest = base1.validate_locked(
        Path(a.training_visible_path).resolve(),
        Path(a.split_manifest).resolve(),
        int(a.seed),
        int(a.forget_num),
    )

    synthetic_records = synth.build_synthetic_records(
        records, count=int(a.synthetic_paraphrases_per_record)
    )
    all_records = list(records) + synthetic_records
    synthetic_coverage = synth.coverage_report(records)

    ns = argparse.Namespace(
        model_path=a.model_path,
        dtype=a.dtype,
        device_map=a.device_map,
        gradient_checkpointing=False,
    )
    model, tok = gagd.load_model_and_tokenizer(ns, for_training=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Untying is mandatory, not hygiene: with tied weights, editing an input
    # embedding row also edits that token's LM-head row, which would change
    # its logit in every context and destroy exactly the locality this
    # architecture is built to keep.
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
    frequency_documents = load_frequency_documents(
        a.wikidata_dir, int(a.frequency_doc_start), int(a.frequency_docs)
    )
    counts = token_frequency_counts(tok, frequency_documents, vocab_size)
    if int(a.frequency_docs) > 0 and not frequency_documents:
        print(
            f"WARNING: no frequency documents loaded from {a.wikidata_dir!r}; "
            "every subject token row will be trained unfiltered."
        )

    selected_ids, subject_reports = select_subject_rows(
        records,
        tok,
        llama_like=llama_like,
        counts=counts,
        max_frequency=int(a.max_subject_token_frequency),
    )
    if not selected_ids:
        raise RuntimeError("no subject embedding rows selected")
    fallback_records = [r for r in subject_reports if r["rarest_token_fallback"]]
    print(
        f"Selected {len(selected_ids)} subject embedding rows across "
        f"{len(records)} records; {len(fallback_records)} record(s) had no token "
        f"under the frequency threshold and fell back to their rarest token."
    )

    sensitive_cases = context.expand_answer_field_cases(
        all_records, tok, field="target_true", llama_like=llama_like
    )
    reference_cases = context.expand_answer_field_cases(
        all_records, tok, field="target_new", llama_like=llama_like
    )
    reference_index = base1._reference_index_by_sensitive_case(
        sensitive_cases, reference_cases
    )

    base_hidden = core.forward_last_hidden(
        model, tok, sensitive_cases, device, int(a.cache_batch_size)
    )
    base_readout = sensitive_readout_scores(
        model, tok, output_layer, sensitive_cases, device, int(a.cache_batch_size), llama_like
    )

    hidden_size = int(input_layer.weight.shape[1])
    delta_module = core.SelectedRowDelta(
        len(selected_ids),
        hidden_size,
        direction_basis=None,
        device=input_layer.weight.device,
    )
    parameters = list(delta_module.parameters())
    opt = torch.optim.AdamW(parameters, lr=float(a.lr), weight_decay=0.0)
    sampler = core.IndexSampler(len(sensitive_cases), int(a.batch_size), int(a.seed))

    out_dir = gagd.resolve_output_path(a.output_dir)
    ckpt = out_dir / "checkpoint"
    out_dir.mkdir(parents=True, exist_ok=True)

    emb_hook = base1.register_input_embedding_delta_hook(
        input_layer, selected_ids, delta_module.effective_delta
    )
    try:
        model.eval()
        with (out_dir / "train_log.jsonl").open("w", encoding="utf-8") as log_f:
            for step in range(1, int(a.steps) + 1):
                idx = sampler.next()
                batch = [sensitive_cases[i] for i in idx]
                opt.zero_grad(set_to_none=True)

                logits, hidden = forward_last_logits_and_hidden(
                    model, tok, batch, device
                )
                logp = F.log_softmax(logits.float(), dim=-1)
                rows = torch.arange(logp.shape[0], device=logp.device)

                tids_true = core.official_target_ids(
                    tok, batch, llama_like=llama_like, device=device
                )
                reference_batch = [reference_cases[reference_index[i]] for i in idx]
                tids_new = core.official_target_ids(
                    tok, reference_batch, llama_like=llama_like, device=device
                )
                logp_true = logp[rows, tids_true]
                logp_new = logp[rows, tids_new]

                # GA on the sensitive token and GD on the non-sensitive
                # reference in one hinged term. Exact for token_index 0, where
                # the sensitive and reference prompts are identical; for later
                # answer tokens it reads target_new's token at the target_true
                # prefix, which is still a valid contrastive signal. Reported
                # failures and scale selection always use the exact
                # full-sequence official margin below, never this training
                # approximation.
                margin = logp_new - logp_true
                margin_hinge = F.relu(float(a.train_margin) - margin).mean()

                directions = sensitive_readout_directions(output_layer, tids_true)
                surgical = surgical_off_direction_penalty(
                    hidden, base_hidden[idx], directions
                )
                delta_now = delta_module.effective_delta()
                l2 = delta_now.square().mean()

                total = (
                    float(a.margin_weight) * margin_hinge
                    + float(a.surgical_weight) * surgical
                    + float(a.delta_l2) * l2
                )
                if not torch.isfinite(total):
                    raise FloatingPointError(
                        f"Non-finite subject-keyed Stage-1 loss at step {step}"
                    )
                total.backward()
                grad_norm = None
                if float(a.grad_clip) > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        parameters, float(a.grad_clip)
                    )
                    if not torch.isfinite(grad_norm):
                        raise FloatingPointError(
                            f"Non-finite gradient norm at step {step}"
                        )
                opt.step()

                if step == 1 or step % 25 == 0 or step == int(a.steps):
                    row = {
                        "step": int(step),
                        "total_loss": float(total.detach().cpu()),
                        "margin_hinge": float(margin_hinge.detach().cpu()),
                        "batch_mean_margin": float(margin.mean().detach().cpu()),
                        "batch_min_margin": float(margin.min().detach().cpu()),
                        "surgical_off_direction": float(surgical.detach().cpu()),
                        "delta_l2": float(l2.detach().cpu()),
                        "embedding_delta_norm": float(
                            delta_now.detach().norm().cpu()
                        ),
                        "lm_head_delta_norm": 0.0,
                        "benchmark_retain_seen": 0,
                        "heldout_probes_seen": 0,
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

    direct_count = len(records)
    scales = core.parse_scales(a.candidate_scales)
    scale_reports: List[Dict[str, Any]] = []
    for scale in scales:
        handle = base1.register_input_embedding_delta_hook(
            input_layer,
            selected_ids,
            lambda scale=scale: trained_delta * float(scale),
        )
        try:
            margins = base1._direct_margins(
                model, tok, all_records, device, llama_like, int(a.cache_batch_size)
            )
        finally:
            handle.remove()
        direct_margins = margins[:direct_count]
        synthetic_margins = margins[direct_count:]
        scale_reports.append(
            {
                "scale": float(scale),
                "direct_failures": int(
                    (margins < float(a.stage1_constraint_margin)).sum().item()
                ),
                "minimum_margin": float(margins.min().detach().cpu()),
                "direct_only_failures": int(
                    (direct_margins < float(a.stage1_constraint_margin)).sum().item()
                ),
                "direct_only_minimum_margin": float(
                    direct_margins.min().detach().cpu()
                ),
                "synthetic_failures": (
                    int(
                        (synthetic_margins < float(a.stage1_constraint_margin))
                        .sum()
                        .item()
                    )
                    if synthetic_margins.numel()
                    else 0
                ),
                "synthetic_minimum_margin": (
                    float(synthetic_margins.min().detach().cpu())
                    if synthetic_margins.numel()
                    else None
                ),
                "embedding_delta_norm": float(trained_delta.norm().cpu() * scale),
            }
        )

    selected_scale = base1.select_stage1_scale(scale_reports)
    final_delta = trained_delta * float(selected_scale)
    base1.materialize_input_delta(input_layer, selected_ids, final_delta)

    final_all_margins = base1._direct_margins(
        model, tok, all_records, device, llama_like, int(a.cache_batch_size)
    )
    final_margins = final_all_margins[:direct_count]
    final_synthetic_margins = final_all_margins[direct_count:]
    final_failures = [
        int(i)
        for i, value in enumerate(final_margins.detach().cpu().tolist())
        if float(value) < float(a.stage1_constraint_margin)
    ]
    final_synthetic_failures = [
        int(i)
        for i, value in enumerate(final_synthetic_margins.detach().cpu().tolist())
        if float(value) < float(a.stage1_constraint_margin)
    ]
    final_readout = sensitive_readout_scores(
        model, tok, output_layer, sensitive_cases, device, int(a.cache_batch_size), llama_like
    )

    ckpt.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt)
    tok.save_pretrained(ckpt)

    config: Dict[str, Any] = {
        "schema_version": 1,
        "method": METHOD,
        "protocol": PROTOCOL,
        "dataset": "mcf",
        "source_protocol": manifest.get("protocol"),
        "seed": int(a.seed),
        "forget_num": int(a.forget_num),
        "target_contract": {
            "sensitive_answer": "requested_rewrite.target_true",
            "non_sensitive_reference": "requested_rewrite.target_new",
            "field_swapping": False,
        },
        "lm_head_untied_before_training": True,
        "lm_head_trainable_parameters": 0,
        "lm_head_edited": False,
        "transformer_trainable_parameters": 0,
        "editable_embedding_rows": "subject_token_rows_only_frequency_filtered",
        "locality_mechanism": (
            "combinatorial, not geometric: a neighborhood prompt whose subject "
            "differs contains none of the edited rows, so no gradient path to it "
            "exists and its forward pass is bitwise identical to Base. Earlier "
            "MCF stages instead edited target_true LM-head rows, which every "
            "CounterFact neighborhood prompt must also produce -- coupling Eff "
            "and Spe through one shared parameter."
        ),
        "sensitive_direction_definition": (
            "u_s = normalize(LM_head[target_true token]), the closed-form "
            "readout direction for the sensitive answer; never estimated"
        ),
        "surgical_penalty": (
            "|| (h - h_base) - ((h - h_base).u_s) u_s ||^2 -- confines the "
            "hidden-state change to u_s; representation hygiene, not locality"
        ),
        "selected_token_ids": selected_ids,
        "selected_row_count": len(selected_ids),
        "subject_row_selection": subject_reports,
        "records_using_rarest_token_fallback": len(fallback_records),
        "max_subject_token_frequency": int(a.max_subject_token_frequency),
        "frequency_documents_used": len(frequency_documents),
        "frequency_doc_range": [
            int(a.frequency_doc_start),
            int(a.frequency_doc_start) + int(a.frequency_docs),
        ],
        "frequency_corpus_disjoint_from_official_ppl_docs": (
            "official PPL is hardcoded to documents [:20]; this run counted "
            f"frequencies from [{int(a.frequency_doc_start)}:"
            f"{int(a.frequency_doc_start) + int(a.frequency_docs)}]"
        ),
        "synthetic_paraphrases_per_record": int(a.synthetic_paraphrases_per_record),
        "synthetic_record_count": len(synthetic_records),
        "synthetic_paraphrase_coverage": synthetic_coverage,
        "embedding_trainable_parameters": int(
            sum(p.numel() for p in delta_module.parameters())
        ),
        "margin_loss": (
            "relu(train_margin - [logp(target_new) - logp(target_true)]); GA on "
            "sensitive and GD on non-sensitive in one hinged term"
        ),
        "margin_weight": float(a.margin_weight),
        "train_margin": float(a.train_margin),
        "surgical_weight": float(a.surgical_weight),
        "delta_l2": float(a.delta_l2),
        "steps": int(a.steps),
        "batch_size": int(a.batch_size),
        "lr": float(a.lr),
        "selected_scale": float(selected_scale),
        "scale_reports": scale_reports,
        "stage1_constraint_margin": float(a.stage1_constraint_margin),
        "stage1_direct_failures": len(final_failures),
        "stage1_failing_positions": final_failures,
        "stage1_minimum_margin": float(final_margins.min().detach().cpu()),
        "stage1_synthetic_failures": len(final_synthetic_failures),
        "stage1_synthetic_failing_positions": final_synthetic_failures,
        "stage1_synthetic_minimum_margin": (
            float(final_synthetic_margins.min().detach().cpu())
            if final_synthetic_margins.numel()
            else None
        ),
        "stage1_combined_failures": len(final_failures) + len(final_synthetic_failures),
        "stage1_combined_minimum_margin": float(
            final_all_margins.min().detach().cpu()
        ),
        "base_mean_sensitive_readout": float(base_readout),
        "final_mean_sensitive_readout": float(final_readout),
        "sensitive_readout_drop": float(base_readout - final_readout),
        "final_embedding_delta_norm": float(final_delta.norm().cpu()),
        "final_lm_head_delta_norm": 0.0,
        "lora_used": False,
        "official_paraphrases_seen": 0,
        "official_neighborhood_seen": 0,
        "benchmark_retain_seen": 0,
        "ppl_eval_text_seen": 0,
    }
    core.write_json(out_dir / "stage1_config.json", config)
    print(json.dumps(config, indent=2))
    print(f"Stage-1 checkpoint: {ckpt}")
    print(
        f"Stage-1 direct failures: {len(final_failures)}/{len(records)}; "
        f"min margin={config['stage1_minimum_margin']:.6f}"
    )
    print(
        f"Stage-1 synthetic-paraphrase failures: {len(final_synthetic_failures)}/"
        f"{len(synthetic_records)}; "
        f"min margin={config['stage1_synthetic_minimum_margin']}"
    )
    print(
        f"Mean sensitive readout h.u_s: {base_readout:.6f} -> {final_readout:.6f} "
        f"(drop {base_readout - final_readout:.6f})"
    )


if __name__ == "__main__":
    main()
