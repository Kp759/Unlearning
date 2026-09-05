#!/usr/bin/env python
"""Stage-by-stage instrumentation of the fitted Method-5 intervention.

Answers one question: when a paraphrase produces no sensitive-NLL change, at
which stage was the effect lost?

    descriptor -> raw kernel -> conditioned alpha -> quotient transfer
               -> output correction -> numerical realization -> outcome

Nothing is refit.  Every component is reloaded from ``method5_sidecar.pt`` and
used exactly as the seed-1 run fitted it, so this measures that checkpoint
rather than a re-derivation of it.

Three frozen ablations share those identical components, isolating the two
output mechanisms that the Method-4 -> Method-5 comparison changed together:

    logit_only      explicit token penalty on, quotient off
    quotient_only   penalty off, quotient on
    full            both (the fitted Method-5 configuration)

``--oracle`` additionally replays each event with alpha forced to the known
owner coordinate e_i.  That uses privileged fact-ownership information and is a
diagnostic only -- it is never an ordinary Gen number, and the report labels it
so it cannot be mistaken for one.

Official paraphrase prompts are read here for MEASUREMENT ONLY.  No coefficient,
basis, anchor set, radius, or mask is fitted or re-selected.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

import mcf_retain_anchored_context_head_seed1 as m4  # noqa: E402
import mcf_zero_unlearn_official_eval as off  # noqa: E402
from diagnose_method5_gate_coverage import paraphrase_specs  # noqa: E402
from mcf_sampling import sample_official_mcf_records  # noqa: E402
from retain_anchored_context_head import (  # noqa: E402
    AnchoredFeatureMap,
    FrozenRandomProjector,
    wendland_c2_kernel,
)

ABLATIONS = ("logit_only", "quotient_only", "full")


def summarize(values: list[float]) -> dict:
    xs = [v for v in values if v is not None and math.isfinite(v)]
    if not xs:
        return {"n": 0}
    xs_sorted = sorted(xs)
    q = lambda p: xs_sorted[min(len(xs_sorted) - 1, int(p * (len(xs_sorted) - 1)))]  # noqa: E731
    return {
        "n": len(xs),
        "mean": statistics.fmean(xs),
        "median": statistics.median(xs),
        "min": xs_sorted[0],
        "q25": q(0.25),
        "q75": q(0.75),
        "max": xs_sorted[-1],
    }


def matched_negative_specs(data, forget_records, tokenizer, tok_limit: int):
    """Hard negatives that share one binding component with a forget fact.

    Similarity against unrelated retain prompts is too easy a test of fact
    binding.  These three families hold one of subject / relation / answer fixed
    while changing the fact, which is what a fact-identity descriptor must
    separate.
    """
    forget_keys = set()
    subjects, relations, answers = {}, {}, {}
    for record in forget_records:
        rr = record["requested_rewrite"]
        subject = rr["subject"]
        relation = rr.get("relation_id")
        answer = rr["target_true"]["str"]
        forget_keys.add((subject, relation))
        subjects.setdefault(subject, []).append(record)
        relations.setdefault(relation, []).append(record)
        answers.setdefault(answer, []).append(record)

    buckets = {"same_subject_diff_relation": [], "same_relation_diff_subject": [], "same_answer_diff_fact": []}
    for candidate in data:
        rr = candidate.get("requested_rewrite")
        if not isinstance(rr, dict):
            continue
        subject = rr.get("subject")
        relation = rr.get("relation_id")
        answer = rr.get("target_true", {}).get("str")
        if (subject, relation) in forget_keys:
            continue
        if subject in subjects and relation is not None:
            buckets["same_subject_diff_relation"].append(candidate)
        elif relation in relations and subject is not None:
            buckets["same_relation_diff_subject"].append(candidate)
        elif answer in answers:
            buckets["same_answer_diff_fact"].append(candidate)

    out = {}
    for name, records in buckets.items():
        chosen = [off.normalize_record(r) for r in records[:tok_limit]]
        out[name] = m4.build_specs(chosen, tokenizer, max_events_per_record=1) if chosen else []
    return out


@torch.no_grad()
def measure(model, tok, specs, comps, owner_index, args, oracle: bool):
    """Per-event staged measurement.  ``owner_index[k]`` is the fact that owns
    event ``k``, or -1 when the event has no owner (negatives / retain)."""
    rp, basis, fmap, coeff, token_ids, quotient, qmask, enabled = comps
    head = model.get_output_embeddings()
    model_dtype = next(model.parameters()).dtype
    forget_desc = fmap.forget
    device = forget_desc.device
    token_row = {int(t): i for i, t in enumerate(token_ids.tolist())}

    # The saved forget descriptors are fused and unit-norm.  Because each block
    # was separately normalized before concatenation, renormalizing the two
    # slices recovers the original context and channel blocks exactly.
    dim_ctx = int(rp.output_dim)
    anchor_ctx = F.normalize(forget_desc[:, :dim_ctx], dim=-1, eps=1e-8)
    anchor_chan = F.normalize(forget_desc[:, dim_ctx:], dim=-1, eps=1e-8)

    rows = []
    k = 0
    old = tok.padding_side
    tok.padding_side = "right"
    try:
        for start in range(0, len(specs), args.extract_batch_size):
            batch = specs[start:start + args.extract_batch_size]
            enc = tok([s.text for s in batch], padding=True, return_tensors="pt").to(device)
            out = model(**enc, output_hidden_states=True, return_dict=True, use_cache=False)
            for row_i, spec in enumerate(batch):
                for pos, tid in zip(spec.event_positions, spec.event_token_ids):
                    h_final = out.hidden_states[-1][row_i, pos].float()
                    h_causal = out.hidden_states[args.hidden_index][row_i, pos].float()

                    ctx = rp(h_final.unsqueeze(0))
                    chan = F.normalize((h_causal.unsqueeze(0) @ basis), dim=-1, eps=1e-8)
                    fused = F.normalize(torch.cat([ctx, args.causal_weight * chan], -1), dim=-1, eps=1e-8)

                    owner = owner_index[k] if k < len(owner_index) else -1
                    k_all = wendland_c2_kernel(fused, forget_desc, radius=args.radius)[0]
                    alpha_gate = fmap.alpha(fused)[0]
                    if oracle:
                        if owner < 0:
                            k += 1
                            continue
                        alpha = torch.zeros_like(alpha_gate)
                        alpha[owner] = 1.0
                    else:
                        alpha = alpha_gate
                    alpha = alpha * enabled

                    base_logits = head(h_final.to(model_dtype)).float()
                    base_nll = float(-F.log_softmax(base_logits, -1)[tid])

                    record = {
                        "record_index": int(spec.record_index),
                        "token_id": int(tid),
                        "owner_fact": int(owner),
                        "cos_context_own": float(ctx[0] @ anchor_ctx[owner]) if owner >= 0 else None,
                        "cos_channel_own": float(chan[0] @ anchor_chan[owner]) if owner >= 0 else None,
                        "cos_fused_own": float(fused[0] @ forget_desc[owner]) if owner >= 0 else None,
                        "cos_fused_max": float((fused @ forget_desc.T).max()),
                        "kernel_own": float(k_all[owner]) if owner >= 0 else None,
                        "kernel_max": float(k_all.max()),
                        "alpha_own": float(alpha_gate[owner]) if owner >= 0 else None,
                        "alpha_absmax": float(alpha_gate.abs().max()),
                        "quotient_available": bool(qmask[owner] > 0) if owner >= 0 else None,
                        # The bf16 ULP at the head scales with the logit's
                        # magnitude, so record it alongside the realized delta.
                        "base_sensitive_logit": float(base_logits[tid]),
                        "base_sensitive_nll": base_nll,
                    }

                    # --- quotient stage -------------------------------------
                    q_alpha = alpha * qmask
                    h_q = quotient.apply(h_final.unsqueeze(0), q_alpha.unsqueeze(0))[0]
                    delta_h = h_q - h_final
                    record["quotient_delta_h_norm"] = float(delta_h.norm())
                    if owner >= 0:
                        v = quotient.effect_directions[owner]
                        n = quotient.neutral_final_hidden[owner]
                        record["quotient_projected_amplitude"] = float((h_final - n) @ v)

                    # --- explicit penalty stage -----------------------------
                    row = token_row.get(int(tid))
                    intended = float(alpha @ coeff[row]) if row is not None else 0.0
                    record["penalty_intended_fp32"] = intended
                    cast = base_logits[tid].to(model_dtype) + torch.tensor(
                        intended, dtype=model_dtype, device=device
                    )
                    record["penalty_realized_after_cast"] = float(cast.float() - base_logits[tid])
                    record["penalty_lost_to_rounding"] = (
                        record["penalty_intended_fp32"] - record["penalty_realized_after_cast"]
                    )

                    # --- ablations ------------------------------------------
                    for mode in ABLATIONS:
                        logits = (
                            head(h_q.to(model_dtype)).float()
                            if mode in ("quotient_only", "full")
                            else base_logits.clone()
                        )
                        if mode in ("logit_only", "full") and row is not None:
                            logits = logits.to(model_dtype)
                            logits[tid] += torch.tensor(intended, dtype=model_dtype, device=device)
                            logits = logits.float()
                        nll = float(-F.log_softmax(logits, -1)[tid])
                        record[f"{mode}_delta_sensitive_nll"] = nll - base_nll
                    rows.append(record)
                    k += 1
    finally:
        tok.padding_side = old
    return rows


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    keys = [
        "cos_context_own", "cos_channel_own", "cos_fused_own", "cos_fused_max",
        "kernel_own", "kernel_max", "alpha_own", "alpha_absmax",
        "quotient_delta_h_norm", "quotient_projected_amplitude",
        "penalty_intended_fp32", "penalty_realized_after_cast", "penalty_lost_to_rounding",
        *[f"{m}_delta_sensitive_nll" for m in ABLATIONS],
    ]
    out = {"n": len(rows), "stages": {k: summarize([r.get(k) for r in rows]) for k in keys}}
    avail = [r["quotient_available"] for r in rows if r.get("quotient_available") is not None]
    out["quotient_unavailable_events"] = int(sum(1 for x in avail if not x))
    out["alpha_own_frac_gt_0.1"] = (
        statistics.fmean([float(abs(r["alpha_own"]) > 0.1) for r in rows if r.get("alpha_own") is not None])
        if any(r.get("alpha_own") is not None for r in rows) else None
    )
    out["penalty_fully_rounded_away_frac"] = statistics.fmean(
        [float(r["penalty_intended_fp32"] != 0.0 and r["penalty_realized_after_cast"] == 0.0) for r in rows]
    )
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for name in ("model-path", "mcf-path", "sidecar", "output-path"):
        p.add_argument(f"--{name}", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument("--retain-num", type=int, default=1000)
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--descriptor-dim", type=int, default=32)
    p.add_argument("--projection-seed", type=int, default=1729)
    p.add_argument("--causal-weight", type=float, default=1.0)
    p.add_argument("--radius", type=float, default=1.0)
    p.add_argument("--quotient-strength", type=float, default=1.0)
    p.add_argument("--retain-jitter", type=float, default=1e-4)
    p.add_argument("--cardinal-jitter", type=float, default=1e-4)
    p.add_argument("--extract-batch-size", type=int, default=16)
    p.add_argument("--negatives-per-family", type=int, default=200)
    p.add_argument("--oracle", action="store_true", help="also replay with alpha = e_i (privileged)")
    a = p.parse_args()

    device = torch.device(a.device)
    sidecar = torch.load(a.sidecar, map_location="cpu")
    a.hidden_index = int(sidecar["selected_hidden_index"])
    basis = sidecar["channel_basis"].float().to(device)

    tok = AutoTokenizer.from_pretrained(a.model_path, local_files_only=True)
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        a.model_path, torch_dtype=m4.dtype_from_str(a.dtype), local_files_only=True
    ).to(device)
    model.eval()
    model.config.use_cache = False
    for q in model.parameters():
        q.requires_grad_(False)

    from retain_anchored_context_head import FactIndexedCausalQuotient  # noqa: E402

    fmap = AnchoredFeatureMap.fit(
        retain=sidecar["retain_descriptors"].float().to(device),
        forget=sidecar["forget_descriptors"].float().to(device),
        radius=a.radius,
        retain_jitter=a.retain_jitter,
        cardinal_jitter=a.cardinal_jitter,
    )
    quotient = FactIndexedCausalQuotient(
        effect_directions=sidecar["effect_directions"].float().to(device),
        neutral_final_hidden=sidecar["neutral_final_hidden"].float().to(device),
        strength=a.quotient_strength,
    ).to(device)
    rp = FrozenRandomProjector(
        input_dim=model.config.hidden_size, output_dim=a.descriptor_dim,
        seed=a.projection_seed, device=device, dtype=torch.float32,
    )
    comps = (
        rp, basis, fmap,
        sidecar["coefficients"].float().to(device),
        sidecar["selected_token_ids"].to(device),
        quotient,
        sidecar["quotient_fact_mask"].float().to(device),
        torch.ones(fmap.num_facts, device=device),
    )

    data = json.load(open(a.mcf_path))
    fr, rr = sample_official_mcf_records(data, a.forget_num, a.retain_num, a.seed, strict=True)
    fr = [off.normalize_record(x) for x in fr]
    rr = [off.normalize_record(x) for x in rr]

    direct_specs = m4.build_specs(fr, tok, max_events_per_record=None)
    direct_owner = list(range(sum(len(s.event_positions) for s in direct_specs)))
    para_specs, _ = paraphrase_specs(fr, tok, max_events_per_record=None)

    # Own-fact ownership for paraphrase events: fact index is direct-event order.
    direct_key = {}
    k = 0
    for spec in direct_specs:
        for ordinal in range(len(spec.event_positions)):
            direct_key[(spec.record_index, ordinal)] = k
            k += 1
    para_owner = [
        direct_key.get((s.record_index, o), -1)
        for s in para_specs for o in range(len(s.event_positions))
    ]

    negatives = matched_negative_specs(data, fr, tok, a.negatives_per_family)

    report = {
        "schema_version": 1,
        "kind": "method5_stage_instrumentation",
        "measurement_only": True,
        "official_paraphrases_used_for_fit": False,
        "components_refit": False,
        "config": {k: v for k, v in vars(a).items() if k not in ("model_path", "mcf_path")},
        "groups": {},
    }

    def run_group(name, specs, owners, oracle=False):
        if not specs:
            report["groups"][name] = {"n": 0}
            return
        rows = measure(model, tok, specs, comps, owners, a, oracle)
        report["groups"][name] = aggregate(rows)
        report["groups"][name]["per_event"] = rows[:400]

    run_group("direct_forget", direct_specs, direct_owner)
    run_group("forget_paraphrase", para_specs, para_owner)
    for family, specs in negatives.items():
        run_group(f"negative_{family}", specs, [-1] * sum(len(s.event_positions) for s in specs))
    if a.oracle:
        run_group("ORACLE_forget_paraphrase_privileged", para_specs, para_owner, oracle=True)

    Path(a.output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output_path).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: {x: y for x, y in v.items() if x != "per_event"}
                      for k, v in report["groups"].items()}, indent=2))


if __name__ == "__main__":
    main()
