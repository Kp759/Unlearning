#!/usr/bin/env python
"""Why the Method-5 gate does not fire on paraphrases.

MEASUREMENT ONLY.  This script reads official paraphrase prompts to measure
gate coverage.  It fits nothing: no ``C``, no ``U_r``, no anchor set, and no
radius is selected here for downstream use.  The Method-4/5 training contract
is therefore unchanged -- any radius this script surfaces must be re-derived
from training-visible data before it is used in a fitted run.

Background.  ``wendland_c2_kernel`` raises the compact-support exponent with the
descriptor dimension (``ell = d_dim//2 + 2``, ``power = ell + 1``).  At
``descriptor_dim_total = 40`` that is ``(1-r)^23 (23r + 1)``, so the *nominal*
radius 1.0 has a far smaller *effective* support than it appears to.  All
descriptors are L2-normalized onto the unit sphere by
``CausalDescriptorProjector.forward``, so distance and cosine are locked
together: ``d = sqrt(2 - 2 cos)``.  This script measures where own-paraphrase
and retain pairs actually land on that curve.
"""

from __future__ import annotations

import argparse
import json
import math
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
from mcf_sampling import sample_official_mcf_records  # noqa: E402
from retain_anchored_context_head import (  # noqa: E402
    AnchoredFeatureMap,
    FrozenRandomProjector,
    wendland_c2_kernel,
)

QUANTILES = (0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0)


def paraphrase_specs(records, tokenizer, *, max_events_per_record=None):
    """SequenceSpecs over official paraphrase prompts, same target_true events.

    Mirrors ``m4._sequence_spec`` but substitutes each paraphrase prompt for the
    ``requested_rewrite`` prefix.  ``record_index`` stays the index of the parent
    forget record so paraphrase events can be paired back to their own fact.
    """
    specs, ordinals = [], []
    for record_index, record in enumerate(records):
        rr = record["requested_rewrite"]
        target_true = rr["target_true"]["str"]
        target_ids = m4._answer_token_ids(tokenizer, target_true)
        if not target_ids:
            raise ValueError(f"record {record_index} has empty target_true tokenization")
        for prompt in record.get("paraphrase_prompts", []):
            text = f"{prompt} {target_true}"
            full_ids = list(tokenizer(text)["input_ids"])
            if len(full_ids) < len(target_ids) or full_ids[-len(target_ids):] != target_ids:
                # Paraphrase prefixes can retokenize the answer boundary; skip
                # rather than silently misalign an event position.
                continue
            answer_start = len(full_ids) - len(target_ids)
            n = len(target_ids) if max_events_per_record is None else min(len(target_ids), int(max_events_per_record))
            positions = [answer_start + j - 1 for j in range(n)]
            if min(positions) < 0:
                continue
            specs.append(
                m4.SequenceSpec(
                    text=text,
                    event_positions=positions,
                    event_token_ids=target_ids[:n],
                    record_index=int(record_index),
                )
            )
            ordinals.append(list(range(n)))
    return specs, ordinals


@torch.no_grad()
def descriptor_blocks(model, tok, projector, basis, specs, hidden_idx, batch_size, device, causal_weight):
    """Return the two descriptor blocks separately plus the fused descriptor.

    ``context`` and ``channel`` are each unit-norm, exactly as
    ``CausalDescriptorProjector.forward`` builds them, so the fused cosine is a
    fixed convex combination of the two block cosines:

        cos_fused = (cos_context + w^2 cos_channel) / (1 + w^2)
    """
    ctx, chan, rids, ords = [], [], [], []
    old = tok.padding_side
    tok.padding_side = "right"
    try:
        for start in range(0, len(specs), batch_size):
            batch = specs[start:start + batch_size]
            enc = tok([s.text for s in batch], padding=True, return_tensors="pt").to(device)
            out = model(**enc, output_hidden_states=True, return_dict=True, use_cache=False)
            hf, hc = [], []
            for row, spec in enumerate(batch):
                for ordinal, pos in enumerate(spec.event_positions):
                    hf.append(out.hidden_states[-1][row, pos])
                    hc.append(out.hidden_states[hidden_idx][row, pos])
                    rids.append(int(spec.record_index))
                    ords.append(int(ordinal))
            if hf:
                ctx.append(projector(torch.stack(hf)).float())
                chan.append(F.normalize(torch.stack(hc).float() @ basis, dim=-1, eps=1e-8))
    finally:
        tok.padding_side = old
    context = torch.cat(ctx)
    channel = torch.cat(chan)
    fused = F.normalize(torch.cat([context, causal_weight * channel], dim=-1), dim=-1, eps=1e-8)
    return context, channel, fused, rids, ords


def quantile_report(values: torch.Tensor) -> dict:
    if values.numel() == 0:
        return {"n": 0}
    v = values.float().flatten()
    qs = torch.tensor(QUANTILES, device=v.device, dtype=v.dtype)
    out = {"n": int(v.numel()), "mean": float(v.mean())}
    for q, x in zip(QUANTILES, torch.quantile(v, qs).tolist()):
        out[f"q{int(q * 100):02d}"] = float(x)
    return out


def kernel_profile(descriptor_dim: int, radius: float) -> dict:
    """Cosine -> kernel value table for the *actual* fitted kernel."""
    ell = descriptor_dim // 2 + 2
    power = ell + 1
    table = {}
    for cos in (1.0, 0.999, 0.99, 0.98, 0.97, 0.95, 0.90, 0.80, 0.70, 0.50, 0.0):
        d = math.sqrt(max(0.0, 2.0 - 2.0 * cos)) / radius
        table[f"cos={cos}"] = (max(0.0, 1.0 - d) ** power) * (power * d + 1.0)
    # Cosine at which the kernel first exceeds a usable magnitude.
    thresholds = {}
    for target in (0.5, 0.1, 0.01, 1e-3):
        lo, hi = 0.0, 1.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            d = math.sqrt(max(0.0, 2.0 - 2.0 * mid)) / radius
            k = (max(0.0, 1.0 - d) ** power) * (power * d + 1.0)
            if k < target:
                lo = mid
            else:
                hi = mid
        thresholds[f"min_cos_for_k>={target}"] = 0.5 * (lo + hi)
    return {"wendland_power": power, "radius": radius, "cos_to_k": table, **thresholds}


def own_fact_pairs(direct_rids, direct_ords, para_rids, para_ords):
    """Map each paraphrase event to its own-fact direct event index."""
    index = {(r, o): i for i, (r, o) in enumerate(zip(direct_rids, direct_ords))}
    pairs = []
    for j, (r, o) in enumerate(zip(para_rids, para_ords)):
        i = index.get((r, o))
        if i is not None:
            pairs.append((j, i))
    return pairs


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
    p.add_argument("--retain-events-per-record", type=int, default=1)
    p.add_argument("--extract-batch-size", type=int, default=16)
    p.add_argument("--retain-jitter", type=float, default=1e-4)
    p.add_argument("--cardinal-jitter", type=float, default=1e-4)
    p.add_argument(
        "--radius-sweep",
        default="1.0,1.5,2.0,2.5,3.0,4.0,6.0",
        help="comma-separated nominal radii to refit the anchored map at",
    )
    a = p.parse_args()

    device = torch.device(a.device)
    sidecar = torch.load(a.sidecar, map_location="cpu")
    basis = sidecar["channel_basis"].float().to(device)
    hidden_idx = int(sidecar["selected_hidden_index"])

    tok = AutoTokenizer.from_pretrained(a.model_path, local_files_only=True)
    tok.pad_token = tok.pad_token or tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        a.model_path, torch_dtype=m4.dtype_from_str(a.dtype), local_files_only=True
    ).to(device)
    model.eval()
    model.config.use_cache = False
    for q in model.parameters():
        q.requires_grad_(False)

    data = json.load(open(a.mcf_path))
    fr, rr = sample_official_mcf_records(data, a.forget_num, a.retain_num, a.seed, strict=True)
    fr = [off.normalize_record(x) for x in fr]
    rr = [off.normalize_record(x) for x in rr]

    forget_specs = m4.build_specs(fr, tok, max_events_per_record=None)
    retain_specs = m4.build_specs(rr, tok, max_events_per_record=a.retain_events_per_record)
    para_specs, _ = paraphrase_specs(fr, tok, max_events_per_record=None)
    if not para_specs:
        raise RuntimeError("no paraphrase events survived tokenization alignment")

    # Every retain record is itself an anchor, so alpha(r_j) ~ 0 holds there by
    # construction and says nothing about leakage.  The exposed surface when the
    # radius widens is the *non-anchor* prompt variants of the same-answer retain
    # records -- which is what the evaluator's hard-overlap split actually scores.
    sensitive_token_ids = {int(t) for s in forget_specs for t in s.event_token_ids}
    overlap_records = m4._hard_overlap_records(rr, tok, sensitive_token_ids)
    overlap_specs, _ = paraphrase_specs(overlap_records, tok, max_events_per_record=None)

    rp = FrozenRandomProjector(
        input_dim=model.config.hidden_size,
        output_dim=a.descriptor_dim,
        seed=a.projection_seed,
        device=device,
        dtype=torch.float32,
    )
    def blocks_for(specs):
        return descriptor_blocks(
            model, tok, rp, basis, specs, hidden_idx,
            a.extract_batch_size, device, a.causal_weight,
        )

    d_ctx, d_chan, d_fused, d_rids, d_ords = blocks_for(forget_specs)
    p_ctx, p_chan, p_fused, p_rids, p_ords = blocks_for(para_specs)
    r_ctx, r_chan, r_fused, _, _ = blocks_for(retain_specs)
    o_fused = blocks_for(overlap_specs)[2] if overlap_specs else None

    # Sanity: the fitted forget descriptors must reproduce the saved sidecar.
    saved_forget = sidecar["forget_descriptors"].float().to(device)
    drift = float((d_fused - saved_forget).abs().max()) if saved_forget.shape == d_fused.shape else float("nan")

    pairs = own_fact_pairs(d_rids, d_ords, p_rids, p_ords)
    if not pairs:
        raise RuntimeError("no paraphrase event paired to a direct event")
    pj = torch.tensor([j for j, _ in pairs], device=device)
    pi = torch.tensor([i for _, i in pairs], device=device)

    own_cos_ctx = (p_ctx[pj] * d_ctx[pi]).sum(-1)
    own_cos_chan = (p_chan[pj] * d_chan[pi]).sum(-1)
    own_cos_fused = (p_fused[pj] * d_fused[pi]).sum(-1)

    # Background: how close does an unrelated retain event get to its nearest
    # forget anchor?  This is the ceiling any usable radius must stay under.
    retain_best_fused = (r_fused @ d_fused.T).max(dim=1).values
    retain_best_ctx = (r_ctx @ d_ctx.T).max(dim=1).values
    retain_best_chan = (r_chan @ d_chan.T).max(dim=1).values
    overlap_best_fused = (
        (o_fused @ d_fused.T).max(dim=1).values if o_fused is not None else torch.empty(0, device=device)
    )

    report = {
        "schema_version": 1,
        "kind": "method5_gate_coverage_diagnostic",
        "measurement_only": True,
        "official_paraphrases_used_for_fit": False,
        "counts": {
            "forget_events": len(d_rids),
            "paraphrase_events": len(p_rids),
            "paired_paraphrase_events": len(pairs),
            "retain_events": r_fused.shape[0],
            "hard_overlap_retain_records": len(overlap_records),
            "hard_overlap_nonanchor_events": 0 if o_fused is None else int(o_fused.shape[0]),
        },
        "descriptor": {
            "dim_context": int(d_ctx.shape[1]),
            "dim_channel": int(d_chan.shape[1]),
            "dim_total": int(d_fused.shape[1]),
            "causal_weight": a.causal_weight,
            "selected_hidden_index": hidden_idx,
            "forget_descriptor_drift_vs_sidecar": drift,
        },
        "kernel_at_nominal_radius_1": kernel_profile(int(d_fused.shape[1]), 1.0),
        "cosine": {
            "own_paraphrase_context": quantile_report(own_cos_ctx),
            "own_paraphrase_channel": quantile_report(own_cos_chan),
            "own_paraphrase_fused": quantile_report(own_cos_fused),
            "retain_nearest_forget_context": quantile_report(retain_best_ctx),
            "retain_nearest_forget_channel": quantile_report(retain_best_chan),
            "retain_nearest_forget_fused": quantile_report(retain_best_fused),
            "hard_overlap_nonanchor_nearest_forget_fused": quantile_report(overlap_best_fused),
        },
        # The decisive number.  A usable radius exists only if own-paraphrase
        # descriptors sit clearly above the non-anchor hard-overlap ceiling; if
        # these distributions interleave, no radius separates them and the
        # descriptor itself has to change.
        "separation": {
            "median_own_paraphrase_fused": float(own_cos_fused.median()),
            "q95_retain_nearest_fused": float(torch.quantile(retain_best_fused.float(), 0.95)),
            "margin_vs_retain": float(
                own_cos_fused.median() - torch.quantile(retain_best_fused.float(), 0.95)
            ),
            "q95_hard_overlap_nearest_fused": (
                None if overlap_best_fused.numel() == 0
                else float(torch.quantile(overlap_best_fused.float(), 0.95))
            ),
            "margin_vs_hard_overlap": (
                None if overlap_best_fused.numel() == 0
                else float(own_cos_fused.median() - torch.quantile(overlap_best_fused.float(), 0.95))
            ),
        },
        "radius_sweep": [],
    }

    for radius in [float(x) for x in a.radius_sweep.split(",")]:
        entry = {"radius": radius, "kernel": kernel_profile(int(d_fused.shape[1]), radius)}
        try:
            k_rr = wendland_c2_kernel(r_fused, r_fused, radius=radius)
            eye = torch.eye(k_rr.shape[0], device=k_rr.device, dtype=k_rr.dtype)
            entry["retain_gram_cond"] = float(torch.linalg.cond(k_rr + a.retain_jitter * eye))
            fmap = AnchoredFeatureMap.fit(
                retain=r_fused,
                forget=d_fused,
                radius=radius,
                retain_jitter=a.retain_jitter,
                cardinal_jitter=a.cardinal_jitter,
            )
            alpha_para = fmap.alpha(p_fused)
            # Facts are indexed by forget *event* order, which is exactly the
            # row order of ``d_fused``, so the own-fact column is ``pi``.
            own_alpha = alpha_para[pj, pi]
            alpha_retain = fmap.alpha(r_fused)
            entry.update(
                {
                    "cardinal_max_abs_error": float(fmap.cardinal_residual().abs().max()),
                    "retain_max_abs_alpha": float(alpha_retain.abs().max()),
                    "paraphrase_own_alpha": quantile_report(own_alpha),
                    "paraphrase_own_alpha_frac_gt_0.01": float((own_alpha.abs() > 0.01).float().mean()),
                    "paraphrase_own_alpha_frac_gt_0.1": float((own_alpha.abs() > 0.1).float().mean()),
                    "paraphrase_own_alpha_frac_gt_0.5": float((own_alpha.abs() > 0.5).float().mean()),
                    "paraphrase_max_abs_alpha": float(alpha_para.abs().max()),
                }
            )
            if o_fused is not None:
                alpha_overlap = fmap.alpha(o_fused)
                entry["hard_overlap_nonanchor_max_abs_alpha"] = float(alpha_overlap.abs().max())
                entry["hard_overlap_nonanchor_frac_gt_0.1"] = float(
                    (alpha_overlap.abs().max(dim=1).values > 0.1).float().mean()
                )
        except Exception as exc:  # noqa: BLE001 - a failed fit is itself a result
            entry["error"] = f"{type(exc).__name__}: {exc}"
        report["radius_sweep"].append(entry)

    Path(a.output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output_path).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
