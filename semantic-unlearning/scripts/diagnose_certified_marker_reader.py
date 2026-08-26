#!/usr/bin/env python3
"""Ablation: does reading with the certified MARKER beat reading the residual?

Stage 2 currently reads with the writer's realized residual,

    q_h = normalize((I - P) h_f')

which is certified against the protected basis but turns out to be only
~12% marker: the measured run had |q.h_f| = 47.0 against a write amplitude of
5.6, so ~88% of the reader direction is h_f's PRE-EXISTING residual. Nothing
ever asked that part to avoid neighborhood prompts, and it does not:
kappa_cross was 0.653 for q_h versus 0.493 for the intended marker v.

This measures the alternative,

    q_v = normalize((I - P) v),

which keeps the certificate exactly -- it is still in the protected null
space, so B q_v = 0 and every protected logit is unchanged -- while dropping
the residual dilution.

This is an ABLATION FOR ATTRIBUTION, not an expected fix. kappa_cross(v) was
already 0.493, so even a perfectly clean marker-aligned reader cannot reach
the regime the architecture needs. The point is to separate two effects that
would otherwise be confounded if the protection source were changed at the
same time:

  protection      reader                 kappa_cross
  answer-only     residual q_h           0.653   (measured)
  answer-only     certified marker q_v   this run
  relation+answer certified marker q_v   next step

It runs from the saved stage1_writer.pt, so Stage 1 is never retrained.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch

import gagd_compare as gagd
import mcf_marker_write_read as mwr
from mcf_sampling import sample_official_mcf_records


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", required=True)
    p.add_argument("--writer-state", required=True, help="stage1_writer.pt from the main script")
    p.add_argument("--out", required=True)
    p.add_argument("--mcf-path", default="data/multi_counterfact.json")
    p.add_argument("--wikidata-dir", default="data/wikidata")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument("--retain-num", type=int, default=1000)
    p.add_argument("--corpus-docs", type=int, default=50000)
    p.add_argument("--corpus-doc-start", type=int, default=20)
    p.add_argument("--contexts-per-token", type=int, default=512)
    p.add_argument("--corpus-max-tokens", type=int, default=256)
    p.add_argument("--gate-holdout-frac", type=float, default=0.25)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device-map", choices=("single", "auto"), default="single")
    return p.parse_args(list(argv) if argv is not None else None)


def dist(values: Sequence[float]) -> Dict[str, float]:
    vals = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not vals:
        return {"n": 0}
    return {
        "n": len(vals), "min": vals[0], "p10": vals[max(0, len(vals) // 10)],
        "median": vals[len(vals) // 2],
        "p90": vals[min(len(vals) - 1, (9 * len(vals)) // 10)], "max": vals[-1],
    }


def show(label: str, d: Dict[str, float]) -> None:
    if not d.get("n"):
        print(f"  {label:<30} (no data)")
        return
    print(
        f"  {label:<30} min {d['min']:+.4f}  p10 {d['p10']:+.4f}  "
        f"median {d['median']:+.4f}  p90 {d['p90']:+.4f}  max {d['max']:+.4f}"
    )


def leakage(direction: torch.Tensor, signal: torch.Tensor, others: torch.Tensor):
    s = abs(float(torch.dot(direction, signal)))
    if others.numel() == 0:
        return s, float("nan"), float("nan")
    worst = float((others @ direction).abs().max())
    return s, worst, worst / (s + 1e-9)


def main(argv: Sequence[str] | None = None) -> None:
    a = parse_args(argv)
    gagd.set_seed(int(a.seed))

    state = torch.load(Path(a.writer_state).resolve(), map_location="cpu", weights_only=False)
    print(f"loaded writer state: {len(state['writer_row_ids'])} rows, "
          f"{len(state['markers'])} markers")

    data = json.loads(Path(a.mcf_path).read_text(encoding="utf-8"))
    forget_records, _ = sample_official_mcf_records(
        data, int(a.forget_num), int(a.retain_num), int(a.seed)
    )

    ns = argparse.Namespace(
        model_path=a.model_path, dtype=a.dtype,
        device_map=a.device_map, gradient_checkpointing=False,
    )
    model, tok = gagd.load_model_and_tokenizer(ns, for_training=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    device = gagd.first_device(model)
    embedding = model.get_input_embeddings()
    hidden_size = int(embedding.weight.shape[1])

    # reinstate the trained writer
    writer = mwr.EmbeddingRowDelta(embedding, state["writer_row_ids"], device)
    with torch.no_grad():
        writer.delta.detach().copy_(state["writer_delta"].to(device))

    records: List[Dict[str, Any]] = []
    for index, record in enumerate(forget_records):
        rewrite = record["requested_rewrite"]
        if isinstance(rewrite, list):
            rewrite = rewrite[0]
        subject = str(rewrite["subject"])
        case_id = int(record.get("case_id", index))
        if case_id not in state["markers"]:
            continue
        records.append({
            "case_id": case_id,
            "prompt": str(rewrite["prompt"]).format(subject),
            "answer": str(rewrite["target_true"]["str"]),
            "answer_rows": gagd.token_ids_for_text(
                tok, gagd.normalize_answer(str(rewrite["target_true"]["str"]))
            ),
            "neighborhood_prompts": [
                str(x) for x in record.get("neighborhood_prompts", [])
            ],
            "marker": state["markers"][case_id].float(),
            "q_saved": state["realized_readers"][case_id].float(),
            "h_saved": state["h_writer"][case_id].float(),
        })
    print(f"{len(records)} records matched to saved markers")

    # Consistency: the reinstated writer must reproduce the saved hidden
    # states. This MUST use batch size 1, matching how the main script's gate
    # loop computed them. A batched forward pads the shorter prompts, which
    # changes attention kernels and float accumulation order; in bf16 that
    # alone shifts the final hidden state by ~2% relative, which would
    # silently move every kappa in this ablation. Both are computed so the
    # cause is confirmed rather than assumed.
    unbatched = mwr.prompt_hidden(
        model, tok, [r["prompt"] for r in records], device, 1
    )
    drift = max(
        float((unbatched[k] - r["h_saved"]).norm() / (r["h_saved"].norm() + 1e-9))
        for k, r in enumerate(records)
    )
    batched = mwr.prompt_hidden(
        model, tok, [r["prompt"] for r in records], device, int(a.batch_size)
    )
    batching_effect = max(
        float((batched[k] - unbatched[k]).norm() / (unbatched[k].norm() + 1e-9))
        for k in range(len(records))
    )
    print(
        f"writer reinstatement drift (batch=1, vs saved): {drift:.3e}\n"
        f"batching artifact (batch={a.batch_size} vs batch=1): {batching_effect:.3e}"
    )
    if drift > 1e-3:
        raise RuntimeError(
            f"reinstated writer does not reproduce the saved hidden states "
            f"(drift {drift:.3e} at matched batch size); the ablation would "
            f"not be comparable to the run it ablates"
        )
    for k, r in enumerate(records):
        r["h_writer"] = unbatched[k]

    # protection under the writer, with the same reproducible split
    edited_answer_rows = sorted({t for r in records for t in r["answer_rows"]})
    documents = gagd.load_wiki_frequency_documents(
        a.wikidata_dir, int(a.corpus_doc_start), int(a.corpus_docs)
    )
    if not documents:
        raise RuntimeError(f"no documents in {a.wikidata_dir!r}")
    print(f"mining protected contexts under the writer ({len(documents)} docs)...")
    contexts = mwr.mine_token_contexts(
        model, tok, documents, edited_answer_rows, device,
        int(a.contexts_per_token), int(a.corpus_max_tokens),
        int(a.batch_size), hidden_size,
    )
    split_gen = torch.Generator().manual_seed(int(a.seed) + 90102)  # matches main script
    fit: Dict[int, torch.Tensor] = {}
    gate: Dict[int, torch.Tensor] = {}
    for token_id, states in contexts.items():
        perm = torch.randperm(states.shape[0], generator=split_gen)
        n_gate = max(1, int(round(states.shape[0] * float(a.gate_holdout_frac))))
        gate[token_id] = states[perm[:n_gate]]
        fit[token_id] = states[perm[n_gate:]]

    def answer_states(rows, source) -> torch.Tensor:
        parts = [source[t] for t in rows if t in source and source[t].numel()]
        return torch.cat(parts, dim=0) if parts else torch.empty((0, hidden_size))

    bases: Dict[Tuple[int, ...], torch.Tensor] = {}
    for record in records:
        key = tuple(record["answer_rows"])
        if key not in bases:
            bases[key] = mwr.orthonormal_basis(answer_states(record["answer_rows"], fit))

    rows: List[Dict[str, Any]] = []
    for record in records:
        basis = bases[tuple(record["answer_rows"])]
        h = record["h_writer"]
        v = record["marker"]

        def certify(vec: torch.Tensor) -> torch.Tensor:
            projected = vec - basis.T @ (basis @ vec) if basis.numel() else vec.clone()
            return projected / projected.norm().clamp_min(1e-12)

        q_h = certify(h)          # current reader: residual of the hidden state
        q_v = certify(v)          # ablation: residual of the intended marker
        held = answer_states(record["answer_rows"], gate)
        # Batch size 1 here too: kappa_cross compares |q.h_n| against
        # |q.h_forget|, so both sides must be computed the same way or the
        # ratio picks up a padding artifact rather than geometry.
        neigh = (
            mwr.prompt_hidden(model, tok, record["neighborhood_prompts"], device, 1)
            if len(record["neighborhood_prompts"]) >= 3 else torch.empty((0, hidden_size))
        )
        entry: Dict[str, Any] = {"case_id": record["case_id"], "answer": record["answer"]}
        for name, direction in (("q_h", q_h), ("q_v", q_v)):
            s_w, l_w, k_w = leakage(direction, h, held)
            s_c, l_c, k_c = leakage(direction, h, neigh)
            entry.update({
                f"S_{name}": s_w,
                f"kappa_wiki_{name}": k_w,
                f"kappa_cross_{name}": k_c,
                f"cos_v_{name}": float(torch.dot(v, direction)),
                f"cert_{name}": (
                    float((basis @ direction).abs().max()) if basis.numel() else 0.0
                ),
            })
        rows.append(entry)

    summary = {
        "schema_version": 1,
        "protocol": "mcf_certified_marker_reader_ablation_v1",
        "writer_state": str(Path(a.writer_state).resolve()),
        "records": len(rows),
        "writer_reinstatement_drift_batch1": drift,
        "batching_artifact_relative": batching_effect,
        "definition": (
            "q_h = normalize((I-P) h_forget), the current reader. "
            "q_v = normalize((I-P) v), the certified marker reader. Both "
            "satisfy B q = 0 so both keep the exact locality certificate; "
            "they differ only in how much of h_forget's pre-existing residual "
            "the reader inherits."
        ),
        "curves": {
            f"{stat}_{name}": dist([r[f"{stat}_{name}"] for r in rows])
            for name in ("q_h", "q_v")
            for stat in ("S", "kappa_wiki", "kappa_cross", "cos_v", "cert")
        },
        "per_record": rows,
    }
    out = Path(a.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("\n" + "=" * 76)
    print("  ABLATION: residual reader (q_h, current) vs certified marker reader (q_v)")
    print("=" * 76)
    for stat, label in (
        ("kappa_cross", "kappa_cross  (DECISION)"),
        ("kappa_wiki", "kappa_wiki"),
        ("S", "S = |q.h_forget|"),
        ("cos_v", "cos(v, q)"),
        ("cert", "max|B q|  (certificate)"),
    ):
        print(f"\n{label}")
        show("  q_h  residual (current)", summary["curves"][f"{stat}_q_h"])
        show("  q_v  certified marker", summary["curves"][f"{stat}_q_v"])
    mh = summary["curves"]["kappa_cross_q_h"].get("median")
    mv = summary["curves"]["kappa_cross_q_v"].get("median")
    if mh and mv and math.isfinite(mh) and math.isfinite(mv):
        print(
            f"\n  kappa_cross median: {mh:.4f} -> {mv:.4f}"
            f"   ({'improved' if mv < mh else 'no improvement'})"
        )
        print(
            "  Attribution: reader dilution accounts for the change above; whatever "
            "remains is the protection source, addressed by relation-conditioned "
            "contexts H_{y,r}."
        )
    print(f"\nreport: {out}")
    writer.close()


if __name__ == "__main__":
    main()
