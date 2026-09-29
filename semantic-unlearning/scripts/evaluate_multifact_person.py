#!/usr/bin/env python3
"""Official-style evaluation of the multi-fact person benchmark.

    python -u scripts/evaluate_multifact_person.py \
        --run-dir outputs/multifact_person_v1/seed1/L19/linear_global \
        --wikidata-dir data/wikidata --device cuda --dtype bfloat16 --local-files-only

The same probes are scored twice with the official teacher-forced token
convention (MQuAKE/ZeroUnlearn): first with the frozen base model (before any
hook exists), then with SURE (router + trained rows, request boundary = the
probe prefix). Accuracy = 100 x case-macro token top-1.

    forget               direct (Eff, training-visible), single (held-out
                         phrasing), multi (held-out: other facts of the same
                         person precede it in the same sentence)       lower is better
    retain_same_person   the other facts of each forget person, same three
                         forms; multi split by whether the forgotten fact is
                         stated earlier in the same sentence           higher is better
    retain_other_person  people not in the forget set                  higher is better

Routing: fraction of probes on which a head fires (for forget probes, the
fraction routed to the fact's own row). PPL: runtime-aligned on the shipped
wikidata text, base and SURE. Writes official_multifact_eval.json and .md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import multifact_person_data as mf  # noqa: E402


def _fmt(value):
    return "–" if value is None else (f"{value:.2f}" if isinstance(value, float) else str(value))


def markdown(result):
    base, sure = result["base"]["summary"], result["sure"]["summary"]
    lines = [f"# Multi-fact person benchmark — seed {result['seed']}, L{result['layer']}", "",
             "Accuracy = 100 × case-macro teacher-forced token top-1. Forget: lower is better; "
             "retain: higher is better.", "",
             "| role | probe | base | SURE | SURE routes (fire / own row) | probes |",
             "|---|---|---|---|---|---|"]
    for role in ("forget", "retain_same_person", "retain_other_person"):
        for kind in ("direct", "single", "multi"):
            b, s = base[role][kind], sure[role][kind]
            route = _fmt(s.get("route_active_fraction"))
            if role == "forget":
                route += f" / {_fmt(s.get('routed_to_own_row_fraction'))}"
            lines.append(f"| {role} | {kind} | {_fmt(b['accuracy'])} | {_fmt(s['accuracy'])} | "
                         f"{route} | {s['probes']} |")
    same_b, same_s = base["retain_same_person"], sure["retain_same_person"]
    lines += ["", "Same-person retain in multi-fact sentences:",
              f"- forgotten fact stated earlier in the sentence: base "
              f"{_fmt(same_b.get('multi_forget_fact_in_context'))} → SURE "
              f"{_fmt(same_s.get('multi_forget_fact_in_context'))}",
              f"- forgotten fact not in the prefix: base "
              f"{_fmt(same_b.get('multi_forget_fact_not_in_context'))} → SURE "
              f"{_fmt(same_s.get('multi_forget_fact_not_in_context'))}", "",
              "Forget, multi-fact, by position of the forgotten fact (1 = one fact before it):"]
    for pos, value in sure["forget"]["multi_by_position"].items():
        lines.append(f"- position {pos}: base {_fmt(base['forget']['multi_by_position'].get(pos))} → "
                     f"SURE {_fmt(value)}")
    lines += ["", f"PPL (runtime-aligned): base {_fmt(result['base']['runtime_aligned_PPL'])} → "
                  f"SURE {_fmt(result['sure']['runtime_aligned_PPL'])}"]
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--probes", default=None, help="default: the manifest's eval_probes_path")
    p.add_argument("--wikidata-dir", default="data/wikidata")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--skip-ppl", action="store_true")
    p.add_argument("--out", default=None)
    a = p.parse_args(argv)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from linear_router import load_router_artifact
    from mcf_zero_unlearn_official_eval import (
        dtype_from_str,
        load_official_ppl_text,
        runtime_aligned_perplexity,
    )
    from mquake_zero_unlearn_official_eval import is_llama_like

    run_dir = Path(a.run_dir).resolve()
    manifest = json.loads((run_dir / "association_manifest.json").read_text())
    if manifest.get("dataset") != mf.DATASET:
        raise ValueError(f"{run_dir} is not a {mf.DATASET} run")
    probes_path = Path(a.probes or manifest["eval_probes_path"]).resolve()
    expected_sha = manifest.get("eval_probes_sha256")
    if expected_sha and mf.file_sha256(probes_path) != expected_sha:
        raise ValueError("eval_probes.json changed since the split was locked")
    data = json.loads(probes_path.read_text())
    probes = data["probes"]
    artifact = torch.load(run_dir / "fact_association_embeddings.pt", map_location="cpu",
                          weights_only=False)
    row_of_key = {str(f["association_key"]): i for i, f in enumerate(artifact["facts"])}
    forget_keys = {x["forget_fact_key"] for x in data["persons"] if x["role"] == "forget_person"}
    if set(row_of_key) != forget_keys:
        raise RuntimeError("The bank's facts are not exactly the split's forget facts")

    model_path = Path(manifest["model_path"]).resolve()
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True,
                                        local_files_only=a.local_files_only)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=dtype_from_str(a.dtype), local_files_only=a.local_files_only,
        attn_implementation="eager",
    ).to(a.device).eval()
    base_model.requires_grad_(False)
    base_model.config.use_cache = False
    device = next(base_model.parameters()).device
    llama_like = is_llama_like(base_model, tok)
    ppl_text = None if a.skip_ppl else load_official_ppl_text(a.wikidata_dir)

    # Base first, before the router's hook is attached to the model.
    base_rows = mf.score_probes(base_model, tok, probes, device, llama_like=llama_like,
                                batch_size=a.batch_size)
    base_ppl = (None if ppl_text is None else
                runtime_aligned_perplexity(base_model, tok, ppl_text, device, max_input_length=100)["ppl"])
    print(json.dumps({"phase": "base_done", **mf.headline(mf.summarize(base_rows))}), flush=True)

    model, bank = load_router_artifact(base_model, artifact)
    model.eval()
    sure_rows = mf.score_probes(model, tok, probes, device, llama_like=llama_like,
                                batch_size=a.batch_size, bank=bank, row_of_key=row_of_key)
    sure_ppl = (None if ppl_text is None else
                runtime_aligned_perplexity(model, tok, ppl_text, device, max_input_length=100)["ppl"])

    base_summary, sure_summary = mf.summarize(base_rows), mf.summarize(sure_rows)
    result = {
        "dataset": mf.DATASET,
        "seed": manifest.get("seed"),
        "layer": int(artifact["layer"]),
        "run_dir": str(run_dir),
        "eval_probes_path": str(probes_path),
        "router": {"architecture": artifact.get("architecture"),
                   "routing_policy": artifact.get("routing_policy"),
                   "decision_rule": artifact.get("decision_rule"),
                   "bias_calibration_global_shift": (artifact.get("bias_calibration") or {}).get("global_shift")},
        "metric_definition": {
            "accuracy": "100 x case-macro teacher-forced object-token top-1 (official MQuAKE convention)",
            "forget": "lower is better", "retain": "higher is better",
            "multi": "fact asked at position >= 1 of a sentence stating other facts of the same person",
        },
        "headline": {"base": mf.headline(base_summary), "sure": mf.headline(sure_summary)},
        "base": {"summary": base_summary, "runtime_aligned_PPL": base_ppl},
        "sure": {"summary": sure_summary, "runtime_aligned_PPL": sure_ppl},
        "rows": {"base": base_rows, "sure": sure_rows},
        "runtime_counters": bank.counters(),
    }
    out = Path(a.out).resolve() if a.out else run_dir / "official_multifact_eval.json"
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    text = markdown(result)
    out.with_suffix(".md").write_text(text)
    print(text, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
