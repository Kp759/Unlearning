#!/usr/bin/env python3
"""Evaluate SURE on Eval-DU+ with the paper's knowledge score.

    python -u scripts/evaluate_evaldu_plus.py \
        --run-dir outputs/evaldu_plus_v1/seed1/L19/linear_global --local-files-only

Knowledge score (upstream eval_completion_word): exp(mean log-prob) of the
completion word's tokens given the preceding tokens. Scored twice on the same
probes: the fine-tuned model (before unlearning; hooks not attached yet) and
SURE (router + trained rows, request boundary = the completion's position).

    test     held-out paraphrases, 3 per fact (the paper's extraction trade-off)
    chunk    the FT-Mul-Chunk texts cut before a fact's completion: other facts
             of the same person are stated earlier in the same text
    unlearn  UL-Mul paraphrases (SURE's training-visible prompts for forget facts)

Groups (fact level, mean over a fact's probes): forget (the split's facts; lower
is better), retain_same_person (facts sharing a person with a forget fact),
retain_other_person, retain_all (higher is better). "normalized" = SURE / base,
the paper's normalization by the fine-tuned model. Each block is reported for
all probes and for probes whose prefix names one of the fact's people (SURE
routes on the person; without a name the question is unanswerable).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaldu_plus_data as ed  # noqa: E402

GROUPS = ("forget", "retain_same_person", "retain_other_person", "retain_all")


def _fmt(value, digits=3):
    return "–" if value is None else (f"{value:.{digits}f}" if isinstance(value, float) else str(value))


def _ratio(a, b):
    return None if a is None or b in (None, 0) else a / b


def compare(base, sure):
    out = {}
    for probe_set, block in sure.items():
        out[probe_set] = {}
        for label, groups in block.items():
            if label not in ("all", "person_in_prefix"):
                continue
            out[probe_set][label] = {
                g: {"base": base[probe_set][label][g]["knowledge_score"],
                    "sure": groups[g]["knowledge_score"],
                    "normalized": _ratio(groups[g]["knowledge_score"],
                                         base[probe_set][label][g]["knowledge_score"]),
                    "facts": groups[g]["facts"],
                    "route_active_fraction": groups[g].get("route_active_fraction"),
                    "routed_to_own_row_fraction": groups[g].get("routed_to_own_row_fraction")}
                for g in GROUPS
            }
    return out


def markdown(result):
    lines = [f"# Eval-DU+ (FT-Mul-Chunk) — SURE, split {result['split']}, seed {result['seed']}, "
             f"L{result['layer']}", "",
             "Knowledge score = exp(mean log-prob) of the completion word (the paper's metric). "
             "Base = the fine-tuned model before unlearning. Normalized = SURE / base. "
             "Forget: lower is better; retain: higher is better.", "",
             f"Forget facts in the SURE bank: {result['coverage']['forget_in_bank']} / "
             f"{result['coverage']['forget']} (a fact needs a training prompt that names one of its people).", ""]
    names = {"test": "Held-out paraphrases (extraction)",
             "chunk": "Inside the multi-fact biography chunks",
             "unlearn": "UL-Mul paraphrases (training-visible for forget facts)"}
    for probe_set in ("test", "chunk", "unlearn"):
        if probe_set not in result["comparison"]:
            continue
        for label in ("person_in_prefix", "all"):
            block = result["comparison"][probe_set][label]
            lines += [f"## {names[probe_set]} — {'prefix names the person' if label == 'person_in_prefix' else 'all probes'}", "",
                      "| group | facts | base | SURE | normalized | SURE fire rate |", "|---|---|---|---|---|---|"]
            for g in GROUPS:
                row = block[g]
                fire = _fmt(row.get("route_active_fraction"), 2)
                if g == "forget":
                    fire += f" (own row {_fmt(row.get('routed_to_own_row_fraction'), 2)})"
                lines.append(f"| {g} | {row['facts']} | {_fmt(row['base'])} | {_fmt(row['sure'])} | "
                             f"{_fmt(row['normalized'])} | {fire} |")
            lines.append("")
    extra_b = result["base"]["summary"].get("chunk", {}).get("retain_same_person_forget_fact_earlier_in_chunk")
    extra_s = result["sure"]["summary"].get("chunk", {}).get("retain_same_person_forget_fact_earlier_in_chunk")
    if extra_s:
        lines += [f"Retained same-person facts stated AFTER a forgotten fact in the same chunk: base "
                  f"{_fmt(extra_b['knowledge_score'])} → SURE {_fmt(extra_s['knowledge_score'])} "
                  f"({extra_s['probes']} probes).", ""]
    lines.append(f"PPL (runtime-aligned): base {_fmt(result['base']['runtime_aligned_PPL'], 2)} → "
                 f"SURE {_fmt(result['sure']['runtime_aligned_PPL'], 2)}")
    return "\n".join(lines) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--sets", default="test,chunk,unlearn")
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

    run_dir = Path(a.run_dir).resolve()
    manifest = json.loads((run_dir / "association_manifest.json").read_text())
    if manifest.get("dataset") != ed.DATASET:
        raise ValueError(f"{run_dir} is not an {ed.DATASET} run")
    split = json.loads(Path(manifest["split_manifest_path"]).read_text())
    probes_path = Path(split["eval_probes_path"])
    if ed.file_sha256(probes_path) != split["eval_probes_sha256"]:
        raise ValueError("eval_probes.json changed since the split was built")
    all_probes = json.loads(probes_path.read_text())
    sets = [s for s in a.sets.split(",") if s]
    probes = [p for s in sets for p in all_probes[s]]
    probes_by_id = {p["id"]: p for p in probes}
    facts, forget = split["facts"], split["forget"]

    artifact = torch.load(run_dir / "fact_association_embeddings.pt", map_location="cpu",
                          weights_only=False)
    row_of_fact = {int(f["index"]): i for i, f in enumerate(artifact["facts"])}
    if not set(row_of_fact) <= set(forget):
        raise RuntimeError("The bank holds a fact outside the forget split")

    model_path = Path(manifest["model_path"]).resolve()
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=a.local_files_only)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=dtype_from_str(a.dtype), local_files_only=a.local_files_only,
        attn_implementation="eager").to(a.device).eval()
    base_model.requires_grad_(False)
    base_model.config.use_cache = False
    device = next(base_model.parameters()).device
    ppl_text = None if a.skip_ppl else load_official_ppl_text(a.wikidata_dir)

    base_rows, skipped = ed.knowledge_scores(base_model, tok, probes, device, batch_size=a.batch_size)
    base_ppl = (None if ppl_text is None else
                runtime_aligned_perplexity(base_model, tok, ppl_text, device, max_input_length=100)["ppl"])
    model, bank = load_router_artifact(base_model, artifact)
    model.eval()
    sure_rows, _ = ed.knowledge_scores(model, tok, probes, device, batch_size=a.batch_size,
                                       bank=bank, row_of_fact=row_of_fact)
    sure_ppl = (None if ppl_text is None else
                runtime_aligned_perplexity(model, tok, ppl_text, device, max_input_length=100)["ppl"])

    base_summary, groups = ed.summarize(base_rows, facts, forget, probes_by_id)
    sure_summary, _ = ed.summarize(sure_rows, facts, forget, probes_by_id)
    result = {
        "dataset": ed.DATASET, "split": split["split"], "seed": split["seed"],
        "layer": int(artifact["layer"]), "run_dir": str(run_dir), "model_path": str(model_path),
        "router": {"architecture": artifact.get("architecture"),
                   "routing_policy": artifact.get("routing_policy"),
                   "decision_rule": artifact.get("decision_rule"),
                   "bias_calibration_global_shift": (artifact.get("bias_calibration") or {}).get("global_shift")},
        "coverage": {"forget": len(forget), "forget_in_bank": len(row_of_fact),
                     "probes_scored": len(sure_rows), "probes_skipped_completion_not_found": len(skipped)},
        "metric_definition": {
            "knowledge_score": "exp(mean log-prob) of the completion word tokens (upstream eval_completion_word)",
            "normalized": "SURE / fine-tuned model, per group",
            "fact_level": "mean over a fact's probes, then mean over facts in the group",
        },
        "comparison": compare(base_summary, sure_summary),
        "base": {"summary": base_summary, "runtime_aligned_PPL": base_ppl},
        "sure": {"summary": sure_summary, "runtime_aligned_PPL": sure_ppl},
        "rows": {"base": base_rows, "sure": sure_rows},
        "runtime_counters": bank.counters(),
    }
    out = Path(a.out).resolve() if a.out else run_dir / "official_evaldu_eval.json"
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    text = markdown(result)
    out.with_suffix(".md").write_text(text)
    print(text, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
