#!/usr/bin/env python3
"""Build the multi-fact person benchmark (locked split) for one seed.

Several facts about the same real person are stated in ONE sentence; one fact
is forgotten and the others must be retained. See multifact_person_data.py for
the sentence and probe design.

    python -u scripts/build_multifact_person_dataset.py \
        --mcf-path data/multi_counterfact.json \
        --mquake-path data/MQuAKE-CF-3k-v2.json --mquake-path data/MQuAKE-CF.json \
        --model-path <llama> --output-dir outputs/multifact_person_v1/seed1/data \
        --seed 1 --local-files-only

Steps:
  1. candidate facts of people from MCF + MQuAKE (Wikidata triples, their own
     direct cloze prompts), one verb phrase per relation
  2. knowledge filter with the frozen base model: a fact is kept only if the
     base model gets every object token right (teacher-forced top-1, the
     official Eff convention) on BOTH its direct cloze and its "{S} {vp}"
     prompt, so forget/retain accuracies start from a model that knows them
  3. per person: distinct objects, one relation per answer group, <= 4 facts;
     people with >= 2 such facts are eligible; name-colliding people dropped
  4. seed sampling: 50 forget people (>= 3-fact sentences first), one forget
     fact each; 100 disjoint retain people
Writes:
  training_visible_forget.json  forget facts' direct prompts only (retain-blind,
                                MQuAKE direct-record format)
  eval_probes.json              every direct / single / multi-fact probe (eval only)
  split_manifest.json           sampling, sources + sha256, filter statistics
  dataset_report.md             counts and example sentences
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import multifact_person_data as mf  # noqa: E402


def knowledge_filter(people, model_path, *, min_candidates, device, dtype, batch_size,
                     local_files_only, threshold):
    """{fact_key: {"direct": acc, "single": acc}} for people with enough candidate facts."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from mquake_zero_unlearn_official_eval import is_llama_like

    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True,
                                        local_files_only=local_files_only)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=getattr(torch, dtype), local_files_only=local_files_only,
        attn_implementation="eager",
    ).to(device).eval()
    model.requires_grad_(False)
    llama_like = is_llama_like(model, tok)

    probes = []
    for subject, relations in sorted(people.items()):
        if len(relations) < min_candidates:
            continue
        for fact in relations.values():
            before, _ = mf.verb_phrase(fact)
            for kind, prefix in (("direct", fact["direct_template"].format(subject)),
                                 ("single", f"{subject} {before}")):
                probes.append({"id": f"probe_{len(probes)}", "person": subject,
                               "fact_key": fact["key"], "relation": fact["relation"],
                               "answer": fact["object"], "role": "candidate", "type": kind,
                               "position": None, "k": None, "forget_in_context": False,
                               "prefix": prefix})
    print(json.dumps({"phase": "knowledge_filter", "probes": len(probes)}), flush=True)
    rows = mf.score_probes(model, tok, probes, device, llama_like=llama_like,
                           batch_size=batch_size)
    scores = defaultdict(dict)
    for row in rows:
        scores[row["fact_key"]][row["type"]] = row["accuracy"]
    known = {key for key, s in scores.items()
             if s.get("direct", 0.0) >= threshold and s.get("single", 0.0) >= threshold}
    return known, dict(scores), {"llama_like": llama_like, "scored_facts": len(scores)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mcf-path", default=None)
    p.add_argument("--mquake-path", action="append", default=[],
                   help="repeatable; MQuAKE-CF-3k-v2.json and MQuAKE-CF.json")
    p.add_argument("--model-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--forget-num", type=int, default=50)
    p.add_argument("--retain-persons", type=int, default=100)
    p.add_argument("--min-facts", type=int, default=2)
    p.add_argument("--max-facts", type=int, default=4)
    p.add_argument("--any-answer-groups", action="store_true",
                   help="allow two relations of one answer group in a sentence")
    p.add_argument("--known-threshold", type=float, default=1.0,
                   help="per-probe token accuracy a fact needs on BOTH prompts")
    p.add_argument("--skip-knowledge-filter", action="store_true",
                   help="treat every candidate as known (debug only)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--local-files-only", action="store_true")
    a = p.parse_args(argv)

    output = Path(a.output_dir).resolve()
    sources, mcf_records, mquake_sources = [], [], []
    if a.mcf_path:
        path = Path(a.mcf_path).resolve()
        mcf_records = json.loads(path.read_text())
        sources.append({"name": "MultiCounterFact", "path": str(path), "sha256": mf.file_sha256(path)})
    for raw in a.mquake_path:
        path = Path(raw).resolve()
        if not path.is_file():
            print(json.dumps({"phase": "source_missing", "path": str(path)}), flush=True)
            continue
        mquake_sources.append((path.stem, json.loads(path.read_text())))
        sources.append({"name": path.stem, "path": str(path), "sha256": mf.file_sha256(path)})
    if not sources:
        raise SystemExit("No source dataset found")
    people, candidate_stats = mf.candidate_facts(mcf_records, mquake_sources)
    print(json.dumps({"phase": "candidates", **candidate_stats}), flush=True)

    if a.skip_knowledge_filter:
        known = {f["key"] for rels in people.values() for f in rels.values()}
        scores, filter_info = {}, {"skipped": True}
    else:
        known, scores, filter_info = knowledge_filter(
            people, Path(a.model_path).resolve(), min_candidates=a.min_facts, device=a.device,
            dtype=a.dtype, batch_size=a.batch_size, local_files_only=a.local_files_only,
            threshold=a.known_threshold,
        )
    by_relation = defaultdict(lambda: [0, 0])
    for key, s in scores.items():
        relation = key.split("\t")[1]
        by_relation[relation][1] += 1
        by_relation[relation][0] += int(key in known)
    print(json.dumps({"phase": "knowledge_filter_done", "known": len(known),
                      "scored": len(scores)}), flush=True)

    records, persons, probes, sampling = mf.build_split(
        people, known, seed=a.seed, forget_num=a.forget_num, retain_persons=a.retain_persons,
        min_facts=a.min_facts, max_facts=a.max_facts,
        distinct_answer_groups=not a.any_answer_groups,
    )
    output.mkdir(parents=True, exist_ok=False)
    visible = output / "training_visible_forget.json"
    visible.write_text(json.dumps(records, indent=2) + "\n")
    probes_path = output / "eval_probes.json"
    probes_path.write_text(json.dumps({
        "schema": "multifact_person_probes_v1", "dataset": mf.DATASET, "seed": a.seed,
        "persons": persons, "probes": probes}, indent=2) + "\n")
    counts = Counter((p["role"], p["type"]) for p in probes)
    manifest = {
        "dataset": mf.DATASET,
        "seed": a.seed,
        "sampling": sampling,
        "sources": sources,
        "candidate_stats": candidate_stats,
        "knowledge_filter": {
            **filter_info,
            "rule": ("every object token top-1 (teacher-forced, official Eff convention) on "
                     "BOTH the direct cloze and the '{S} {vp}' prompt"),
            "threshold": a.known_threshold,
            "dtype": a.dtype,
            "known_facts": len(known),
            "known_by_relation": {r: {"known": v[0], "scored": v[1]}
                                  for r, v in sorted(by_relation.items())},
        },
        "model_path": str(Path(a.model_path).resolve()),
        "training_visible_path": str(visible),
        "training_visible_sha256": mf.file_sha256(visible),
        "eval_probes_path": str(probes_path),
        "eval_probes_sha256": mf.file_sha256(probes_path),
        "probe_counts": {f"{role}/{kind}": n for (role, kind), n in sorted(counts.items())},
        "verb_phrases": mf.VERB_PHRASES,
        "protocol": {
            "training_visible": "forget facts' direct cloze prompts only",
            "retain_facts_used_for_training_or_selection": False,
            "multi_fact_sentences_used_for_training_or_selection": False,
            "single_vp_prompts_used_for_training_or_selection": False,
            "knowledge_filter_uses_base_model_only": True,
        },
    }
    (output / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    lines = [f"# {mf.DATASET}, seed {a.seed}", "",
             f"- sources: {', '.join(s['name'] for s in sources)}",
             f"- people with candidate facts: {candidate_stats['people']}; facts known by the "
             f"base model: {len(known)} of {len(scores) or 'n/a'} scored",
             f"- eligible people: {sampling['eligible_people']}; forget people: "
             f"{sampling['forget_num_instances']}; retain people: {sampling['retain_person_count']}",
             f"- facts per forget sentence: {sampling['facts_per_forget_sentence']}", "",
             "| role / probe | count |", "|---|---|"]
    lines += [f"| {role} / {kind} | {n} |" for (role, kind), n in sorted(counts.items())]
    lines += ["", "## Example forget sentences", ""]
    for person in [x for x in persons if x["role"] == "forget_person"][:8]:
        sentence, _ = mf.render_sentence(person["subject"], person["facts"])
        forget = next(f for f in person["facts"] if f["key"] == person["forget_fact_key"])
        lines.append(f"- {sentence}  — forget **{forget['object']}** ({forget['relation']})")
    lines += ["", "## Example multi-fact probes (forget role)", ""]
    for probe in [p for p in probes if p["role"] == "forget" and p["type"] == "multi"][:6]:
        lines.append(f"- `{probe['prefix']}` → {probe['answer']} (position {probe['position']})")
    (output / "dataset_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
