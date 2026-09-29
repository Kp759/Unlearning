#!/usr/bin/env python3
"""Eval-DU+ split for SURE: forget facts, training-visible prompts, all probes.

    python -u scripts/build_evaldu_plus_split.py \
        --upstream data/evaldu_plus_upstream \
        --output-dir outputs/evaldu_plus_v1/seed1/data

--split facts100 (default): the paper's 100 random facts (unlearn_fact_id.pt);
         people12: the 12 people / 102 facts split. Both are fixed upstream
         files; "seed 1" labels this run, it draws nothing.
--unlearn-data mul (default, UL-Mul: 3 new paraphrases per forget fact) | single.

Writes split_manifest.json (facts, forget ids, training prompts, provenance)
and eval_probes.json (test / unlearn / chunk probes; evaluation only).
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaldu_plus_data as ed  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--upstream", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--split", choices=("facts100", "people12"), default="facts100")
    p.add_argument("--unlearn-data", choices=("mul", "single"), default="mul")
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args(argv)

    upstream_dir = Path(a.upstream).resolve()
    data = ed.load_upstream(upstream_dir)
    facts, raw = data["facts"], data["raw"]
    forget = sorted({int(k) for k in data["splits"][a.split]})
    key = "unlearn" if a.unlearn_data == "mul" else "unlearn_single"
    prompts, dropped = ed.training_prompts(facts, raw, forget, key=key)
    probes = {
        "test": ed.sentence_probes(facts, raw, "test"),
        "unlearn": ed.sentence_probes(facts, raw, "unlearn"),
        "chunk": ed.chunk_probes(facts, raw, data["chunks"], data["names"]),
    }
    try:
        commit = subprocess.run(["git", "-C", str(upstream_dir), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None

    output = Path(a.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    probes_path = output / "eval_probes.json"
    probes_path.write_text(json.dumps(probes) + "\n")
    in_bank = sorted({int(r["fact"]) for r in prompts})
    forget_people = {person for k in forget for person in facts[k]["people"]}
    manifest = {
        "dataset": ed.DATASET, "seed": a.seed, "split": a.split, "unlearn_data": a.unlearn_data,
        "upstream": ed.UPSTREAM, "upstream_commit": commit,
        "upstream_sources_sha256": data["sources"], "upstream_checks": data["checks"],
        "forget": forget,
        "forget_facts_with_training_prompt": in_bank,
        "forget_facts_without_training_prompt": sorted(set(forget) - set(in_bank)),
        "training_prompts": prompts,
        "training_prompts_dropped": dropped,
        "facts": facts,
        "eval_probes_path": str(probes_path),
        "eval_probes_sha256": ed.file_sha256(probes_path),
        "counts": {
            "facts": len(facts), "forget": len(forget), "forget_in_bank": len(in_bank),
            "retain_same_person_facts": sum(
                1 for f in facts if f["index"] not in set(forget)
                and set(f["people"]) & forget_people),
            "probes": {name: len(rows) for name, rows in probes.items()},
            "forget_relations": dict(Counter(facts[k]["relation"] for k in forget)),
        },
        "protocol": {
            "training_visible": f"UL-{a.unlearn_data.capitalize()} paraphrases of the forget facts, "
                                "cut before the completion word, naming one of the fact's people",
            "test_or_chunk_probes_used_for_training_or_selection": False,
            "retain_facts_used_for_training_or_selection": False,
        },
    }
    (output / "split_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "evaldu_split_ready", **manifest["counts"],
                      "training_prompts": len(prompts), "dropped": dropped,
                      "upstream_commit": commit, "output_dir": str(output)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
