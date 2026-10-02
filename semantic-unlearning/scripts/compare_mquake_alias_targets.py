#!/usr/bin/env python3
"""Reference MQuAKE run vs rows retrained with --alias-targets (same router).

    python scripts/compare_mquake_alias_targets.py --reference REF_RUN --candidate ALIAS_RUN \
        --out-prefix OUT/comparison
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _ppl(value):
    if isinstance(value, dict):
        return value.get("ppl")
    return value


def official_metrics(run):
    path = Path(run) / "official_mquake_eval.json"
    if not path.is_file():
        return {}
    d = json.loads(path.read_text())
    out = {}
    for split in ("forget", "retain"):
        for key in ("Eff", "AtomicGen"):
            out[f"{split}_{key}"] = (d.get(split) or {}).get(key)
    out["PPL"] = _ppl(d.get("runtime_aligned_PPL"))
    return out


def alias_metrics(run):
    path = Path(run) / "alias_leak_eval.json"
    if not path.is_file():
        return {}
    d = json.loads(path.read_text())["summary"]
    out = {}
    for prompt_type, block in d.items():
        kinds = block["targets_by_kind"]
        for kind in ("answer", "alias_diff_first"):
            if kind in kinds:
                out[f"{prompt_type}_{kind}_greedy_SURE"] = kinds[kind]["greedy"]["sure"]
                out[f"{prompt_type}_{kind}_first_token_p_SURE"] = kinds[kind]["first_token_probability"]["sure"]
        out[f"{prompt_type}_alias_recovery_rate"] = block["alias_recovery_rate"]
        out[f"{prompt_type}_answer_forgotten_facts"] = block["answer_forgotten_facts"]
    return out


def _fmt(value):
    if value is None:
        return "–"
    return f"{value:.4g}" if isinstance(value, float) else str(value)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--reference", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--out-prefix", required=True)
    a = p.parse_args(argv)
    rows = []
    for name, fn in (("official", official_metrics), ("alias", alias_metrics)):
        ref, cand = fn(a.reference), fn(a.candidate)
        for key in sorted(set(ref) | set(cand), key=lambda k: (k.split("_")[0], k)):
            rows.append({"group": name, "metric": key, "reference": ref.get(key), "alias_targets": cand.get(key)})
    lines = ["# MQuAKE: rows with alias targets vs reference (same router)", "",
             f"reference: `{a.reference}`  ", f"alias targets: `{a.candidate}`", "",
             "Lower is better for forget metrics, alias greedy / first-token p and alias recovery; "
             "higher for retain; PPL should not move.", "",
             "| group | metric | reference | alias targets |", "|---|---|---|---|"]
    lines += [f"| {r['group']} | {r['metric']} | {_fmt(r['reference'])} | {_fmt(r['alias_targets'])} |" for r in rows]
    out = Path(a.out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps(rows, indent=2) + "\n")
    out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
