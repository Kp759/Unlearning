#!/usr/bin/env python3
"""Cross-method, cross-dataset comparison built only from official-evaluator JSON.

Every method under comparison -- ours, ZeroUnlearn, MEMIT, ROME, FT, GA, NPO,
ECO -- produces a *checkpoint*. The three official evaluators
(``mcf_zero_unlearn_official_eval.py``, ``mquake_...``, ``zsre_...``) each take
an arbitrary ``--model-dir``. So the way to guarantee that every method is
scored identically is not to write a new evaluator: it is to route every
checkpoint through the same existing official evaluator and compare only its
output.

This script does the comparison half. It reads official-evaluator JSON files
only, never a method's own self-reported summary, so a method cannot report a
number the shared evaluator did not produce. It refuses to mix datasets,
refuses to silently skip a missing cell, and records provenance for each
number it prints.

``aggregate_mcf_multimethod_results.py`` does something similar but hardcodes
eight internal SURE variants and an MCF-specific file layout. This one takes
methods and paths on the command line and works for any of the three datasets.

Usage
-----
    python scripts/compare_methods_official_eval.py \\
      --dataset mcf \\
      --seeds 1,2,3,4,5 \\
      --method "SURE (ours)=outputs/mcf_subject_emb_margin6_final_seed{seed}/stage1_official_eval.json" \\
      --method "ZeroUnlearn=outputs/mcf_zerounlearn/seed{seed}/official_eval_locked.json" \\
      --method "MEMIT=outputs/mcf_memit_canonical_3b/seed{seed}/official_eval_locked.json" \\
      --method "ROME=outputs/mcf_rome_canonical_3b/seed{seed}/official_eval_locked.json" \\
      --out results/mcf_method_comparison

Each ``--method`` is ``Display Name=path/template``. ``{seed}`` is substituted
per seed. Writes ``<out>.csv``, ``<out>.md`` and ``<out>.json``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# Per-dataset metric contract. "key" is where the number lives in the official
# evaluator's JSON; "lower_is_better" drives the arrow in the rendered table.
#   forget.<name>  -> nested under the "forget" split
#   <name>         -> top level
DATASET_METRICS: Dict[str, Tuple[Tuple[str, str, bool], ...]] = {
    "mcf": (
        ("Eff", "forget.Eff", True),
        ("Gen", "forget.Gen", True),
        ("Spe", "forget.Spe", False),
        ("PPL", "forget_PPL", None),
    ),
    "zsre": (
        ("Eff", "forget.Eff", True),
        ("Gen", "forget.Gen", True),
        ("Spe", "forget.Spe", False),
        ("PPL", "forget_PPL", None),
    ),
    "mquake": (
        ("Eff", "forget.Eff", True),
        ("AtomicGen", "forget.AtomicGen", True),
        ("Retain", "retain.Eff", False),
        ("PPL", "forget_PPL", None),
    ),
}


@dataclass(frozen=True)
class Method:
    display: str
    template: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=sorted(DATASET_METRICS))
    p.add_argument("--seeds", required=True, help="comma-separated, e.g. 1,2,3")
    p.add_argument(
        "--method",
        action="append",
        required=True,
        metavar="NAME=PATH_TEMPLATE",
        help="Repeatable. '{seed}' is substituted. Path must be official-evaluator JSON.",
    )
    p.add_argument("--out", required=True, help="output prefix (no extension)")
    p.add_argument(
        "--metrics",
        default=None,
        help="Comma-separated metric names to keep; defaults to the dataset's full set.",
    )
    p.add_argument(
        "--allow-missing",
        action="store_true",
        help=(
            "Report missing cells as null instead of failing. Off by default: a "
            "silently dropped seed turns an unfavourable mean into a favourable "
            "one, which is exactly the failure mode this comparison must not have."
        ),
    )
    return p.parse_args(argv)


def parse_methods(raw: Sequence[str]) -> List[Method]:
    methods: List[Method] = []
    seen: set = set()
    for item in raw:
        if "=" not in item:
            raise SystemExit(f"--method must be NAME=PATH_TEMPLATE, got {item!r}")
        name, template = item.split("=", 1)
        name, template = name.strip(), template.strip()
        if not name or not template:
            raise SystemExit(f"--method has an empty name or path: {item!r}")
        if name in seen:
            raise SystemExit(f"duplicate method name {name!r}")
        seen.add(name)
        methods.append(Method(name, template))
    return methods


def dig(payload: Mapping[str, Any], dotted: str) -> Optional[float]:
    node: Any = payload
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            return None
        node = node[part]
    if isinstance(node, bool) or node is None:
        return None
    if isinstance(node, (list, tuple)):
        node = node[0] if node else None
    try:
        value = float(node)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def looks_official(payload: Mapping[str, Any], dataset: str) -> bool:
    """Reject a method's own summary masquerading as an evaluator result.

    The official evaluators all emit a 'forget' split; the per-method training
    summaries (stage1_config.json, repair_summary.json) do not. This is a
    structural check, not a trusted self-declared field.
    """
    forget = payload.get("forget")
    if not isinstance(forget, Mapping):
        return False
    required = {"Eff"}
    return required.issubset(set(forget))


def load_cell(path: Path, dataset: str, metrics) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not looks_official(payload, dataset):
        raise SystemExit(
            f"{path} does not look like {dataset} official-evaluator output "
            "(no 'forget' split with an Eff field). Point this at the evaluator's "
            "--out JSON, not a method's own training summary."
        )
    values = {name: dig(payload, key) for name, key, _ in metrics}
    return {
        "values": values,
        "provenance": {
            "path": str(path),
            "method_field": payload.get("method"),
            "seed_field": payload.get("seed"),
            "unlearn_num": payload.get("unlearn_num")
            or payload.get("unlearn_num_instances"),
        },
    }


def summarize(values: Sequence[Optional[float]]) -> Dict[str, Optional[float]]:
    present = [v for v in values if v is not None]
    if not present:
        return {"mean": None, "sd": None, "n": 0, "min": None, "max": None}
    sd = statistics.pstdev(present) if len(present) > 1 else 0.0
    return {
        "mean": statistics.mean(present),
        "sd": sd,
        "n": len(present),
        "min": min(present),
        "max": max(present),
    }


def fmt(value: Optional[float], places: int = 2) -> str:
    return "--" if value is None else f"{value:.{places}f}"


def main(argv: Sequence[str] | None = None) -> None:
    a = parse_args(argv)
    methods = parse_methods(a.method)
    seeds = [int(s) for s in str(a.seeds).split(",") if str(s).strip()]
    if not seeds:
        raise SystemExit("--seeds parsed to nothing")

    metrics = DATASET_METRICS[a.dataset]
    if a.metrics:
        keep = {m.strip() for m in a.metrics.split(",") if m.strip()}
        metrics = tuple(m for m in metrics if m[0] in keep)
        if not metrics:
            raise SystemExit(f"--metrics kept nothing; available: "
                             f"{[m[0] for m in DATASET_METRICS[a.dataset]]}")
    metric_names = [m[0] for m in metrics]

    per_seed: List[Dict[str, Any]] = []
    provenance: List[Dict[str, Any]] = []
    missing: List[str] = []

    for method in methods:
        for seed in seeds:
            path = Path(method.template.format(seed=seed))
            if not path.exists():
                missing.append(f"{method.display} seed {seed}: {path}")
                per_seed.append(
                    {"method": method.display, "seed": seed,
                     **{n: None for n in metric_names}}
                )
                continue
            cell = load_cell(path, a.dataset, metrics)
            per_seed.append(
                {"method": method.display, "seed": seed, **cell["values"]}
            )
            provenance.append({"method": method.display, "seed": seed,
                               **cell["provenance"]})

    if missing and not a.allow_missing:
        listing = "\n  ".join(missing)
        raise SystemExit(
            f"{len(missing)} missing result file(s):\n  {listing}\n"
            "Refusing to aggregate an incomplete grid -- a dropped seed silently "
            "changes a mean. Pass --allow-missing to report them as null."
        )

    summary: List[Dict[str, Any]] = []
    for method in methods:
        rows = [r for r in per_seed if r["method"] == method.display]
        entry: Dict[str, Any] = {"method": method.display}
        for name in metric_names:
            entry[name] = summarize([r[name] for r in rows])
        summary.append(entry)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "seed", *metric_names])
        for r in per_seed:
            w.writerow([r["method"], r["seed"], *[fmt(r[n]) for n in metric_names]])
        w.writerow([])
        w.writerow(["method", "stat", *metric_names])
        for e in summary:
            w.writerow([e["method"], "mean", *[fmt(e[n]["mean"]) for n in metric_names]])
            w.writerow([e["method"], "sd", *[fmt(e[n]["sd"]) for n in metric_names]])

    arrows = {name: ("↓" if low is True else "↑" if low is False else "")
              for name, _, low in metrics}
    lines = [
        f"# {a.dataset.upper()} method comparison",
        "",
        f"Seeds: {', '.join(str(s) for s in seeds)} "
        f"(n={len(seeds)}) · all numbers from the {a.dataset} official evaluator.",
        "",
        "| Method | " + " | ".join(f"{n} {arrows[n]}".strip() for n in metric_names) + " |",
        "|---|" + "---|" * len(metric_names),
    ]
    for e in summary:
        cells = []
        for n in metric_names:
            s = e[n]
            cells.append("--" if s["mean"] is None
                         else f"{s['mean']:.2f} ± {s['sd']:.2f}")
        lines.append(f"| {e['method']} | " + " | ".join(cells) + " |")
    lines += ["", "## Per seed", "",
              "| Method | Seed | " + " | ".join(metric_names) + " |",
              "|---|---|" + "---|" * len(metric_names)]
    for r in per_seed:
        lines.append(
            f"| {r['method']} | {r['seed']} | "
            + " | ".join(fmt(r[n]) for n in metric_names) + " |"
        )
    out.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    out.with_suffix(".json").write_text(
        json.dumps(
            {
                "dataset": a.dataset,
                "seeds": seeds,
                "metrics": [
                    {"name": n, "source_key": k, "lower_is_better": low}
                    for n, k, low in metrics
                ],
                "methods": [m.display for m in methods],
                "per_seed": per_seed,
                "summary": summary,
                "provenance": provenance,
                "missing": missing,
                "evaluator_contract": (
                    "every number is read from the dataset's official evaluator "
                    "JSON; no method's self-reported summary is consulted"
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print("\n".join(lines[:6 + len(summary)]))
    if missing:
        print(f"\n{len(missing)} missing cell(s) reported as null.")
    print(f"\nWrote {out.with_suffix('.csv')}, {out.with_suffix('.md')}, "
          f"{out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
