"""Leaderboard: aggregate results.json files (methods x seeds) into tables and a plot.

    python -m scalpel report --runs results/pilot/*/*/results.json --out results/pilot/leaderboard

writes leaderboard.md / leaderboard.json / pareto.png. Means over seeds ± sd; the
per-seed bootstrap CI is shown when there is one seed.
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

KEYS = ("SCALPEL", "SCALPEL_core", "E", "E_mean", "E_min", "E_direct", "E_complete", "L", "L_mean", "L_min", "U", "R")


def _ms(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0
    return m, sd


# Leaderboard names for method keys: the method we built runs as `ours` (its run directories keep
# that name) and is shown as SURE; the first SURE adapter (exact matching, request prompts only)
# is shown as sure_original and excluded from the published leaderboard (--exclude sure).
DISPLAY_NAMES = {"ours": "SURE", "ours_l16": "SURE_l16", "ours_early": "SURE_early", "sure": "sure_original"}


def load_runs(paths: Sequence[Path], exclude: Sequence[str] = ()) -> List[Dict]:
    """Every results.json, except copies set aside next to the method's own run directory
    (runs/ours_judge05, runs/noop.incomplete_<time>): a directory named after the method plus a
    suffix holds an older run of it, which must not be averaged in as another seed. Methods in
    `exclude` (run-directory names) are dropped; the rest are renamed for display (DISPLAY_NAMES)."""
    out = []
    exclude = set(exclude)
    for p in paths:
        try:
            r = json.loads(Path(p).read_text())
        except (OSError, json.JSONDecodeError):
            continue
        name, parent = str(r.get("method", {}).get("name", "")), Path(p).parent.name
        if name and parent != name and parent.startswith(name) and parent[len(name)] in "._":
            continue
        if name in exclude:
            continue
        r["method"]["name"] = DISPLAY_NAMES.get(name, name)
        r["_path"] = str(p)
        out.append(r)
    return out


def summarize(runs: Sequence[Dict], scorer: str = "tf") -> Dict[str, Dict]:
    by = defaultdict(list)
    for r in runs:
        # the split name keeps per-dataset splits (scalpel-mcf, scalpel-zsre, ...) in separate sections;
        # its seed suffix (scalpel-fact-dev-s3) is dropped so the seeds of one setting are averaged
        split = re.sub(r"-s\d+$", "", str(r.get("split") or "?"))
        setting = (f"{r.get('track') or 'fact'} · {split} · {r.get('model_id', '?')} · "
                   f"{r.get('n_requests', '?')} requests")
        by[(setting, r["method"]["name"])].append(r)
    table = {}
    for (track, name), rs in sorted(by.items()):
        agg_key = f"aggregates_{scorer}"
        row = {"track": track, "method": name, "family": rs[0]["method"]["family"], "seeds": sorted({r["seed"] for r in rs}),
               "valid": all(r.get("valid", True) for r in rs), "cheat": rs[0]["method"].get("is_reference_cheat", False),
               "runtime_components": rs[0]["method"].get("runtime_components", [])}
        for k in KEYS:
            row[k] = _ms([(r.get(agg_key) or {}).get(k) for r in rs])
        fam = defaultdict(list)
        for r in rs:
            for f, v in (r.get(f"families_{scorer}") or {}).items():
                fam[f].append(v["score"])
        row["families"] = {f: _ms(v) for f, v in sorted(fam.items())}
        strict = [((r.get("strict_tier") or {}).get("aggregates") or {}).get("SCALPEL") for r in rs]
        row["strict_SCALPEL"] = _ms(strict) if any(s is not None for s in strict) else (None, None)
        row["unlearn_seconds"] = _ms([r["cost"]["unlearn_seconds"] for r in rs])
        row["inference_overhead"] = _ms([r["cost"]["inference_overhead"] for r in rs])
        row["behaviour"] = rs[0].get("behaviour", {})
        # mean over seeds of the scalar routing diagnostics (was the first seed only)
        routing = [r["routing"] for r in rs if r.get("routing")]
        row["routing"] = ({k: _ms([d.get(k) for d in routing])[0] for k in routing[0] if not isinstance(routing[0][k], dict)}
                          if routing else None)
        row["ci"] = rs[0].get(f"ci_{scorer}") if len(rs) == 1 else None
        row["attacks"] = {k: _ms([(r.get("robustness") or {}).get("residual_by_attack", {}).get(k) for r in rs])
                          for k in ("quant_int8", "quant_int4", "relearn", "lens")}
        table[f"{track}:{name}"] = row
    return table


def _f(ms, digits=1):
    m, sd = ms if isinstance(ms, tuple) else (ms, None)
    if m is None:
        return "–"
    return f"{m:.{digits}f}" + (f" ± {sd:.{digits}f}" if sd else "")


def markdown(table: Dict[str, Dict], title: str = "LETHE-Bench leaderboard") -> str:
    lines = [f"# {title}", "",
             "All scores 0–100, higher is better. E = erasure on held-out forget families, L = locality, "
             "U = utility relative to base, R = robustness under attacks (lens, int8/int4, relearning). "
             "LETHE = harmonic mean of the available E, L, U, R (1-point floor on each). E_complete = share of requests "
             "whose every held-out forget probe is erased. *Strict* = weights only (runtime components removed). Rows marked † are reference cheats the benchmark must defeat.", ""]
    tracks = sorted({r["track"] for r in table.values()})
    if len(tracks) > 1:          # several splits / models: one SCALPEL matrix first
        methods = sorted({r["method"] for r in table.values()},
                         key=lambda m: -max((r["SCALPEL"][0] or -1) for r in table.values() if r["method"] == m))
        cell = {(r["track"], r["method"]): r for r in table.values()}
        lines += ["## LETHE score across splits", "", "| Method | " + " | ".join(tracks) + " |", "|---|" + "---|" * len(tracks)]
        for m in methods:
            cheat = any(r["cheat"] for r in table.values() if r["method"] == m)
            lines.append(f"| {m}{' †' if cheat else ''} | " + " | ".join(
                _f(cell[(t, m)]["SCALPEL"]) if (t, m) in cell else "–" for t in tracks) + " |")
        lines.append("")
    for track in tracks:
        rows = sorted([r for r in table.values() if r["track"] == track],
                      key=lambda r: -(r["SCALPEL"][0] if r["SCALPEL"][0] is not None else -1))
        lines += [f"## {track}", "",
                  "| Method | LETHE | E | E_mean | E_min | E_direct | E_complete | L | L_mean | L_min | U | R | Strict | seeds |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            name = r["method"] + (" †" if r["cheat"] else "") + ("" if r["valid"] else " (INVALID)")
            lines.append(f"| {name} | {_f(r['SCALPEL'])} | {_f(r['E'])} | {_f(r['E_mean'])} | {_f(r['E_min'])} | {_f(r['E_direct'])} | {_f(r['E_complete'])} | "
                         f"{_f(r['L'])} | {_f(r['L_mean'])} | {_f(r['L_min'])} | {_f(r['U'])} | {_f(r['R'])} | {_f(r['strict_SCALPEL'])} | "
                         f"{len(r['seeds'])} |")
        fams = sorted({f for r in rows for f in r["families"]})
        lines += ["", "### Per family (forget: residual %, lower is better; retain: retained %, higher is better)", "",
                  "| Method | " + " | ".join(f.replace("forget/", "F:").replace("retain/", "R:") for f in fams) + " |",
                  "|---|" + "---|" * len(fams)]
        for r in rows:
            lines.append(f"| {r['method']} | " + " | ".join(_f(r["families"].get(f, (None, None)), 0) for f in fams) + " |")
        legacy = [(r, _legacy(r)) for r in rows]
        if any(v[0] is not None for _, v in legacy):
            lines += ["", "### Legacy view vs LETHE-Bench (same run)", "",
                      "Legacy view = only the families the official MCF / zsRE metrics use: erasure on the request "
                      "and its paraphrases (Eff/Gen), retention on same-answer neighbours and unrelated questions (Spe). "
                      "A shortcut that looks perfect here and collapses under LETHE-Bench is what the legacy metrics reward.", "",
                      "| Method | legacy erasure | legacy locality | LETHE-Bench E | LETHE-Bench L | LETHE |", "|---|---|---|---|---|---|"]
            for r, (le, ll) in legacy:
                lines.append(f"| {r['method']}{' †' if r['cheat'] else ''} | {_f(le)} | {_f(ll)} | {_f(r['E'])} | "
                             f"{_f(r['L'])} | {_f(r['SCALPEL'])} |")
        lines += ["", "### Attacks (residual on forget facts after the attack, relative to the base model after the same attack; lower is better)", "",
                  "| Method | int8 | int4 | relearn (held-out half) | logit lens (upper half, chance-corrected) |",
                  "|---|---|---|---|---|"]
        for r in rows:
            a = r["attacks"]
            lines.append(f"| {r['method']} | {_f(a['quant_int8'])} | {_f(a['quant_int4'])} | {_f(a['relearn'])} | {_f(a['lens'])} |")
        beh = [r for r in rows if r["behaviour"]]
        if beh:
            lines += ["", "### Open-ended behaviour (greedy generation; probes the base model answers in generation)", "",
                      "| Method | forget: leak | forget: refusal | forget: other (wrong, empty, degenerate) | retain: correct | retain: false refusal |",
                      "|---|---|---|---|---|---|"]
            for r in beh:
                f, t = r["behaviour"].get("forget", {}), r["behaviour"].get("retain", {})
                lines.append(f"| {r['method']} | {_f(f.get('leak'))} | {_f(f.get('refusal'))} | "
                             f"{_f(f.get('hallucination_or_other'))} | {_f(t.get('correct'))} | {_f(t.get('false_refusal'))} |")
        routed = [r for r in rows if r.get("routing")]
        if routed:
            lines += ["", "### Detect-then-edit decomposition", "",
                      "| Method | route recall (forget) | residual from routing misses | residual when routed | false-fire rate (retain) |",
                      "|---|---|---|---|---|"]
            for r in routed:
                d = r["routing"]
                lines.append(f"| {r['method']} | {d['forget_route_recall']:.1f} | {d['forget_residual_from_routing']:.1f} | "
                             f"{d['forget_residual_from_actuation']:.1f} | {d['retain_false_fire_rate']:.1f} |")
        lines += ["", "### Cost", "", "| Method | unlearn seconds | inference overhead (× base) |", "|---|---|---|"]
        for r in rows:
            lines.append(f"| {r['method']} | {_f(r['unlearn_seconds'])} | {_f(r['inference_overhead'], 2)} |")
        lines.append("")
    return "\n".join(lines) + "\n"


LEGACY_FORGET = ("forget/direct", "forget/paraphrase")
LEGACY_RETAIN = ("retain/same_object", "retain/disjoint")


def _legacy(row: Dict):
    f = [100.0 - row["families"][k][0] for k in LEGACY_FORGET if k in row["families"] and row["families"][k][0] is not None]
    r = [row["families"][k][0] for k in LEGACY_RETAIN if k in row["families"] and row["families"][k][0] is not None]
    return (sum(f) / len(f) if f else None, sum(r) / len(r) if r else None)


def pareto_plot(table: Dict[str, Dict], path: Path) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return False
    rows = sorted([r for r in table.values() if r["E"][0] is not None and r["L"][0] is not None],
                  key=lambda r: -(r["SCALPEL"][0] or 0))
    fig, (ax, key) = plt.subplots(1, 2, figsize=(7.4, 4.6), gridspec_kw={"width_ratios": [3.2, 1.3]})
    placed = []
    for i, r in enumerate(rows, 1):
        x, y = r["L"][0], r["E"][0]
        ax.scatter(x, y, s=46, marker="x" if r["cheat"] else "o",
                   color="#9a9a9a" if r["cheat"] else "#2f6db5", zorder=3)
        k = sum(1 for (px, py) in placed if abs(px - x) < 8 and abs(py - y) < 8)
        placed.append((x, y))
        dy = (-12 - 9 * k) if y > 80 else (5 + 9 * k)        # stack labels away from the frame
        dx = (-7 if x > 85 else 6) + (0 if k % 2 == 0 else (-10 if x > 85 else 10))
        ax.annotate(str(i), (x, y), textcoords="offset points", xytext=(dx, dy), fontsize=8,
                    ha="right" if x > 85 else "left", color="#555555" if r["cheat"] else "#1f4f8f")
    key.axis("off")
    lines = [f"{i:>2}  {r['method']}{' †' if r['cheat'] else ''}" for i, r in enumerate(rows, 1)]
    key.text(0.0, 1.0, "\n".join(lines), va="top", ha="left", fontsize=8, family="monospace",
             transform=key.transAxes)
    key.text(0.0, 0.02, "† reference cheat\n(ordered by LETHE score)", va="bottom", fontsize=7, color="#666666",
             transform=key.transAxes)
    ax.set_xlabel("Locality L (retain families, %)")
    ax.set_ylabel("Erasure E (held-out forget families, %)")
    ax.set_xlim(-3, 103)
    ax.set_ylim(-3, 103)
    ax.grid(alpha=0.3)
    ax.set_title("Erasure vs locality (× = reference cheats; top right is the goal)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def write_report(run_paths: Sequence[Path], out: Path, title: str = "LETHE-Bench leaderboard",
                 exclude: Sequence[str] = ()) -> Dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(run_paths, exclude=exclude)
    table = summarize(runs, "tf")
    (out / "leaderboard.json").write_text(json.dumps(table, indent=1, default=str) + "\n")
    md = markdown(table, title)
    gen_table = summarize(runs, "gen")
    if any(r["SCALPEL"][0] is not None for r in gen_table.values()):
        # same layout under its own heading (the split drops the second H1 title)
        md += "\n# Open-ended (generation) scorer\n" + markdown(gen_table, title).split("\n", 1)[1]
    (out / "leaderboard.md").write_text(md)
    pareto_plot(table, out / "pareto.png")
    return table


def split_summary_md(split) -> str:
    """Per-family probe counts and knowledge-filter yields of one split."""
    m = split.manifest
    y = m.get("knowledge_filter_yield", {})
    lines = [f"# Split `{split.name}`", "",
             f"Model: `{split.model_id}` · partition: {split.partition} · seed {split.seed} · "
             f"{len(split.requests)} requests · {len(split.probes)} probes · fingerprint `{split.fingerprint()[:16]}`", ""]
    if "screened_facts" in m:
        lines.append(f"Forget facts: {m['forget_facts']} known out of {m['screened_facts']} screened "
                     f"({100 * m['forget_facts'] / max(1, m['screened_facts']):.1f}% of candidate facts are known by this model); "
                     f"{m.get('forget_facts_with_other_known_facts', 0)} have other known facts about the same subject.")
        lines.append("")
    lines += ["| Family | Probes | Candidates | Known by base | Used in aggregates |", "|---|---|---|---|---|"]
    for k in sorted(set(y) | set(m.get("counts", {}))):
        kept = m.get("counts", {}).get(k, 0)
        cand = y.get(k, {}).get("candidates", 0)
        lines.append(f"| {k} | {kept} | {cand} | {100 * y.get(k, {}).get('kept', 0) / max(1, cand):.0f}% | "
                     f"{'yes' if kept >= 5 else 'no (< 5)'} |")
    src = m.get("sources", {})
    if src.get("merge"):
        lines += ["", f"Fact pool after merging: {src['merge'].get('facts')} "
                      f"({src['merge'].get('subject_relation_conflicts_dropped', 0)} conflicting (subject, relation) "
                      f"pairs and {src['merge'].get('subject_object_overlap_dropped', 0)} self-answering facts dropped)."]
    return "\n".join(lines) + "\n"
