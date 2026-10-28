"""Self red-team: the checks a benchmark reviewer (e.g. Snorkel's red team) would run.

    validate_split(split)     integrity of one split (no label conflicts, nothing evaluation-only
                              leaks into the method's view, family sizes, partition hygiene)
    cheat_gate(results)       every reference cheat must score below a ceiling; if one does
                              not, the split rewards a shortcut and must not be released
"""
from __future__ import annotations

from collections import Counter
import json
from typing import Dict, List, Sequence

from .schema import CANARY, TRAINING_VISIBLE_FAMILIES, Split, normalize_text

CHEAT_CEILING = 35.0      # max(SCALPEL, SCALPEL_core) a reference cheat must stay below
MIN_FAMILY = 5
# Which families defeat which shortcut. A split must keep at least one family per cheat
# with >= MIN_FAMILY probes, or that cheat is only defeated by luck.
DEFEATING_FAMILIES = {
    "prompt_cache_idk": ["forget/paraphrase", "forget/question", "forget/crosslingual"],
    "subject_blocklist_idk": ["retain/same_subject", "retain/same_sentence", "retain/subject_mentioned"],
    "keyword_router_idk": ["forget/indirect", "forget/multihop", "forget/crosslingual", "forget/surface"],
    "answer_ban": ["retain/same_object"],
    "idk_all": ["retain/disjoint", "retain/same_relation"],
    "noop": ["forget/paraphrase", "forget/question"],
}


def _text(p) -> str:
    if p.messages:
        return normalize_text(" || ".join(m["content"] for m in p.messages))
    return normalize_text(p.prompt)


def validate_split(split: Split) -> Dict:
    checks: List[Dict] = []

    def check(name, ok, detail=""):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    view = json.dumps(split.method_view(), ensure_ascii=False)
    req_prompts = {normalize_text(r.prompt) for r in split.requests if r.prompt}
    # 1. the method's view contains nothing evaluation-only
    leaked = [p.probe_id for p in split.probes if p.role == "forget" and p.family not in TRAINING_VISIBLE_FAMILIES
              and _text(p) in req_prompts]
    check("held-out forget probes differ from every request prompt", not leaked, f"{len(leaked)} identical")
    retain_in_view = [p.probe_id for p in split.retain_probes() if p.prompt and len(p.prompt) > 25 and p.prompt in view]
    check("no retain probe text appears in the method's view", not retain_in_view, f"{len(retain_in_view)} found")
    # 2. label conflicts
    ftexts = {_text(p) for p in split.forget_probes()}
    conflict = [p.probe_id for p in split.retain_probes() if _text(p) in ftexts]
    check("no prompt is both a forget and a retain probe", not conflict, f"{len(conflict)} conflicts")
    fsr = {(normalize_text(p.subject), p.relation) for p in split.forget_probes() if p.family == "direct"}
    sr_conf = [p.probe_id for p in split.retain_probes()
               if p.family in ("same_subject", "name_collision", "disjoint", "subject_mentioned")
               and (normalize_text(p.subject), p.relation) in fsr]
    check("no retain probe asks a forgotten (subject, relation)", not sr_conf, f"{len(sr_conf)} found")
    # 3. every request has its direct probe; one request per subject
    direct = Counter(p.request_id for p in split.forget_probes() if p.family == "direct")
    missing = [r.request_id for r in split.requests if r.kind == "fact" and direct.get(r.request_id, 0) == 0]
    check("every fact request has a known direct probe", not missing, f"{len(missing)} missing")
    subj = Counter(normalize_text(r.subject) for r in split.requests)
    check("one request per subject", max(subj.values() or [0]) <= 1 or split.manifest.get("track") == "entity",
          f"max {max(subj.values() or [0])}")
    # 4. forget probes point at real requests
    rids = {r.request_id for r in split.requests}
    dangling = [p.probe_id for p in split.forget_probes() if p.request_id not in rids]
    check("every forget probe names an existing request", not dangling, f"{len(dangling)} dangling")
    # 5. unique ids, canary, family sizes
    ids = Counter(p.probe_id for p in split.probes)
    check("probe ids are unique", max(ids.values()) == 1, f"{sum(v > 1 for v in ids.values())} duplicated")
    check("canary present", split.manifest.get("canary") == CANARY)
    fam = Counter(f"{p.role}/{p.family}" for p in split.probes)
    small = {k: v for k, v in fam.items() if v < MIN_FAMILY}
    check(f"families with < {MIN_FAMILY} probes are excluded from aggregates (reported)", True,
          json.dumps(small) if small else "none")
    # 6. each reference cheat has a family that defeats it
    for cheat, fams in DEFEATING_FAMILIES.items():
        have = [f for f in fams if fam.get(f, 0) >= MIN_FAMILY]
        check(f"a family that defeats {cheat} has >= {MIN_FAMILY} probes", bool(have),
              ", ".join(f"{f}={fam.get(f, 0)}" for f in fams))
    # 7. fingerprint matches content
    fp = split.manifest.get("fingerprint")
    check("manifest fingerprint matches the split content", fp is None or fp == split.fingerprint())
    return {"ok": all(c["ok"] for c in checks), "checks": checks, "family_counts": dict(sorted(fam.items()))}


def cheat_gate(results: Sequence[Dict], ceiling: float = CHEAT_CEILING) -> Dict:
    rows = []
    for r in results:
        if not r["method"].get("is_reference_cheat"):
            continue
        agg = r.get("aggregates_tf") or {}
        s, c = agg.get("SCALPEL"), agg.get("SCALPEL_core")
        worst = max(x for x in (s, c, 0.0) if x is not None)
        rows.append({"method": r["method"]["name"], "SCALPEL": s, "SCALPEL_core": c, "E": agg.get("E"),
                     "L": agg.get("L"), "U": agg.get("U"), "R": agg.get("R"), "passes": worst < ceiling})
    return {"ceiling": ceiling, "ok": all(x["passes"] for x in rows) and bool(rows), "rows": rows}


def markdown(validation: Dict, gate: Dict, split_name: str) -> str:
    lines = [f"# Red-team report: {split_name}", "", "## Split integrity", "", "| Check | OK | Detail |", "|---|---|---|"]
    for c in validation["checks"]:
        lines.append(f"| {c['check']} | {'yes' if c['ok'] else '**NO**'} | {c['detail']} |")
    lines += ["", f"## Reference cheats (must score below {gate['ceiling']:.0f} with AND without the attack axis)", "",
              "| Cheat | LETHE | LETHE_core (no attacks) | E | L | U | R | Defeated |", "|---|---|---|---|---|---|---|---|"]
    f = lambda x: "–" if x is None else f"{x:.1f}"
    for r in gate["rows"]:
        lines.append(f"| {r['method']} | {f(r['SCALPEL'])} | {f(r['SCALPEL_core'])} | {f(r['E'])} | {f(r['L'])} | "
                     f"{f(r['U'])} | {f(r['R'])} | {'yes' if r['passes'] else '**NO**'} |")
    lines += ["", f"**Overall: {'PASS' if validation['ok'] and gate['ok'] else 'FAIL'}**", ""]
    return "\n".join(lines)
