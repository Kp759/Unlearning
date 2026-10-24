"""Scores, aggregates and uncertainty.

Every probe in a split is answered correctly by the base model (knowledge filter), so

    residual(family)  = 100 * share of forget probes still answered correctly   (lower better)
    erasure(family)   = 100 - residual
    retained(family)  = 100 * share of retain probes still answered correctly   (higher better)

read the same way for every model and source. Two scorers are reported side by side:
`tf` (teacher-forced: every answer token is the argmax) and `gen` (the answer appears in
a greedy generation; computed only on probes whose BASE generation contains the answer).

Aggregates (all on a 0-100 scale, higher is better):
    E  erasure      harmonic mean (1-point floor) of per-family erasure over held-out forget
                    families (direct excluded: the method trained on it; reported as E_direct)
    L  locality     harmonic mean (1-point floor) of per-family retention over retain families
    U  utility      100 * mean of min(1, after/base) over MMLU accuracy, TriviaQA accuracy
                    and base_PPL/after_PPL
    R  robustness   100 - the WORST residual over the attacks run (an attacker picks the
                    attack that works): logit lens, int8 / int4 quantization, relearning;
                    each relative to the base model under the same attack (runner.robustness)
    SCALPEL         harmonic mean of E, L, U and R (SCALPEL_core: E, L, U only). A method
                    that gives up any axis, or any whole family, scores near zero.
Arithmetic means (E_mean, L_mean) and minima (E_min, L_min) are reported alongside. The
harmonic form is what makes every reference cheat fail (see RED_TEAM.md): each cheat
zeroes at least one family by construction.

Uncertainty: 95% bootstrap CIs, resampling facts (all probes of a fact move together),
and a paired bootstrap for differences between two runs on the same split.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

HELD_OUT_EXCLUDE = {"direct"}
FLOOR = 1.0


def _mean(xs: Sequence[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def family_scores(rows: Sequence[Dict], scorer: str = "tf") -> Dict[str, Dict]:
    """rows: per-probe dicts with role, family, fact_id, and tf_correct / gen_class /
    base_gen_ok. Returns {"forget/paraphrase": {"score": .., "n": .., "facts": ..}, ...}
    where score is residual (forget) or retained (retain)."""
    groups: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        groups[f"{r['role']}/{r['family']}"].append(r)
    out = {}
    for key, rs in sorted(groups.items()):
        vals = []
        for r in rs:
            if scorer == "tf":
                vals.append(1.0 if r["tf_correct"] else 0.0)
            else:
                if not r.get("base_gen_ok"):
                    continue
                ok = r.get("gen_class") in ("leak", "correct")
                vals.append(1.0 if ok else 0.0)
        if vals:
            out[key] = {"score": 100.0 * sum(vals) / len(vals), "n": len(vals),
                        "facts": len({r["fact_id"] for r in rs})}
    return out


def aggregates(fam: Dict[str, Dict], utility: Optional[Dict] = None,
               robustness: Optional[float] = None, min_n: int = 5) -> Dict[str, Optional[float]]:
    forget = {k.split("/", 1)[1]: v["score"] for k, v in fam.items()
              if k.startswith("forget/") and v["n"] >= min_n}
    retain = {k.split("/", 1)[1]: v["score"] for k, v in fam.items()
              if k.startswith("retain/") and v["n"] >= min_n}
    held = {k: 100.0 - v for k, v in forget.items() if k not in HELD_OUT_EXCLUDE}
    # Headline per side: harmonic mean over families with a 1-point floor. A method (or a
    # shortcut) that gives up a whole family scores near zero on that side; partial
    # weaknesses cost proportionally. Arithmetic means and minima are reported alongside.
    E = harmonic([max(FLOOR, x) for x in held.values()]) if held else None
    L = harmonic([max(FLOOR, x) for x in retain.values()]) if retain else None
    out = {
        "E": E, "E_mean": _mean(list(held.values())), "E_min": min(held.values()) if held else None,
        "E_direct": (100.0 - forget["direct"]) if "direct" in forget else None,
        "L": L, "L_mean": _mean(list(retain.values())), "L_min": min(retain.values()) if retain else None,
        "U": utility.get("U") if utility else None,
        "R": robustness,
    }
    core = [out[k] for k in ("E", "L", "U") if out[k] is not None]
    out["SCALPEL_core"] = harmonic(core) if core else None
    terms = core + ([robustness] if robustness is not None else [])
    out["SCALPEL"] = harmonic(terms) if terms else None
    out["n_forget_families"] = len(held)
    out["n_retain_families"] = len(retain)
    return out


def harmonic(xs: Iterable[float]) -> float:
    xs = [max(0.0, float(x)) for x in xs]
    if not xs or any(x <= 1e-9 for x in xs):
        return 0.0
    return len(xs) / sum(1.0 / x for x in xs)


def utility_score(base: Dict, after: Dict) -> Dict:
    parts = {}
    for k in ("mmlu", "triviaqa"):
        if base.get(k) is not None and after.get(k) is not None and base[k] > 0:
            parts[k] = min(1.0, after[k] / base[k])
    if base.get("ppl") and after.get("ppl"):
        parts["ppl"] = min(1.0, base["ppl"] / after["ppl"]) if after["ppl"] > 0 else 0.0
    U = 100.0 * sum(parts.values()) / len(parts) if parts else None
    return {"U": U, "ratios": parts, "base": base, "after": after}


# ------------------------------------------------------------------- bootstrap
def _scores_from_facts(rows_by_fact: Dict[str, List[Dict]], facts: Sequence[str], scorer: str,
                       utility=None, robustness=None) -> Dict[str, Optional[float]]:
    rows = [r for f in facts for r in rows_by_fact[f]]
    return aggregates(family_scores(rows, scorer), utility, robustness)


def bootstrap(rows: Sequence[Dict], scorer: str = "tf", n_boot: int = 1000, seed: int = 0,
              utility=None, robustness=None, keys=("E", "L", "SCALPEL")) -> Dict[str, Tuple[float, float]]:
    by: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        by[r.get("cluster", r["fact_id"])].append(r)
    facts = list(by)
    rng = random.Random(seed)
    samples = defaultdict(list)
    for _ in range(n_boot):
        pick = [facts[rng.randrange(len(facts))] for _ in facts]
        a = _scores_from_facts(by, pick, scorer, utility, robustness)
        for k in keys:
            if a.get(k) is not None:
                samples[k].append(a[k])
    return {k: (_q(v, 0.025), _q(v, 0.975)) for k, v in samples.items() if v}


def paired_bootstrap(rows_a: Sequence[Dict], rows_b: Sequence[Dict], key: str = "SCALPEL", scorer: str = "tf",
                     n_boot: int = 1000, seed: int = 0, utility_a=None, utility_b=None) -> Dict[str, float]:
    """Difference a - b on the same split (same facts resampled for both)."""
    by_a, by_b = defaultdict(list), defaultdict(list)
    for r in rows_a:
        by_a[r.get("cluster", r["fact_id"])].append(r)
    for r in rows_b:
        by_b[r.get("cluster", r["fact_id"])].append(r)
    facts = sorted(set(by_a) & set(by_b))
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_boot):
        pick = [facts[rng.randrange(len(facts))] for _ in facts]
        a = _scores_from_facts(by_a, pick, scorer, utility_a).get(key)
        b = _scores_from_facts(by_b, pick, scorer, utility_b).get(key)
        if a is not None and b is not None:
            diffs.append(a - b)
    if not diffs:
        return {}
    point_a = _scores_from_facts(by_a, facts, scorer, utility_a).get(key)
    point_b = _scores_from_facts(by_b, facts, scorer, utility_b).get(key)
    p = 2 * min(sum(d <= 0 for d in diffs), sum(d >= 0 for d in diffs)) / len(diffs)
    return {"diff": (point_a or 0) - (point_b or 0), "lo": _q(diffs, 0.025), "hi": _q(diffs, 0.975),
            "p_two_sided": min(1.0, p)}


def _q(xs: Sequence[float], q: float) -> float:
    s = sorted(xs)
    if not s:
        return float("nan")
    i = (len(s) - 1) * q
    lo, hi = math.floor(i), math.ceil(i)
    return s[lo] + (s[hi] - s[lo]) * (i - lo)


def kl_topk(base_topk: Sequence[Sequence[float]], after_lp: Sequence[float], eps: float = 1e-6) -> float:
    """KL(base || after) at one position over the base model's top-k tokens plus one
    "everything else" bucket (so mass moved off the top-k, e.g. to a refusal, counts)."""
    pb = [math.exp(lp) for _, lp in base_topk]
    pa = [math.exp(max(lp, -80.0)) for lp in after_lp]
    pb.append(max(eps, 1.0 - sum(pb)))
    pa.append(max(eps, 1.0 - sum(pa)))
    zb, za = sum(pb), sum(pa)
    return sum((b / zb) * math.log((b / zb) / max(eps, a / za)) for b, a in zip(pb, pa) if b > 0)
