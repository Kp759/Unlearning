#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List


def load(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def ms(values: Iterable[float]) -> Dict[str, float | int]:
    xs = [float(x) for x in values]
    if not xs:
        return {"n": 0, "mean": float("nan"), "sd": float("nan")}
    return {
        "n": len(xs),
        "mean": float(statistics.mean(xs)),
        "sd": float(statistics.stdev(xs)) if len(xs) > 1 else 0.0,
    }


def metric(rows: List[Dict[str, Any]], *keys: str) -> List[float]:
    out: List[float] = []
    for row in rows:
        x: Any = row
        for key in keys:
            x = x[key]
        if x is not None:
            out.append(float(x))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--seeds", default="1,2,3,4,5,6,7,8,9,10")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    root = Path(a.root).resolve()
    seeds = [int(x) for x in a.seeds.split(",") if x.strip()]
    per_seed: List[Dict[str, Any]] = []

    for seed in seeds:
        sr = root / f"seed{seed}"
        s1 = load(sr / "official_eval_stage1_posthoc.json")
        final = load(sr / "official_eval_locked.json")
        cfg = load(sr / "stage1_collision_safe" / "config_used.json")
        repair = load(sr / "stage2_sensitive_row_repair" / "summary.json")
        mh = load(sr / "multihop_eval_final.json")

        row = {
            "seed": seed,
            "forget_instances": int(cfg["forget_instances"]),
            "forget_atomic_facts": int(cfg["forget_atomic_facts"]),
            "direct_sensitive_token_cases": int(cfg["direct_sensitive_token_cases"]),
            "sensitive_rows": int(cfg["sensitive_rows"]),
            "ordinary_context_hidden_states": int(cfg["ordinary_context_hidden_states"]),
            "stage1_direct_correct_before": int(cfg["correct_before"]),
            "stage1_direct_correct_after": int(cfg["correct_after_restoration"]),
            "stage1_sensitive_delta_fro_norm": float(cfg["sensitive_delta_fro_norm"]),
            "stage1": {
                "F_Eff": float(s1["forget"]["Eff"]),
                "F_AtomicGen": float(s1["forget"]["AtomicGen"]),
                "R_Eff": float(s1["retain"]["Eff"]),
                "R_AtomicGen": float(s1["retain"]["AtomicGen"]),
                "PPL": float(s1["forget_PPL"]),
            },
            "final": {
                "F_Eff": float(final["forget"]["Eff"]),
                "F_AtomicGen": float(final["forget"]["AtomicGen"]),
                "R_Eff": float(final["retain"]["Eff"]),
                "R_AtomicGen": float(final["retain"]["AtomicGen"]),
                "PPL": float(final["forget_PPL"]),
            },
            "stage2": {
                "active_correct_tokens_before": int(repair.get("stage2_active_correct_tokens_before", 0)),
                "selected_rows": int(repair.get("stage2_rows", 0)),
                "full_delta_norm": float(repair.get("stage2_full_delta_norm", 0.0)),
                "selected_scale": float(repair.get("stage2_selected_scale", 0.0)),
            },
            "multihop": {
                "standard_MHLeak_exact_any": float(mh["results"]["standard"]["MHLeak_exact_any"]),
                "standard_MHLeak_contains_any": float(mh["results"]["standard"]["MHLeak_contains_any"]),
                "cot_MHLeak_exact_any": float(mh["results"]["cot"]["MHLeak_exact_any"]),
                "cot_MHLeak_contains_any": float(mh["results"]["cot"]["MHLeak_contains_any"]),
            },
        }
        per_seed.append(row)

    summary = {
        "schema_version": 1,
        "method": "SURE MQuAKE collision-safe contextual GA/GD plus sparse active LM-head repair",
        "root": str(root),
        "seeds": seeds,
        "n": len(per_seed),
        "sample_sd": True,
        "stage1": {k: ms(metric(per_seed, "stage1", k)) for k in ("F_Eff", "F_AtomicGen", "R_Eff", "R_AtomicGen", "PPL")},
        "final": {k: ms(metric(per_seed, "final", k)) for k in ("F_Eff", "F_AtomicGen", "R_Eff", "R_AtomicGen", "PPL")},
        "stage2": {
            k: ms(metric(per_seed, "stage2", k))
            for k in ("active_correct_tokens_before", "selected_rows", "full_delta_norm", "selected_scale")
        },
        "stage1_diagnostics": {
            "forget_atomic_facts": ms(metric(per_seed, "forget_atomic_facts")),
            "direct_sensitive_token_cases": ms(metric(per_seed, "direct_sensitive_token_cases")),
            "sensitive_rows": ms(metric(per_seed, "sensitive_rows")),
            "ordinary_context_hidden_states": ms(metric(per_seed, "ordinary_context_hidden_states")),
            "direct_correct_before": ms(metric(per_seed, "stage1_direct_correct_before")),
            "direct_correct_after": ms(metric(per_seed, "stage1_direct_correct_after")),
            "sensitive_delta_fro_norm": ms(metric(per_seed, "stage1_sensitive_delta_fro_norm")),
        },
        "multihop": {
            k: ms(metric(per_seed, "multihop", k))
            for k in (
                "standard_MHLeak_exact_any",
                "standard_MHLeak_contains_any",
                "cot_MHLeak_exact_any",
                "cot_MHLeak_contains_any",
            )
        },
        "exact_zero_final_F_Eff_seeds": [r["seed"] for r in per_seed if r["final"]["F_Eff"] == 0.0],
        "per_seed": per_seed,
    }

    out = Path(a.out).resolve() if a.out else root / "aggregate_seeds1_10.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("===== MQuAKE COLLISION-SAFE 10-SEED AGGREGATE =====")
    for block in ("stage1", "final"):
        print(block.upper())
        for k, v in summary[block].items():
            print(f"  {k:12s} {v['mean']:.6f} +/- {v['sd']:.6f}")
    print("STAGE2")
    for k, v in summary["stage2"].items():
        print(f"  {k:28s} {v['mean']:.6f} +/- {v['sd']:.6f}")
    print("MULTIHOP")
    for k, v in summary["multihop"].items():
        print(f"  {k:34s} {v['mean']:.6f} +/- {v['sd']:.6f}")
    print("exact-zero final F-Eff:", len(summary["exact_zero_final_F_Eff_seeds"]), "/", len(per_seed), summary["exact_zero_final_F_Eff_seeds"])
    print("wrote:", out)


if __name__ == "__main__":
    main()
