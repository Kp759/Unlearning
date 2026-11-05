#!/usr/bin/env python3
"""One table of the measurements behind the paper's Analysis section (theory_checks.slurm).

    python scripts/theory_summary.py            # -> outputs/theory_v1/SUMMARY.md (+ .json); copied to RESULTS_THEORY_CHECKS.md
"""
from __future__ import annotations

import json
from pathlib import Path

READ = {"Llama-3.2-3B": "outputs/router_diag_ours_control_v1/mcf/seed1/theory/read_layers.json",
        "Qwen3-4B": "outputs/layer_smoke_qwen3_4b/mcf/seed1/theory/read_layers.json"}
GEOM = {f"Llama-3.2-3B {ds}": f"outputs/l16_ours_v1/{ds}/seed1/ours/theory/geometry.json" for ds in ("mcf", "zsre", "mquake")}
GEOM["Qwen3-4B mcf"] = "outputs/layer_smoke_qwen3_4b_r26_w23/mcf/seed1/ours/theory/geometry.json"


def f(x, nd=3):
    return "--" if x is None else f"{x:.{nd}f}"


def main():
    lines, summary = ["# Measurements for the Analysis section (seed 1)", ""], {}
    for name, path in READ.items():
        if not Path(path).is_file():
            continue
        r = json.loads(Path(path).read_text())
        summary[f"read {name}"] = r
        lines += [f"## Gate-free heads per block, {name} MCF ({r['associations_used']}/{r['of']} associations)", "",
                  "| block | AUC same answer | AUC different answer | gap | distance ratio |", "|---|---|---|---|---|"]
        for layer, v in r["per_block"].items():
            lines.append(f"| {layer} | {f(v['auc_same_answer'])} | {f(v['auc_diff_answer'])} | "
                         f"{f(v['auc_diff_answer'] - v['auc_same_answer'])} | {f(v['distance_ratio'])} |")
        lines.append("")
    for name, path in GEOM.items():
        if not Path(path).is_file():
            continue
        g = json.loads(Path(path).read_text())
        summary[f"geometry {name}"] = {k: v for k, v in g.items() if k not in ("collateral",)}
        lines += [f"## Geometry, {name} (trained write block {g['trained_write_layer']})", ""]
        sel = g.get("selectivity")
        if sel:
            s = sel["summary"]
            lines += [f"Selectivity efficiency, median over {sel['associations']} associations "
                      f"(top-k mass retain {f(s['topk_mass_retain_median'])}, cos(g_views, g_heldout) "
                      f"{f(s['cos_g_views_g_heldout_median'])})", "",
                      "| alpha | K | g | trained | gradient | random | cos(trained, opt) | cos(gradient, opt) |",
                      "|---|---|---|---|---|---|---|---|"]
            for key, v in s.items():
                if key.count("/") != 2:
                    continue
                alpha, kname, gname = key.split("/")
                lines.append(f"| {alpha} | {kname} | {gname} | {f(v['trained']['median'])} | {f(v['gradient']['median'])} | "
                             f"{f(v['random']['median'])} | {f(v['cos_trained_opt']['median'])} | {f(v['cos_gradient_opt']['median'])} |")
            lines.append("")
        sw = g.get("write_sweep")
        if sw:
            alphas = [str(a) for a in g["alphas"]]
            lines += ["Write-block sweep (medians): S = g_V^T K^-1 g_V, held-out drop of the optimal edit, and of the gradient", "",
                      "| block | " + " | ".join(f"S a={a} | heldout@opt a={a} | heldout@grad a={a}" for a in alphas) + " |",
                      "|---|" + "---|" * (3 * len(alphas))]
            for layer, v in sw.items():
                lines.append(f"| {layer} | " + " | ".join(
                    f"{f(v[a]['S_views_median'], 1)} | {f(v[a]['heldout_drop_at_opt_median'], 2)} | {f(v[a]['heldout_drop_at_gradient_median'], 2)}"
                    for a in alphas) + " |")
            lines.append("")
        col = g.get("collateral")
        if col:
            summary[f"geometry {name}"]["collateral"] = {k: v for k, v in col.items() if k != "per_prompt"}
            lines += [f"Collateral: retain fire rate {f(col['fire_rate_retain_eval'], 5)} ({col['fired_retain_eval']}/"
                      f"{col['retain_eval_prompts']}), mean KL | fire {f(col['mean_kl_given_fire_retain_eval'])}, "
                      f"mean KL over retain {f(col['mean_kl_retain_eval'], 5)}; {col['fired_prompts']} fired prompts measured", "",
                      "| scale a | actual / predicted (median) | Spearman | mean actual KL | mean predicted KL |", "|---|---|---|---|---|"]
            for a, v in col["by_scale"].items():
                lines.append(f"| {a} | {f(v['median_actual_over_predicted'])} | {f(v['spearman_actual_vs_predicted'])} | "
                             f"{f(v['mean_actual_kl'], 4)} | {f(v['mean_predicted_kl'], 4)} |")
            lines.append("")
    out = Path("outputs/theory_v1")
    out.mkdir(parents=True, exist_ok=True)
    (out / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
