#!/usr/bin/env python3
"""Write-layer sweep: one table per question, one row per write layer (CPU, seconds).

Arms are E_L<w>_query_logit_<TAG>S<L> (same router, same recipe; only the Δe write layer
differs), produced by WRITE_LAYERS="..." bash scripts/submit_e2e_v21_seed1.sh.

    python scripts/write_layer_sweep_summary.py --tag v22 --subject-layer 6 --layers 12 16 19

Writes outputs/mcf_write_layer_sweep_<TAG>S<L>_seed1/summary.md and prints it.
Because routing is a separate unedited pass, every arm fires on exactly the same prompts:
differences between rows come from the write layer alone.
"""
import argparse
import json
from pathlib import Path
import re

EARLY = Path("outputs/mcf_early_write_seed1")
HELD = Path("outputs/mcf_heldout_generation_seed1")
COLL = Path("outputs/mcf_subject_collateral_seed1")


def load(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else None


def pct(x, d=1):
    return "–" if x is None else f"{100 * x:.{d}f}%"


def num(x, d=2):
    return "–" if x is None else f"{x:.{d}f}"


def word_in(answer, text):
    return re.search(r"(?<!\w)" + re.escape(answer.strip()) + r"(?!\w)", text, re.IGNORECASE) is not None


def held_rates(data, s):
    """substring and whole-word answer rate on set s, prompts without the answer in them."""
    if data is None:
        return None, None
    part = [r for r in data["examples"] if r["set"] == s and not r["answer_in_prompt"]]
    if not part:
        return None, None
    return (sum(r["answer_in_continuation"] for r in part) / len(part),
            sum(word_in(r["answer"], r["continuation"]) for r in part) / len(part))


def table(header, rows):
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + \
           ["| " + " | ".join(r) + " |" for r in rows]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tag", default="v22")
    p.add_argument("--subject-layer", type=int, default=6)
    p.add_argument("--layers", type=int, nargs="+", default=[12, 16, 19])
    a = p.parse_args()
    suffix = f"_query_logit_{a.tag}S{a.subject_layer}"
    out_dir = Path(f"outputs/mcf_write_layer_sweep_{a.tag}S{a.subject_layer}_seed1")
    lens_dir = out_dir / "lens"
    arms = [(w, f"E_L{w}{suffix}") for w in a.layers]
    lines = [f"# Δe write-layer sweep, router {a.tag} (subject layer {a.subject_layer}), MCF seed 1\n",
             "Same router and recipe (question-side write, output + logit-lens feedback at L24); "
             "only the write layer changes.\n"]

    # 1) training
    rows = []
    for w, name in arms:
        rep = load(EARLY / name / "linear_global" / "training_report.json")
        if rep is None:
            rows.append([f"L{w}", "(not trained)"] + ["–"] * 4)
            continue
        rounds = rep["rounds"]
        trained = [r for r in rounds if "train" in r]
        last = trained[-1]["train"] if trained else {}
        rel = rep["row_norm_median"] / max(rep["boundary_norm_median"], 1e-12)
        rows.append([f"L{w}", pct(rounds[0]["onpolicy_leak"]), pct(rounds[-1]["onpolicy_leak"]),
                     num(last.get("keep"), 3), num(last.get("out"), 3), num(rel, 3)])
    lines += ["\n## 1. Training (training views)\n"] + table(
        ["write layer", "on-policy leak, round 0 (base)", "on-policy leak, final ↓",
         "fluency KL to base, last round ↓", "output loss, last round ↓", "‖Δe‖ / ‖h‖ (median) ↓"], rows)

    # 2) official MCF
    rows = []
    for w, name in arms:
        r = load(EARLY / name / "linear_global" / "official_mcf_eval.json")
        if r is None:
            rows.append([f"L{w}", "(missing)"] + ["–"] * 4)
            continue
        f, t = r["forget"], r["retain"]
        rows.append([f"L{w}", num(float(f["Eff"]), 3), num(float(f["Gen"]), 3), num(float(f["Spe"]), 2),
                     num(float(t["Spe"]), 2), num(r.get("forget_PPL"), 2)])
    lines += ["\n## 2. Official MCF (answer probability)\n"] + table(
        ["write layer", "forget Eff ↓", "forget Gen ↓", "forget Spe ↑", "retain Spe ↑", "PPL"], rows)

    # 3) held-out generation
    rows = []
    for label, data in [("base", load(HELD / "base.json"))] + [(f"L{w}", load(HELD / f"{n}.json")) for w, n in arms]:
        if data is None:
            rows.append([label, "(missing)"] + ["–"] * 6)
            continue
        s = data["summary"]
        cells = []
        for st in ("rewrite", "paraphrase", "generation", "neighborhood"):
            sub, word = held_rates(data, st)
            cells.append("–" if sub is None else f"{pct(sub)} / {pct(word)}")
        rows.append([label] + cells + [pct(s["paraphrase"]["degenerate"]), pct(s["neighborhood"]["degenerate"]),
                                       pct(s["paraphrase"]["routed_to_own_fact"])])
    lines += ["\n## 3. Free generation, 24 tokens (substring / whole word)\n"] + table(
        ["write layer", "rewrite leak ↓", "paraphrase (held out) leak ↓", "generation leak ↓",
         "neighbourhood answer kept ↑", "paraphrase degenerate ↓", "neighbourhood degenerate ↓",
         "paraphrase routed to own fact"], rows)

    # 4) same-subject collateral (forced edit: what the edit does to the subject's other facts)
    rows = []
    for w, name in arms:
        r = load(COLL / f"{name}.json")
        if r is None:
            rows.append([f"L{w}", "(missing)", "–", "–", "–"])
            continue
        f = r["summary"]["forced"]
        rows.append([f"L{w}", str(f["prompts"]), pct(f["same_first5"], 0), pct(f["same_answer"], 0),
                     num(f["dNLL_mean"], 2)])
    lines += ["\n## 4. Same subject, other relations, edit forced on (collateral of the vector itself)\n",
              "Router fire rate is identical across write layers (routing is the unedited pass), so only "
              "the forced columns differ.\n"] + table(
        ["write layer", "n", "same first 5 ↑", "answer kept ↑", "ΔNLL of base continuation ↓"], rows)

    # 5) lens: is the answer readable inside the model, layer by layer
    for key, title in (("heldout_paraphrase/routed", "held-out paraphrases, routed"),
                       ("training_view/routed", "training views, routed")):
        res = [("base", load(lens_dir / "base.json"))] + [(f"L{w}", load(lens_dir / f"{n}.json")) for w, n in arms]
        res = [(lab, d["summary"].get(key)) for lab, d in res if d is not None and d["summary"].get(key)]
        if not res:
            lines.append(f"\n## 5. Lens ({title}): missing — run the sweep lens job")
            continue
        layers = sorted(int(l) for l in res[0][1]["logit_lens_topk_any_position"])
        jl = sorted(int(l) for l in res[0][1]["jlens_max_z_median"])
        rows = []
        for lab, s in res:
            top = s["logit_lens_topk_any_position"]
            clean = [l for l in layers if all(top[str(m)] <= 0.02 for m in layers if m >= l)]
            rows.append([lab, str(s["prompts"]), pct(s["leak"])]
                        + [pct(top[str(l)], 0) for l in layers]
                        + [num(s["jlens_max_z_median"][str(l)]) for l in jl]
                        + [f"L{clean[0]}" if clean else "never"])
        lines += [f"\n## 5. Answer readable inside the model ({title})\n",
                  "logit lens: answer in top-10 at any generated position; J-lens: median max z "
                  "(0 = unrelated-prompt level). Last column: first layer from which the answer stays "
                  "≤2% top-10 at every later layer.\n"] + table(
            ["run", "n", "leak"] + [f"top-10 L{l}" for l in layers] + [f"J z L{l}" for l in jl]
            + ["clean from"], rows)

    text = "\n".join(lines) + "\n"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
