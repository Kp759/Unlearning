#!/usr/bin/env python3
"""One explained results page for the router diagnostics (router_diag_staged.slurm outputs).

    python scripts/router_diag_explain.py                         # outputs/router_diag_ours_v1
    python scripts/router_diag_explain.py --root outputs/router_diag_v1

Reads, per dataset, <root>/<ds>/seed<S>/{manifest.json, report/metrics.json,
layer_sweep/sweep.json, layer_sweep/penultimate_vs_L19.json} and writes <root>/RESULTS_EXPLAINED.md:
for each diagnostic task, what it measures, why, the numbers for all datasets, the plots, and an
automatic attribution of every routing error to the SUBJECT GATE or to the CLASSIFIER HEADS:

  gate        false negatives where no head was eligible: subject_absent_from_prompt (the
              subject is not named: pronoun, description, other language), and
              subject_in_text_not_token_matched (named, but in a form the match does not accept)
  classifier  false negatives where the right head was eligible but scored below the cutoff
              (below_threshold) or tied with another head (ambiguous_top2); false positives
              where an eligible head fired on a must-not-fire prompt
              (eligible_head_above_threshold); misroutes to another row of the same subject
  label       false positives on prompts identical to a forget prompt (identical_to_forget_prompt):
              a dataset conflict, not a router error
Nothing is recomputed; plots are linked relative to the page.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DATASETS = ("mcf", "zsre", "mquake", "rwku")
NAMES = {"mcf": "MCF", "zsre": "zsRE", "mquake": "MQuAKE", "rwku": "RWKU"}
GATE = {"false_negative:subject_absent_from_prompt", "false_negative:subject_in_text_not_token_matched",
        "false_negative:relation_gate"}
HEAD = {"false_negative:below_threshold", "false_negative:ambiguous_top2",
        "false_positive:eligible_head_above_threshold", "misroute:same_subject_other_row",
        "misroute:other_subject_row"}
LABEL = {"false_positive:identical_to_forget_prompt"}


def _load(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else None


def _f(x, d=3):
    return "–" if x is None else f"{x:.{d}f}"


def _pct(x):
    return "–" if x is None else f"{100 * x:.1f}%"


def attribution(reasons):
    """Split error counts into gate / classifier / label conflict."""
    out = {"gate": 0, "classifier": 0, "label": 0, "other": 0}
    for key, n in (reasons or {}).items():
        side = "gate" if key in GATE else "classifier" if key in HEAD else "label" if key in LABEL else "other"
        out[side] += int(n)
    return out


def sweep_rows(sweep):
    entries = sweep.get("sweep", []) if isinstance(sweep, dict) else (sweep or [])
    rows = []
    for e in entries:
        if e.get("rule") != "deployed" or len(e.get("layers", [])) != 1:
            continue
        r = e["by_eligibility"][e.get("deployed_eligibility", "exact")]
        m = r["eval"]
        rows.append((int(e["layers"][0]), m.get("f1"), m.get("fpr"), m.get("fnr"), r.get("null_fpr")))
    return sorted(rows)


def collect(root, seed):
    data = {}
    for ds in DATASETS:
        d = Path(root) / ds / f"seed{seed}"
        m = _load(d / "report" / "metrics.json")
        if m is None:
            continue
        data[ds] = {"dir": d, "manifest": _load(d / "manifest.json") or {}, "metrics": m,
                    "sweep": sweep_rows(_load(d / "layer_sweep" / "sweep.json") or {}),
                    "penultimate": _load(d / "layer_sweep" / "penultimate_vs_L19.json") or {},
                    "train_umap": _load(d / "train_umap" / "separability.json")}
    return data


def build(root, seed):
    data = collect(root, seed)
    if not data:
        raise SystemExit(f"no report/metrics.json under {root}/<ds>/seed{seed}/")
    rel = lambda ds, f: f"{ds}/seed{seed}/{f}"                       # noqa: E731
    any_m = next(iter(data.values()))["manifest"]
    L = any_m.get("router_layer", 19)
    md = [f"# Router diagnostics, explained ({Path(root).name}, seed {seed})", "",
          f"The router of SURE: one logistic head per protected fact, reading the last prompt token at block {L}, "
          "behind a subject gate (a head may fire only if its fact's subject is named in the prompt). "
          "These diagnostics test the router alone, on every prompt the official evaluators route, "
          "plus prompts with no association, surface-perturbed prompts, and the router's own training prompts. "
          f"Router dirs: " + ", ".join(f"`{Path(v['manifest'].get('router_dir', '?')).parent.parent.parent.name}`"
                                        for v in data.values()) +
          f"; recipe `{any_m.get('router_recipe')}`, subject match `{any_m.get('subject_match')}`.", ""]

    # ------------------------------------------------------------ headline: gate or classifier?
    md += ["## Where the errors come from: gate or classifier", "",
           "Every routing error is labelled with its cause. If no head was even eligible, the **subject gate** "
           "blocked it (the subject is not named, or named in a form the match rejects). If the right head was "
           "eligible but did not fire, fired on the wrong prompt, or picked another row, the **classifier heads** "
           "are responsible. A false positive on a prompt identical to a forget prompt is a **label conflict** in "
           "the dataset (the same text is both forget and retain), not a router error.", "",
           "| dataset | probes | errors | gate | classifier | label conflict | main cause |",
           "|---|---|---|---|---|---|---|"]
    verdicts = {}
    for ds, v in data.items():
        reasons = v["metrics"].get("reasons", {})
        a = attribution(reasons)
        total = sum(a.values())
        n = v["metrics"]["eval"]["by_split"]["all"]["n"]
        main = max(("gate", "classifier", "label"), key=lambda k: a[k]) if total else "none"
        verdicts[ds] = (a, reasons, main)
        md.append(f"| {NAMES[ds]} | {n} | {total} | {a['gate']} | {a['classifier']} | {a['label']} | **{main}** |")
    md += ["", "Detailed reasons per dataset:", ""]
    for ds, (a, reasons, main) in verdicts.items():
        md.append(f"- **{NAMES[ds]}**: " + (", ".join(f"`{k}` {v}" for k, v in sorted(reasons.items(), key=lambda x: -x[1]))
                                             or "no errors"))
    md += ["", "Reading it: `below_threshold` means the right head saw the prompt but was not confident enough "
           "(the fix is more or better positives, e.g. rewrites for facts that got few); `ambiguous_top2` means two "
           "heads of the same subject tied (more same-subject negatives); `eligible_head_above_threshold` means a "
           "head fired on a prompt it should not (hard negatives); `subject_absent_from_prompt` can only be fixed in "
           "the gate (aliases, or a learned gate), never by the heads; `relation_gate` means the router's relation gate "
           "(scripts/fit_relation_gate.py) judged that the prompt asks for another relation.", ""]

    # ------------------------------------------------------------ 1. per-example errors
    md += ["## 1. Per-example errors (scores against the decision line)", "",
           "**What.** For every evaluated prompt, the probability of the head that should fire (forget prompts) or "
           "of the highest head (retain prompts), plotted against the firing cutoff of 0.5. Points on the wrong side "
           "of the line are the errors; their labels are counted above. Each dataset's `report/summary.md` lists the "
           "worst examples with their text.", "",
           "**Why it matters.** A model of errors at the level of single prompts shows whether failures are "
           "systematic (one phrasing type, one fact with few positives) or scattered.", "",
           "| dataset | recall | precision | F1 | FPR | FNR | correct row | misroutes |",
           "|---|---|---|---|---|---|---|---|"]
    for ds, v in data.items():
        a = v["metrics"]["eval"]["by_split"]["all"]
        md.append(f"| {NAMES[ds]} | {_f(a['recall'])} | {_f(a['precision'])} | {_f(a['f1'])} | {_f(a['fpr'])} | "
                  f"{_f(a['fnr'])} | {_f(a['correct_row_rate'])} | {a['misroutes']} |")
    md.append("")
    for ds in data:
        md.append(f"![{NAMES[ds]} scores]({rel(ds, 'report/scores_eval.png')})")
    md.append("")

    # ------------------------------------------------------------ 2. no association
    md += ["## 2. Prompts with no protected association", "",
           "**What.** Prompts about unprotected subjects, generic questions and chat turns: what a deployed router "
           "sees most of the time. Any firing is a false positive, and by construction it is the only way SURE can "
           "change the model's answer to such a prompt.", "",
           "**Theory.** Off the routed set the model is exactly the base model, so locality loss is bounded by this "
           "false-fire rate. The gate makes it near zero for prompts that do not contain a protected name; what "
           "remains are prompts that mention a protected subject for an unrelated reason.", "",
           "| dataset | null prompts | fired | deployment FPR [95% CI] |", "|---|---|---|---|"]
    for ds, v in data.items():
        nl = v["metrics"].get("null") or {}
        d = nl.get("deployment_fpr") or {}
        md.append(f"| {NAMES[ds]} | {d.get('n', '–')} | {d.get('k', '–')} | {_pct(d.get('rate'))} "
                  f"[{_pct(d.get('low'))}, {_pct(d.get('high'))}] |")
    md += ["", "Per-group breakdown (which kind of null prompt fires) is in each `report/summary.md`, section 4.", ""]
    for ds in data:
        md.append(f"![{NAMES[ds]} null]({rel(ds, 'report/scores_null.png')})")
    md.append("")

    # ------------------------------------------------------------ 3. UMAP
    md += ["## 3. Positives vs negatives at blocks 16 and 19 (UMAP)", "",
           "**What.** Ten prompts that should fire and ten hard negatives (must-not-fire prompts that contain a "
           "protected subject), embedded from the block-16 and block-19 residual stream, coloured by class. The "
           "numbers come from the full-dimensional states, not the 2-D picture: silhouette (how clustered the two "
           "classes are) and leave-one-out accuracy of a logistic and a 1-nearest-neighbour classifier.", "",
           "**Theory.** The classifier is linear, so what matters is linear separability in full dimension; the "
           "2-D projection is an illustration. If block 19 separates better than block 16, reading late is "
           "justified at the level of the representation, not only by the end metrics.", "",
           "| dataset | layer | silhouette (full dim) | LOO logistic | LOO 1-NN |", "|---|---|---|---|---|"]
    for ds, v in data.items():
        for layer, s in (v["metrics"].get("separability") or {}).items():
            loo = s.get("loo") or {}
            md.append(f"| {NAMES[ds]} | L{layer} | {_f(s.get('silhouette_cosine_full_dim'))} | "
                      f"{_f(loo.get('logistic'))} | {_f(loo.get('nearest_neighbour'))} |")
    md += ["", "20 points per panel: one leave-one-out error moves accuracy by 0.05.", ""]
    for ds in data:
        md.append(f"![{NAMES[ds]} umap]({rel(ds, 'report/umap_L16_L19.png')})")
    md.append("")
    trained = {ds: v["train_umap"] for ds, v in data.items() if v.get("train_umap")}
    if trained:
        md += ["### 3b. Each head's own training prompts (scripts/router_train_umap.py)", "",
               "**What.** The prompts the router is fit on (`linear_router_dataset.json`: about 20 positives and "
               "20 negatives per fact), per fact: its positives against its own negatives. Numbers are averaged "
               "over facts, full dimension; the head columns are the deployed router's own decisions on these "
               "prompts at its read layer.", "",
               "| dataset | facts | + / − per fact | layer | silhouette mean | LOO 1-NN mean (worst) | "
               "LOO probe mean (worst) | head FN | head FP |", "|---|---|---|---|---|---|---|---|---|"]
        for ds, t in trained.items():
            rows = t["per_fact"]
            npos, nneg = sum(r["n_pos"] for r in rows), sum(r["n_neg"] for r in rows)
            fn, fp = sum(len(r["head_fn"]) for r in rows), sum(len(r["head_fp"]) for r in rows)
            for layer in t["layers_shown"]:
                b = t["by_layer"][str(layer)]
                head = (f"{fn}/{npos} | {fp}/{nneg}" if layer == t["router_layer"] else "– | –")
                md.append(f"| {NAMES[ds]} | {len(rows)} | {npos / max(1, len(rows)):.1f} / {nneg / max(1, len(rows)):.1f} | "
                          f"L{layer} | {_f(b['silhouette_mean'])} | {_f(b['nn_mean'])} ({_f(b['nn_min'])}) | "
                          f"{_f(b['probe_mean'])} ({_f(b['probe_min'])}) | {head} |")
        md.append("")
        for ds in trained:
            md.append(f"![{NAMES[ds]} training prompts per fact]({rel(ds, 'train_umap/umap_facts.png')})")
            md.append(f"![{NAMES[ds]} training separability vs layer]({rel(ds, 'train_umap/separability_vs_layer.png')})")
        md.append("")

    # ------------------------------------------------------------ 4. confusion
    md += ["## 4. Confusion matrices", "",
           "**What.** Binary: fire vs should-fire, per split. N-way: which row fired against which row was "
           "correct, with a final 'none' row and column; off-diagonal cells in the fact block are misroutes, the "
           "'none' column holds the misses, the 'none' row holds the false fires." +
           (" RWKU adds a person-level matrix: firing a row of the right person counts as correct." if "rwku" in data else ""), "",
           "**Theory.** Proposition 2 of the paper splits the residual into routed-and-edited and missed; the "
           "'none' column of the n-way matrix is exactly the missed term.", ""]
    for ds in data:
        md.append(f"![{NAMES[ds]} binary]({rel(ds, 'report/confusion_binary.png')}) "
                  f"![{NAMES[ds]} n-way]({rel(ds, 'report/confusion_nway.png')})")
    if "rwku" in data:
        md.append(f"![RWKU person]({rel('rwku', 'report/confusion_person.png')})")
    md.append("")

    # ------------------------------------------------------------ 5/7. layer sweep + penultimate
    md += ["## 5. Classifier quality at every block, and the penultimate block", "",
           "**What.** The same classifier, same data, same calibration rule, refit at every block and scored on the "
           "same evaluation prompts. The penultimate block is compared with block 19 under the deployed rule: it "
           "counts as better only if F1 rises and neither the false-positive rate nor the null false-positive rate "
           "gets worse.", "",
           "**Theory.** Early blocks carry mostly the surface form; middle-to-late blocks carry the question's "
           "meaning, where paraphrases of one fact converge; the last blocks specialise towards the next token. A "
           "classifier that must recognise paraphrases should peak in the middle-to-late range, and block 19 is in "
           "that range.", "",
           "| dataset | F1 at L19 | best block (F1) | penultimate F1 | penultimate null FPR vs L19 | verdict |",
           "|---|---|---|---|---|---|"]
    for ds, v in data.items():
        rows = {r[0]: r for r in v["sweep"]}
        best = max(v["sweep"], key=lambda r: (r[1] or 0)) if v["sweep"] else None
        pen = (v["penultimate"].get("deployed") or {})
        pl = [k for k in pen if k.startswith("L") and not k.endswith("null_fpr")]
        l19 = rows.get(L)
        pk = next((k for k in pl if k != f"L{L}"), None)
        md.append(f"| {NAMES[ds]} | {_f(l19[1] if l19 else None)} | "
                  f"{'L%d (%s)' % (best[0], _f(best[1])) if best else '–'} | "
                  f"{_f((pen.get(pk) or {}).get('f1')) if pk else '–'} | "
                  f"{_f(pen.get(f'{pk}_null_fpr')) if pk else '–'} vs {_f(pen.get(f'L{L}_null_fpr'))} | "
                  f"{'penultimate better' if pen.get('routing_improves') else f'keep L{L}'} |")
    md += ["", "Per-block tables (FPR, FNR, recall, null FPR, perturbed recall, calibration) are in each "
           "`layer_sweep/layer_choice.md`; robustness variants (hard negatives, augmentation, layer concatenation, "
           "no subject gate) in `layer_sweep/robustness.md`.", ""]
    for ds in data:
        md.append(f"![{NAMES[ds]} errors vs layer]({rel(ds, 'layer_sweep/errors_vs_layer.png')})")
    md.append("")

    # ------------------------------------------------------------ 6. RWKU
    if "rwku" in data:
        rw = data["rwku"]["metrics"].get("rwku") or {}
        md += ["## 6. RWKU", "",
               "**What.** RWKU forgets people, not single facts, and its held-out probes ask other facts about the "
               "same person. The breakdown counts errors by probe type and by how the person is named.", "",
               "**Theory.** The fact-level heads are trained with other facts about the same person as negatives, "
               "so on a held-out question about the person they correctly decline: a miss by the fact-level "
               "definition, but the wrong policy for an entity-level benchmark. That is why the person-level "
               "matrix, not the fact-level one, is the right lens here, and why the entity gate (fire on any probe "
               "that names a protected person) is the matching configuration.", "",
               "| probe type | n | should fire | fired | correct | FN | FP | misroute |",
               "|---|---|---|---|---|---|---|---|"]
        for kind, t in (rw.get("by_probe_type") or {}).items():
            md.append(f"| {kind} | {t['n']} | {t['should_fire']} | {t['fired']} | {t['correct']} | "
                      f"{t['false_negative']} | {t['false_positive']} | {t['misroute']} |")
        md.append("")

    # ------------------------------------------------------------ calibration + perturbed
    md += ["## 7. Surface robustness and calibration", "",
           "**What.** Forget prompts with typos, case changes and partial names; and the reliability of the "
           "deployed probabilities. The bias is shifted for recall on held-out rewrites, so probabilities are not "
           "meant to be calibrated; the reliability plot shows how far they are.", "",
           "| dataset | perturbation family | n | recall |", "|---|---|---|---|"]
    for ds, v in data.items():
        for fam, m in (v["metrics"].get("perturbed") or {}).items():
            md.append(f"| {NAMES[ds]} | {fam} | {m['n']} | {_f(m['recall'])} |")
    md.append("")
    for ds, v in data.items():
        cal = (v["metrics"].get("calibration") or {}).get("pair_level") or {}
        md.append(f"- {NAMES[ds]}: pair-level ECE {_f(cal.get('ece'))} over {cal.get('pairs', '–')} pairs. "
                  f"![{NAMES[ds]} reliability]({rel(ds, 'report/reliability.png')})")
    md += ["", "## Summary", ""]
    for ds, (a, reasons, main) in verdicts.items():
        total = sum(a.values())
        md.append(f"- **{NAMES[ds]}**: {total} routing errors; {a['gate']} from the gate, {a['classifier']} from the "
                  f"classifier heads, {a['label']} label conflicts. Main cause: {main}.")
    md.append("")
    out = Path(root) / "RESULTS_EXPLAINED.md"
    out.write_text("\n".join(md) + "\n")
    print(f"wrote {out}")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="outputs/router_diag_ours_v1")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    build(a.root, a.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
