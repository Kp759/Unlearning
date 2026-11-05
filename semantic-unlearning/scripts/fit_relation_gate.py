#!/usr/bin/env python3
"""Relation gate for a linear router: which relation does the prompt ask about?

Analysis only: the paper's routers carry no relation gate (README_ROUTER_DIAGNOSTICS.md,
"Relation gate (analysis only)"); this writes a separate router dir.

The per-fact heads read the last-token state at the router layer, where the subject dominates:
"Gautier de Coincy originates from" fires the head of "The mother tongue of Gautier de Coincy
is" (French). The gate is a softmax probe over MCF relations on the SAME unit-normalised
router-layer query; a fact's head stays eligible only if log p(the fact's relation | prompt)
>= min_log_prob (linear_router.LinearClassifierAssociationBank, artifact["relation_gate"]).
No extra forward pass; heads, cutoff and rows are untouched.

Probe data: MCF prompts (rewrite, paraphrase, neighborhood, attribute) of the records the
router diagnostics treat as training-side (router_diag_probes: the aug half of the records
outside this seed's official forget + retain sample), with every prompt that names a
protected subject dropped, at most --cap per relation. So the probe never sees a protected
subject or an evaluated prompt: it has to learn the relation, not the subject. A share
(--suffix-share) is also seen with an instruction after the question (SUFFIXES), since the
probe reads the last token: without these views "<forget prompt> Please answer briefly."
was blocked (none of SUFFIXES is the diagnostics' own perturbation suffix).

Cutoff: the --quantile of log p(own relation) over the router's own training positives
(linear_router_dataset.json, every split): 0.15 lets 85 % of them through.

    python -u scripts/fit_relation_gate.py \
        --router-dir outputs/mcf_multiseed_ours_v1/seed1/L19/router \
        --out-dir outputs/mcf_multiseed_ours_relgate_v1/seed1/L19/router \
        --seed 1 --device cuda --local-files-only

Writes <out-dir>: a copy of the router dir with relation_gate in fact_association_embeddings.pt,
and relation_gate_report.json (probe accuracy, cutoff, per-relation counts).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import shutil

import torch
from torch.nn import functional as F

from cache_router_states import capture_last_token_states
from linear_router import is_linear_router_artifact, relation_log_probs
import router_diag_probes as probes_lib

# instructions after the question; the diagnostics' perturbation "Please answer briefly." is not one
SUFFIXES = (" Answer in one word.", " Be concise.", " Reply with just the answer.", " Keep it short.",
            " Give a short answer.", "\nAnswer:", " Thanks!", " One word only.")


def relation_prompts(mcf_path, seed, protected_subjects, cap, suffix_share=0.0, rng_seed=0):
    """{relation: [prompt, ...]} from the training-side MCF records (see the module doc)."""
    import mcf_zero_unlearn_official_eval as mcf

    forget, retain = mcf.load_official_eval_records(
        mcf_path, probes_lib.FORGET_NUM, probes_lib.OFFICIAL_RETAIN_NUM, seed, "official")
    sampled = {int(r["case_id"]) for r in forget + retain}
    pool = sorted((r for r in mcf.load_mcf(mcf_path) if int(r["case_id"]) not in sampled),
                  key=lambda r: int(r["case_id"]))
    train_pool, _ = probes_lib._halves(pool)
    protected = [s.casefold() for s in protected_subjects if s.strip()]
    by_relation, seen = {}, set()
    for record in train_pool:
        rr = probes_lib._mcf_rr(record)
        texts = [rr["prompt"].format(rr["subject"])] + list(record.get("paraphrase_prompts", [])) \
            + list(record.get("neighborhood_prompts", [])) + list(record.get("attribute_prompts", []))
        for text in texts:
            key = " ".join(text.casefold().split())
            if key in seen or any(s in key for s in protected):
                continue
            seen.add(key)
            by_relation.setdefault(str(rr["relation_id"]), []).append(text)
    rng = random.Random(rng_seed)
    for relation in by_relation:
        rng.shuffle(by_relation[relation])
        kept = by_relation[relation][:cap]
        by_relation[relation] = kept + [t + rng.choice(SUFFIXES) for t in kept[:int(suffix_share * len(kept))]]
    return dict(sorted(by_relation.items()))


def fit_softmax_probe(x, y, n_classes, l2, device):
    """Multinomial logistic regression (L-BFGS): weight [C, d], bias [C]."""
    x, y = x.to(device), y.to(device)
    weight = torch.zeros(n_classes, x.shape[1], device=device, requires_grad=True)
    bias = torch.zeros(n_classes, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([weight, bias], max_iter=500, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(x @ weight.T + bias, y) + l2 * (weight ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    return weight.detach().cpu(), bias.detach().cpu()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--router-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, required=True, help="the router's seed (its official eval sample)")
    ap.add_argument("--mcf-path", default="data/multi_counterfact.json")
    ap.add_argument("--model-path", default=None, help="default: the router manifest's model_path")
    ap.add_argument("--cap", type=int, default=2000, help="probe prompts per relation")
    ap.add_argument("--suffix-share", type=float, default=0.25,
                    help="share of them also seen with an instruction suffix (SUFFIXES)")
    ap.add_argument("--l2", default="1e-5,1e-4,1e-3", help="picked on a held-out tenth of the probe prompts")
    ap.add_argument("--quantile", type=float, default=0.15,
                    help="share of the router's training positives the gate may block")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="bfloat16", help="what every official evaluator runs")
    ap.add_argument("--local-files-only", action="store_true")
    args = ap.parse_args(argv)

    router_dir, out_dir = Path(args.router_dir), Path(args.out_dir)
    artifact = torch.load(router_dir / "fact_association_embeddings.pt", map_location="cpu", weights_only=False)
    manifest = json.loads((router_dir / "association_manifest.json").read_text())
    if not is_linear_router_artifact(artifact):
        raise ValueError(f"{router_dir} is not a linear-router dir")
    if artifact.get("relation_gate") is not None:
        raise ValueError(f"{router_dir} already has a relation gate")
    facts = list(artifact["facts"])
    layer = int(artifact["layer"])

    by_relation = relation_prompts(args.mcf_path, args.seed, [str(f.get("subject", "")) for f in facts],
                                   args.cap, args.suffix_share)
    relations = list(by_relation)
    fact_relation = [relations.index(str(f.get("relation"))) if str(f.get("relation")) in by_relation else -1
                     for f in facts]
    print(json.dumps({"phase": "probe_data", "relations": {r: len(v) for r, v in by_relation.items()},
                      "facts_not_gated": sum(r < 0 for r in fact_relation)}), flush=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = Path(args.model_path or manifest["model_path"]).resolve()
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=args.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=getattr(torch, args.dtype), local_files_only=args.local_files_only,
        attn_implementation="eager").to(args.device).eval()
    model.requires_grad_(False)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def states(texts):
        ids = [tok(t)["input_ids"] for t in texts]
        return capture_last_token_states(model, ids, [layer], batch_size=args.batch_size, pad_id=pad)[0].float()

    texts = [t for r in relations for t in by_relation[r]]
    y = torch.tensor([i for i, r in enumerate(relations) for _ in by_relation[r]])
    x = F.normalize(states(texts), dim=-1)
    held = torch.tensor([k % 10 == 0 for k in range(len(texts))])
    accuracy = {}
    for l2 in [float(v) for v in args.l2.split(",")]:
        w, b = fit_softmax_probe(x[~held], y[~held], len(relations), l2, args.device)
        accuracy[l2] = float(((x[held] @ w.T + b).argmax(1) == y[held]).float().mean())
        print(json.dumps({"phase": "l2", "l2": l2, "heldout_accuracy": round(accuracy[l2], 4)}), flush=True)
    l2 = max(accuracy, key=accuracy.get)
    weight, bias = fit_softmax_probe(x, y, len(relations), l2, args.device)

    # cutoff from the router's own training positives
    dataset = json.loads((router_dir / "linear_router_dataset.json").read_text())
    fact_index = {str(f["id"]): i for i, f in enumerate(facts)}
    positives = [(e["prompt"], fact_index[e["owner_fact_id"]]) for e in dataset
                 if e.get("kind") == "positive" and e.get("owner_fact_id") in fact_index]
    q = states([p for p, _ in positives])
    own = relation_log_probs(q, weight, bias, fact_relation)[torch.arange(len(positives)),
                                                             torch.tensor([f for _, f in positives])]
    gated = torch.tensor([fact_relation[f] >= 0 for _, f in positives])
    values = own[gated].sort().values
    min_log_prob = float(values[int(args.quantile * (len(values) - 1))])

    gate = {"weight": weight, "bias": bias, "fact_relation": torch.tensor(fact_relation), "relations": relations,
            "min_log_prob": min_log_prob, "layer": layer, "feature": "unit_normalized_last_token",
            "quantile": args.quantile, "suffix_share": args.suffix_share, "l2": l2, "heldout_accuracy": accuracy[l2],
            "probe_prompts": len(texts), "source": "scripts/fit_relation_gate.py"}
    report = {k: v for k, v in gate.items() if k not in ("weight", "bias", "fact_relation")}
    report.update({"router_dir": str(router_dir), "seed": args.seed, "mcf_path": args.mcf_path,
                   "probe_prompts_per_relation": {r: len(v) for r, v in by_relation.items()},
                   "facts_not_gated": [facts[i]["id"] for i, r in enumerate(fact_relation) if r < 0],
                   "training_positives": int(gated.sum()),
                   "training_positives_blocked": int((own[gated] < min_log_prob).sum()),
                   "min_prob": float(torch.tensor(min_log_prob).exp())})

    if out_dir.exists():
        raise FileExistsError(f"{out_dir} exists; move it aside first")
    shutil.copytree(router_dir, out_dir)
    artifact["relation_gate"] = gate
    torch.save(artifact, out_dir / "fact_association_embeddings.pt")
    (out_dir / "relation_gate_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({"phase": "done", **{k: report[k] for k in (
        "heldout_accuracy", "l2", "min_prob", "training_positives", "training_positives_blocked")},
        "out_dir": str(out_dir)}), flush=True)


if __name__ == "__main__":
    main()
