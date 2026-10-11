#!/usr/bin/env python3
"""Projection edit (MCF, layer 19, setting A): remove the answer direction instead of adding Δe.

Same router as the shipped run (the linear classifier reads the last prompt token
at layer 19 and picks at most one forget fact). When it fires for fact i, the hook
replaces the last-token state h by

    h' = h - alpha * U_i U_i^T h

where U_i (orthonormal, k <= 3 rows) spans the "say <answer>" directions of fact
i's answer tokens at layer 19. Nothing is trained and no model weight changes.

Answer directions (one per answer token w, normalised, then orthonormalised per fact):
  jlens  J-lens style: average over generic MCF prompts (forget subjects excluded)
         of d logit_w / d h_19 at the last prompt token
  local  the same gradient, averaged over the fact's own TRAINING views only
         (development views and official eval prompts are never used)

Subcommands
  build      compute directions (or --reuse-dirs), write a projection artifact, and
             print a canonical-prompt diagnostic (P(answer) before/after)
  summarize  table: arms vs the seed-1 Δe reference (official MCF metrics)

    python scripts/mcf_projection_edit.py build --router-dir R --output-dir O \\
        --direction jlens --alpha 1 --local-files-only
    python scripts/mcf_projection_edit.py summarize --root outputs/mcf_projection_seed1
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil

import torch
import torch.nn.functional as F

REFERENCE = ("A: add Δe (seed-1 multiseed regular)",
             "outputs/mcf_multiseed_regular_v1/seed1/L19/linear_global")
MAX_ANSWER_TOKENS = 3


# --------------------------------------------------------------------- directions

def _block_output(out):
    return out[0] if isinstance(out, tuple) else out


def _replace_output(out, hidden):
    return (hidden, *out[1:]) if isinstance(out, tuple) else hidden


def answer_logit_gradients(model, layer, prompts, token_sets, device, batch_size=16):
    """Mean over prompts of d logit_w / d h_layer at each prompt's last token.

    prompts:    list of token-id lists (BOS included)
    token_sets: list (same length) of token-id lists; prompt p contributes to the
                gradient of every w in token_sets[p]
    Returns {w: (sum_grad [d], count)}.
    """
    unembed = model.get_output_embeddings().weight
    final_norm = model.model.norm
    sums, counts = {}, {}
    order = list(range(len(prompts)))
    for start in range(0, len(order), batch_size):
        idx = order[start:start + batch_size]
        lengths = [len(prompts[i]) for i in idx]
        width = max(lengths)
        ids = torch.zeros((len(idx), width), dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        for row, i in enumerate(idx):
            ids[row, :lengths[row]] = torch.tensor(prompts[i], device=device)
            mask[row, :lengths[row]] = 1
        last = torch.tensor([n - 1 for n in lengths], device=device)
        rows = torch.arange(len(idx), device=device)
        leaf, final = {}, {}

        def make_leaf(_m, _a, out):
            h = _block_output(out).detach().requires_grad_(True)
            leaf["h"] = h
            return _replace_output(out, h)

        def grab_final(_m, _a, out):
            final["h"] = out

        handles = [model.model.layers[layer].register_forward_hook(make_leaf),
                   final_norm.register_forward_hook(grab_final)]
        try:
            with torch.enable_grad():
                model.model(input_ids=ids, attention_mask=mask, use_cache=False)
                h_last = final["h"][rows, last].float()                     # [B, d] (post final norm)
                tokens = sorted({w for i in idx for w in token_sets[i]})
                scores = h_last @ unembed[tokens].float().T                  # [B, T] logits
                for j, w in enumerate(tokens):
                    use = torch.tensor([w in token_sets[i] for i in idx], device=device)
                    if not bool(use.any()):
                        continue
                    grad = torch.autograd.grad((scores[:, j] * use).sum(), leaf["h"],
                                               retain_graph=True)[0]
                    g = grad[rows, last].float()[use].sum(0).detach().cpu()
                    sums[w] = sums.get(w, 0) + g
                    counts[w] = counts.get(w, 0) + int(use.sum())
        finally:
            for handle in handles:
                handle.remove()
    return {w: (sums[w], counts[w]) for w in sums}


def orthonormal_basis(vectors, tol=1e-6):
    """Rows -> orthonormal rows spanning them (Gram-Schmidt, drops dependent rows)."""
    basis = []
    for v in vectors:
        v = v.float().clone()
        for b in basis:
            v = v - (v @ b) * b
        if float(v.norm()) > tol:
            basis.append(v / v.norm())
    return basis


def answer_tokens(examples):
    """fact_id -> answer token ids (canonical training view, up to MAX_ANSWER_TOKENS)."""
    out = {}
    for e in examples:
        if e.split != "train" or e.group != "canonical_rewrite":
            continue
        toks = [t for t, lab in zip(e.input_ids, e.labels) if lab != -100]
        out[e.fact_id] = list(dict.fromkeys(toks))[:MAX_ANSWER_TOKENS]
    return out


def generic_prompts(tokenizer, mcf_path, forget_facts, n, seed=0):
    records = json.loads(Path(mcf_path).read_text())
    forget_subjects = {str(f["subject"]).casefold() for f in forget_facts}
    pool = []
    for record in records:
        rewrite = record["requested_rewrite"]
        if str(rewrite["subject"]).casefold() in forget_subjects:
            continue
        pool.append(rewrite["prompt"].format(rewrite["subject"]))
    g = torch.Generator().manual_seed(seed)
    pick = torch.randperm(len(pool), generator=g)[:n].tolist()
    return [tokenizer(pool[i])["input_ids"] for i in pick]


# --------------------------------------------------------------------------- build

@torch.no_grad()
def canonical_diagnostic(model, layer, examples, fact_index, dirs, alpha, device):
    """P(first answer token) on each fact's canonical prompt, base vs projected."""
    rows = []
    for e in examples:
        if e.split != "train" or e.group != "canonical_rewrite":
            continue
        first = next(k for k, lab in enumerate(e.labels) if lab != -100)
        ids = torch.tensor([e.input_ids[:first]], device=device)
        target = e.input_ids[first]
        U = dirs[fact_index[e.fact_id]].to(device)
        U = U[U.norm(dim=-1) > 0]
        state = {}

        def project(_m, _a, out):
            h = _block_output(out).clone()
            v = h[0, -1].float()
            state["removed_frac"] = float((U.T @ (U @ v)).norm() / v.norm())
            h[0, -1] = (v - alpha * (U.T @ (U @ v))).to(h.dtype)
            return _replace_output(out, h)

        base = model(input_ids=ids, use_cache=False).logits[0, -1].float().softmax(-1)
        handle = model.model.layers[layer].register_forward_hook(project)
        try:
            edited = model(input_ids=ids, use_cache=False).logits[0, -1].float().softmax(-1)
        finally:
            handle.remove()
        rows.append({"fact_id": e.fact_id, "prompt": e.prompt, "answer": e.completion,
                     "p_first_base": float(base[target]), "p_first_projected": float(edited[target]),
                     "removed_norm_fraction": state["removed_frac"]})
    return rows


def build(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from prepare_mcf_association_source import load_mcf_forget_data

    router_dir, output = Path(args.router_dir), Path(args.output_dir)
    artifact = torch.load(router_dir / "fact_association_embeddings.pt",
                          map_location="cpu", weights_only=False)
    manifest = json.loads((router_dir / "association_manifest.json").read_text())
    layer = int(artifact["layer"])
    if float(artifact["rows"].abs().max()) != 0.0:
        raise ValueError("Expected a fitted router with zero rows")

    tok = AutoTokenizer.from_pretrained(manifest["model_path"], use_fast=True,
                                        local_files_only=args.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        manifest["model_path"], dtype=torch.float32, local_files_only=args.local_files_only,
        attn_implementation="eager").to(args.device).eval()
    model.requires_grad_(False)
    seed = int((manifest.get("sampling") or {}).get("seed", 1))
    _, facts, examples = load_mcf_forget_data(tok, manifest["mcf_path"], seed=seed)
    if [f["id"] for f in facts] != [f["id"] for f in artifact["facts"]]:
        raise ValueError("Rebuilt MCF facts do not match the router artifact")
    fact_index = {f["id"]: i for i, f in enumerate(facts)}
    answers = answer_tokens(examples)
    if args.answer_tokens == "first":
        answers = {k: v[:1] for k, v in answers.items()}

    output.mkdir(parents=True, exist_ok=True)
    if args.reuse_dirs:
        saved = torch.load(args.reuse_dirs, map_location="cpu", weights_only=False)
        dirs, info = saved["dirs"], saved["info"]
        print(f"reusing directions from {args.reuse_dirs}")
    else:
        if args.direction == "jlens":
            prompts = generic_prompts(tok, manifest["mcf_path"], facts, args.jlens_prompts)
            all_tokens = sorted({w for ws in answers.values() for w in ws})
            grads = answer_logit_gradients(model, layer, prompts, [all_tokens] * len(prompts),
                                           args.device, args.batch_size)
            per_fact = {fid: [grads[w][0] / grads[w][1] for w in ws] for fid, ws in answers.items()}
            source = f"{len(prompts)} generic MCF prompts (forget subjects excluded)"
        else:
            per_fact = {}
            for fid, ws in answers.items():
                views = []
                for e in examples:
                    if e.fact_id == fid and e.split == "train":
                        first = next(k for k, lab in enumerate(e.labels) if lab != -100)
                        views.append(e.input_ids[:first])
                grads = answer_logit_gradients(model, layer, views, [ws] * len(views),
                                               args.device, args.batch_size)
                per_fact[fid] = [grads[w][0] / grads[w][1] for w in ws]
            source = "each fact's own training views"
        k = max(len(v) for v in answers.values())
        dirs = torch.zeros(len(facts), k, int(artifact["feature_mean"].shape[-1]))
        for fid, vecs in per_fact.items():
            for j, b in enumerate(orthonormal_basis(vecs)):
                dirs[fact_index[fid], j] = b
        unembed = model.get_output_embeddings().weight.detach().float().cpu()
        cos_unembed = [float(F.cosine_similarity(dirs[fact_index[fid], 0],
                                                 unembed[answers[fid][0]], dim=0))
                       for fid in answers]
        info = {"direction": args.direction, "source": source,
                "answer_tokens": args.answer_tokens, "k_max": k,
                "answer_token_ids": answers,
                "cos_first_direction_vs_unembedding_median": float(torch.tensor(cos_unembed).median())}
        torch.save({"dirs": dirs, "info": info}, output / "projection_dirs.pt")

    diag = canonical_diagnostic(model, layer, examples, fact_index, dirs, args.alpha, args.device)
    p_before = torch.tensor([r["p_first_base"] for r in diag])
    p_after = torch.tensor([r["p_first_projected"] for r in diag])
    removed = torch.tensor([r["removed_norm_fraction"] for r in diag])
    summary = {
        "facts": len(diag), "alpha": args.alpha, **info,
        "canonical_p_first_base_median": float(p_before.median()),
        "canonical_p_first_projected_median": float(p_after.median()),
        "canonical_facts_p_first_below_1e-2": int((p_after < 1e-2).sum()),
        "canonical_facts_p_first_below_1e-6": int((p_after < 1e-6).sum()),
        "removed_norm_fraction_median": float(removed.median()),
    }
    (output / "projection_diagnostic.json").write_text(
        json.dumps({"summary": summary, "per_fact": diag}, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "answer_token_ids"}, indent=2))

    run = output / "linear_global"
    if not (run / "fact_association_embeddings.pt").exists():
        tmp = output / "linear_global.tmp"
        if tmp.exists():
            shutil.rmtree(tmp)
        shutil.copytree(router_dir, tmp)
        artifact.update({"edit_mode": "project", "project_dirs": dirs.float(),
                         "project_alpha": float(args.alpha), "write_mode": "last"})
        torch.save(artifact, tmp / "fact_association_embeddings.pt")
        manifest.update({"method": "sure_linear_router_projection_edit",
                         "edit_mode": "project", "write_mode": "last",
                         "project_alpha": float(args.alpha), "projection": info,
                         "rows_trained": False, "router_copied_from": str(router_dir.resolve())})
        (tmp / "association_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        tmp.rename(run)
    print(f"artifact: {run}")


# ------------------------------------------------------------------------ summarize

def _fmt(value):
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def summarize(root):
    from summarize_mcf_layer_sweep import collect

    root = Path(root)
    rows = []
    if Path(REFERENCE[1]).is_dir():
        rows.append((collect(REFERENCE[1], REFERENCE[0]), {}))
    for arm in sorted(p for p in root.iterdir() if (p / "linear_global").is_dir()):
        diag_path = arm / "projection_diagnostic.json"
        diag = json.loads(diag_path.read_text())["summary"] if diag_path.is_file() else {}
        rows.append((collect(arm / "linear_global", arm.name), diag))
    cols = (("forget_Eff", "forget Eff ↓"), ("forget_Gen", "forget Gen ↓"),
            ("forget_Spe", "forget Spe"), ("retain_Eff", "retain Eff"),
            ("retain_Gen", "retain Gen"), ("PPL", "PPL"))
    dcols = (("canonical_p_first_base_median", "canon P(ans₁) base"),
             ("canonical_p_first_projected_median", "canon P(ans₁) edited"),
             ("canonical_facts_p_first_below_1e-6", "canon facts <1e-6"),
             ("removed_norm_fraction_median", "‖removed‖/‖h‖"))
    header = ["setting", "status"] + [t for _, t in cols] + [t for _, t in dcols]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row, diag in rows:
        lines.append("| " + " | ".join([row["label"], _fmt(row.get("status"))]
                                       + [_fmt(row.get(k)) for k, _ in cols]
                                       + [_fmt(diag.get(k)) for k, _ in dcols]) + " |")
    text = ("# MCF seed 1, 50 facts, layer 19, setting A: projection vs Δe\n\n"
            "Router identical in every row. Eff/Gen: complete sensitive-answer probability in %, "
            "lower is better. 'canon' columns: first answer token on each fact's canonical prompt.\n\n"
            + "\n".join(lines) + "\n")
    (root / "summary.md").write_text(text)
    print(text)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--router-dir", required=True)
    b.add_argument("--output-dir", required=True)
    b.add_argument("--direction", choices=("jlens", "local"), default="jlens")
    b.add_argument("--answer-tokens", choices=("first", "all"), default="all")
    b.add_argument("--alpha", type=float, default=1.0,
                   help="1 = remove the component (projection); 2 = reflect it")
    b.add_argument("--jlens-prompts", type=int, default=256)
    b.add_argument("--batch-size", type=int, default=16)
    b.add_argument("--reuse-dirs", default=None, help="projection_dirs.pt from another arm")
    b.add_argument("--device", default="cuda")
    b.add_argument("--local-files-only", action="store_true")
    s = sub.add_parser("summarize")
    s.add_argument("--root", required=True)
    args = parser.parse_args(argv)
    if args.command == "build":
        build(args)
    else:
        summarize(args.root)


if __name__ == "__main__":
    main()
