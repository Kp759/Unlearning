#!/usr/bin/env python3
"""Where is the subject linearly readable? Probe heatmap (MCF, one seed, base model).

Rows = layers, columns = position (the subject's LAST token vs the prompt's last token),
cell = held-out open-set accuracy of a linear softmax probe that names WHICH forget
subject a prompt is about, or "none" (other MCF subjects and near-miss names, as for the
router): 1/2 (forget subject named correctly) + 1/2 (other entity rejected). Justifies the v2 router design (subject head
read at subject tokens, mid layers); the effect itself is known (ROME, Geva et al. 2023).

Data: scripts/factorized_router.build_router_data (training split -> probe fit,
validation split -> score), keeping prompts whose subject string is found verbatim (or
lower-cased) in the prompt, so the subject's last token is defined.

    python -u scripts/mcf_subject_probe_heatmap.py --reference-dir R --out-dir O --local-files-only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

from factorized_router import build_router_data, fit_softmax  # noqa: E402


def subject_last_token(tok, prompt, subject):
    """(token index of the subject's last character, n tokens) or None if not found."""
    k = prompt.find(subject)
    if k < 0:
        k = prompt.casefold().find(subject.casefold())
    if k < 0:
        return None
    end = k + len(subject) - 1
    enc = tok(prompt, return_offsets_mapping=True)
    for i, (a, b) in enumerate(enc["offset_mapping"]):
        if a <= end < b:
            return i, len(enc["input_ids"])
    return None


@torch.no_grad()
def extract(model, tok, prompts, positions, layers, device, batch_size=32):
    """[N, len(layers), 2, d] float16: subject-last-token and last-token states."""
    store, handles = {}, []
    for L in layers:
        handles.append(model.model.layers[L].register_forward_hook(
            lambda m, a, o, L=L: store.__setitem__(L, o[0] if isinstance(o, tuple) else o)))
    out = []
    try:
        for start in range(0, len(prompts), batch_size):
            enc = [tok(p)["input_ids"] for p in prompts[start:start + batch_size]]
            width = max(map(len, enc))
            ids = torch.full((len(enc), width), tok.pad_token_id or 0, dtype=torch.long)
            att = torch.zeros_like(ids)
            for i, e in enumerate(enc):
                ids[i, :len(e)] = torch.tensor(e)
                att[i, :len(e)] = 1
            model(input_ids=ids.to(device), attention_mask=att.to(device), use_cache=False)
            r = torch.arange(len(enc), device=device)
            subj = torch.tensor(positions[start:start + len(enc)], device=device)
            last = att.sum(1).to(device) - 1
            out.append(torch.stack([torch.stack([store[L][r, subj], store[L][r, last]], 1)
                                    for L in layers], 1).half().cpu())
    finally:
        for h in handles:
            h.remove()
    return torch.cat(out)


def balanced_accuracy(pred, y, n_classes):
    acc = [float((pred[y == c] == c).float().mean()) for c in range(n_classes) if (y == c).any()]
    return sum(acc) / len(acc)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reference-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--layers", type=int, nargs="+", default=None, help="default: every 2nd layer + last")
    p.add_argument("--none-train", type=int, default=4000)
    p.add_argument("--none-val", type=int, default=2000)
    p.add_argument("--l2", type=float, default=1e-3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--local-files-only", action="store_true")
    args = p.parse_args(argv)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from prepare_mcf_association_source import load_mcf_forget_data

    ref = Path(args.reference_dir)
    manifest = json.loads((ref / "association_manifest.json").read_text())
    seed = int((manifest.get("sampling") or {}).get("seed", 1))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(manifest["model_path"], use_fast=True,
                                        local_files_only=args.local_files_only)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        manifest["model_path"], dtype=torch.bfloat16, local_files_only=args.local_files_only,
        attn_implementation="eager").to(args.device).eval()
    model.requires_grad_(False)
    n_layers = len(model.model.layers)
    layers = args.layers or sorted(set(range(0, n_layers, 2)) | {n_layers - 1})

    records = json.loads(Path(manifest["mcf_path"]).read_text())
    forget_records, facts, examples = load_mcf_forget_data(tok, manifest["mcf_path"], seed=seed)
    examples = [e if isinstance(e, dict) else {k: getattr(e, k) for k in ("fact_id", "prompt", "split", "group")
                                               if hasattr(e, k)} for e in examples]
    data = build_router_data(records, forget_records, facts, examples, seed=seed)
    S = len(data["subjects"])
    rng = random.Random(seed)
    keep = {"train": [], "val": []}
    for it in data["items"]:
        if not it.get("subject_text") or "+typo" in it["kind"]:
            continue
        found = subject_last_token(tok, it["prompt"], it["subject_text"])
        if found is None or found[0] == found[1] - 1:      # subject must not BE the last token
            continue
        keep[it["split"]].append((it, found[0]))
    for split, cap in (("train", args.none_train), ("val", args.none_val)):
        pos = [x for x in keep[split] if x[0]["subject"] < S]
        neg = [x for x in keep[split] if x[0]["subject"] == S]
        keep[split] = pos + rng.sample(neg, min(cap, len(neg)))
    rows = keep["train"] + keep["val"]
    split = torch.tensor([i < len(keep["train"]) for i in range(len(rows))])
    y = torch.tensor([it["subject"] for it, _ in rows])
    print(json.dumps({"phase": "data", "train": len(keep["train"]), "val": len(keep["val"]),
                      "forget_subject_prompts": int((y < S).sum()), "layers": layers}), flush=True)
    H = extract(model, tok, [it["prompt"] for it, _ in rows], [k for _, k in rows], layers, args.device)

    table = {}
    for li, L in enumerate(layers):
        table[L] = {}
        for pi, name in enumerate(("subject_token", "last_token")):
            X = F.normalize(H[:, li, pi].float(), dim=-1)
            X = X - X[split].mean(0)
            W, b = fit_softmax(X[split], y[split], S + 1, args.l2, device=args.device)
            pred = (X[~split] @ W.T + b).argmax(-1)
            yv = y[~split]
            top1 = float((pred[yv < S] == yv[yv < S]).float().mean())
            none_ok = float((pred[yv == S] == S).float().mean())
            table[L][name] = {
                # open set: naming the right forget subject and rejecting every other entity
                # count equally (a 51-class average would let 50 subject classes swamp "none")
                "balanced_acc": 0.5 * (top1 + none_ok),
                "class_balanced_acc_51": balanced_accuracy(pred, yv, S + 1),
                "forget_subject_top1": top1,
                "none_correct": none_ok,
            }
        print(json.dumps({"layer": L, **{k: round(v["balanced_acc"], 3) for k, v in table[L].items()}}), flush=True)

    best_subject = max(table, key=lambda L: table[L]["subject_token"]["balanced_acc"])
    summary = {"layers": table, "best_subject_token_layer": best_subject,
               "n_train": len(keep["train"]), "n_val": len(keep["val"]), "classes": S + 1}
    (out / "heatmap.json").write_text(json.dumps(summary, indent=2) + "\n")
    lines = ["# Where is the subject linearly readable? (MCF seed %d, base model)\n" % seed,
             "Held-out open-set accuracy of a linear probe naming the forget subject or 'none' "
             f"({S + 1} classes; none = other MCF subjects and near-miss names): "
             "½ (forget subject named) + ½ (other entity rejected).\n",
             "| layer | subject's last token | prompt's last token | (forget-subject top-1: subj / last) | (none correct: subj / last) |",
             "|---|---|---|---|---|"]
    for L in layers:
        a, b = table[L]["subject_token"], table[L]["last_token"]
        lines.append(f"| {L} | {100 * a['balanced_acc']:.1f}% | {100 * b['balanced_acc']:.1f}% | "
                     f"{100 * a['forget_subject_top1']:.0f}% / {100 * b['forget_subject_top1']:.0f}% | "
                     f"{100 * a['none_correct']:.0f}% / {100 * b['none_correct']:.0f}% |")
    lines.append(f"\nBest subject-token layer: {best_subject}.")
    (out / "heatmap.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        M = torch.tensor([[table[L]["subject_token"]["balanced_acc"], table[L]["last_token"]["balanced_acc"]]
                          for L in layers])
        fig, ax = plt.subplots(figsize=(3.2, 0.28 * len(layers) + 1))
        im = ax.imshow(M.numpy(), aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks([0, 1], ["subject\ntoken", "last\ntoken"])
        ax.set_yticks(range(len(layers)), [str(L) for L in layers])
        ax.set_ylabel("layer")
        for i in range(len(layers)):
            for j in range(2):
                ax.text(j, i, f"{100 * M[i, j]:.0f}", ha="center", va="center", fontsize=7,
                        color="white" if M[i, j] < 0.6 else "black")
        fig.colorbar(im, ax=ax, fraction=0.08, label="open-set accuracy")
        fig.tight_layout()
        fig.savefig(out / "heatmap.png", dpi=200)
        fig.savefig(out / "heatmap.pdf")
    except Exception as exc:                                  # figure is optional
        print(f"(no figure: {exc})")


if __name__ == "__main__":
    main()
