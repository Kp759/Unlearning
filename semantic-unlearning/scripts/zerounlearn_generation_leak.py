#!/usr/bin/env python3
"""Does ZeroUnlearn leak the forgotten answer in free generation? (MCF, one seed)

Eff/Gen (ZeroUnlearn Eq. 16) only score the answer *immediately after* the prompt.
This runs the same free-generation check we ran for Δe and the projection edit
(`mcf_projection_edit.py routed-generate`) on a ZeroUnlearn-edited model:

  1. base model (bf16, eager attention, same as routed-generate) and the seed's MCF
     forget prompts: canonical rewrite + development paraphrase views, i.e. exactly the
     prompts used for the Δe / projection rows
  2. greedy uncached continuations (24 tokens) from the BASE model
  3. original ZeroUnlearn with the target_true parity mapping (MCF target_true = M_f,
     EOS = M_n; reviewed hparams, W_down of layers 16-18), as in
     run_zerounlearn_targettrue_parity_mcf_multiseed.py; optional official MCF eval
  4. greedy continuations from the EDITED model
  5. restore the original W_down weights (bit-exact; checked) and score both sets with
     the base model: fluency = mean NLL per continuation token, repeated 3-grams

Leak metrics (computed per prompt, every prompt counts: the edit is global):
  leak_any       true answer anywhere in the continuation (casefold substring)
  leak_start     continuation starts with the answer (what Eff/Gen look at)
  leak_any_excl  leak_any over prompts whose text does NOT already contain the answer
                 (e.g. "BMW X5 is produced by" -> BMW is trivially copyable)

Subcommands
  run      steps 1-5 -> OUT/base/routed_generation.json, OUT/zerounlearn/routed_generation.json
           (same schema as routed-generate), OUT/zerounlearn/official_mcf_eval.json
           with --official-eval
  compare  one table on the same prompts: base / Δe / projection alphas / ZeroUnlearn

    python scripts/zerounlearn_generation_leak.py run --seed 1 \\
        --output-dir outputs/zerounlearn_generation_leak_seed1 --official-eval --local-files-only
    python scripts/zerounlearn_generation_leak.py compare \\
        --zero-dir outputs/zerounlearn_generation_leak_seed1 \\
        --sweep-root outputs/mcf_projection_alpha_sweep_seed1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

SEMANTIC_ROOT = Path(__file__).resolve().parents[1]
FORGET_NUM, RETAIN_NUM = 50, 1000


# ----------------------------------------------------------------------- generation

def eos_id_set(model, tok):
    eos = getattr(model.config, "eos_token_id", None)
    if eos is None:
        eos = tok.eos_token_id
    if eos is None:
        return set()
    return {int(e) for e in eos} if isinstance(eos, (list, tuple, set)) else {int(eos)}


@torch.no_grad()
def greedy(model, ids, max_new_tokens, eos_ids):
    """Uncached greedy decoding (same loop as generate_uncached_fixed_boundary)."""
    for _ in range(int(max_new_tokens)):
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
        if int(nxt.item()) in eos_ids:
            break
    return ids


def forget_views(tok, mcf_path, seed):
    from prepare_mcf_association_source import load_mcf_forget_data

    forget_records, _, examples = load_mcf_forget_data(tok, mcf_path, seed=seed)
    views = [e for e in examples
             if (e.split == "train" and e.group == "canonical_rewrite") or e.split == "development"]
    return forget_records, views


def generate_all(model, tok, views, device, max_new_tokens):
    eos_ids = eos_id_set(model, tok)
    rows = []
    for e in views:
        first = next(k for k, lab in enumerate(e.labels) if lab != -100)
        ids = torch.tensor([e.input_ids[:first]], device=device)
        out = greedy(model, ids, max_new_tokens, eos_ids)
        rows.append({"id": e.id, "split": e.split, "prompt": e.prompt,
                     "answer": e.completion.strip(), "prompt_len": first,
                     "ids": out[0].tolist()})
    return rows


@torch.no_grad()
def score(base_model, tok, rows, device, routed):
    """Fill continuation text, leak flags, base-model NLL and repeated 3-grams in place."""
    for r in rows:
        cont = r["ids"][r["prompt_len"]:]
        text = tok.decode(cont, skip_special_tokens=True)
        answer = r["answer"].casefold()
        r["continuation"] = text
        r["routed"] = routed
        r["answer_in_continuation"] = answer in text.casefold()
        r["answer_at_start"] = text.strip().casefold().startswith(answer)
        r["answer_in_prompt"] = answer in r["prompt"].casefold()
        if cont:
            ids = torch.tensor([r["ids"]], device=device)
            logits = base_model(input_ids=ids, use_cache=False).logits[0].float()
            r["base_nll_per_token"] = float(F.cross_entropy(
                logits[r["prompt_len"] - 1:-1], ids[0, r["prompt_len"]:], reduction="mean"))
        grams = [tuple(cont[i:i + 3]) for i in range(max(0, len(cont) - 2))]
        r["repeated_trigram"] = len(grams) != len(set(grams))
    return rows


def leak_summary(rows):
    def rate(key, sub):
        return sum(bool(r[key]) for r in sub) / max(1, len(sub))

    clean = [r for r in rows if not r["answer_in_prompt"]]
    nlls = [r["base_nll_per_token"] for r in rows if "base_nll_per_token" in r]
    return {
        "prompts": len(rows),
        "leak_any": rate("answer_in_continuation", rows),
        "leak_start": rate("answer_at_start", rows),
        "prompts_answer_not_in_prompt": len(clean),
        "leak_any_excl": rate("answer_in_continuation", clean),
        "base_nll_per_token_mean": sum(nlls) / max(1, len(nlls)),
        "repeated_trigram": rate("repeated_trigram", rows),
    }


def write_generation(path, rows, summary):
    path.parent.mkdir(parents=True, exist_ok=True)
    clean = [{k: v for k, v in r.items() if k != "ids"} for r in rows]
    path.write_text(json.dumps({"summary": summary, "examples": clean}, indent=2) + "\n")


# ------------------------------------------------------------------------- ZeroUnlearn

def snapshot(model, names):
    params = dict(model.named_parameters())
    missing = [n for n in names if n not in params]
    if missing:
        raise KeyError(f"edited parameters not found: {missing}")
    return {n: params[n].detach().to("cpu", copy=True) for n in names}


def restore(model, saved):
    params = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in saved.items():
            params[name].copy_(value.to(params[name].device, params[name].dtype))


def apply_zerounlearn(model, tok, args):
    """Original ZeroUnlearn, target_true parity mapping (copied from the multiseed runner)."""
    import run_zerounlearn_official_mcf as official
    import run_zerounlearn_targettrue_parity_mcf as parity

    _, forget_records, retain_records = official.load_official_split(
        Path(args.mcf_path), seed=args.seed, forget_num=FORGET_NUM, retain_num=RETAIN_NUM,
        sample_mode="official")
    neutral_target, _ = official.resolve_eos_neutral_target(tok)
    params_class, apply_unl_to_model = official.import_original_zerounlearn(Path(args.zero_unlearn_root))
    hparams = params_class.from_json(Path(args.hparams_path))
    if list(hparams.layers) != [16, 17, 18]:
        raise RuntimeError(f"reviewed ZeroUnlearn layers changed: {list(hparams.layers)}")
    retain_requests = official.records_to_zero_unlearn_requests(retain_records)
    forget_requests = parity.targettrue_forget_requests(forget_records, neutral_target=neutral_target)
    parity.validate_targettrue_requests(forget_records, forget_requests, neutral_target)
    official.set_all_seeds(args.seed)
    model.float()
    with official.working_directory(official.SEMANTIC_ROOT):
        edited, _ = apply_unl_to_model(
            model=model, tok=tok, retain_requests=retain_requests,
            unlearn_requests=forget_requests, hparams=hparams, copy=False,
            return_orig_weights=False, cache_template=None, save_path=None,
            add_retain=official.ADD_RETAIN, edit_layer_nums=official.EDIT_LAYER_NUMS,
            use_h=official.USE_H)
    edited.to(dtype=torch.bfloat16).eval()
    names = [f"{hparams.rewrite_module_tmp.format(layer)}.weight" for layer in hparams.layers]
    return edited, forget_records, names


def edited_param_names(args):
    if args.edit == "none":
        return []
    hp = json.loads(Path(args.hparams_path).read_text())
    return [f"{hp['rewrite_module_tmp'].format(layer)}.weight" for layer in hp["layers"]]


def run(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        if device.index is None:                     # set_device needs an explicit index
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
    tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=True,
                                        local_files_only=args.local_files_only)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, local_files_only=args.local_files_only,
        attn_implementation=args.attn_implementation, low_cpu_mem_usage=True).to(device).eval()
    model.config.use_cache = False
    model.requires_grad_(False)

    forget_records, views = forget_views(tok, args.mcf_path, args.seed)
    print(f"seed {args.seed}: {len(forget_records)} forget facts, {len(views)} prompts "
          f"(canonical + development)", flush=True)

    t0 = time.perf_counter()
    base_rows = generate_all(model, tok, views, device, args.max_new_tokens)
    print(f"base generation: {time.perf_counter() - t0:.0f}s", flush=True)

    names = edited_param_names(args)
    saved = snapshot(model, names)
    t0 = time.perf_counter()
    if args.edit == "zerounlearn":
        model.requires_grad_(True)                    # ZeroUnlearn optimises v* through the model
        model, zero_records, names2 = apply_zerounlearn(model, tok, args)
        model.requires_grad_(False)
        if names2 != names:
            raise RuntimeError(f"edited parameter names changed: {names2} vs {names}")
        ours = [int(r["case_id"]) for r in forget_records]
        theirs = [int(r["case_id"]) for r in zero_records]
        if ours != theirs:
            raise RuntimeError("forget facts differ between the Δe loader and ZeroUnlearn")
    apply_seconds = time.perf_counter() - t0
    changed = {n: float((dict(model.named_parameters())[n].detach().float().cpu()
                         - saved[n].float()).norm() / saved[n].float().norm()) for n in names}
    print(f"edit={args.edit}: {apply_seconds:.0f}s, relative ||ΔW||: {changed}", flush=True)

    if args.official_eval and args.edit == "zerounlearn":
        from mcf_zero_unlearn_official_eval import evaluate_loaded_model_official

        result = evaluate_loaded_model_official(
            method="ZeroUnlearn-targettrue-parity", model=model, tok=tok,
            model_dir="in-memory:ZeroUnlearn-targettrue-parity", mcf_path=Path(args.mcf_path),
            wikidata_dir=Path(args.wikidata_dir), out_path=None, unlearn_num=FORGET_NUM,
            retain_num=RETAIN_NUM, seed=args.seed, sample_mode="official", skip_ppl=False)
        (out / "zerounlearn").mkdir(parents=True, exist_ok=True)
        (out / "zerounlearn" / "official_mcf_eval.json").write_text(
            json.dumps(result, indent=2, default=str) + "\n")
        print("official eval: forget", {k: result["forget"].get(k) for k in ("Eff", "Gen", "Spe")},
              "PPL", result.get("forget_PPL"), flush=True)

    t0 = time.perf_counter()
    zero_rows = generate_all(model, tok, views, device, args.max_new_tokens)
    print(f"edited generation: {time.perf_counter() - t0:.0f}s", flush=True)

    restore(model, saved)                            # base again, for fluency scoring
    check = generate_all(model, tok, views[:3], device, args.max_new_tokens)
    restore_ok = all(a["ids"] == b["ids"] for a, b in zip(check, base_rows))
    if not restore_ok:
        raise RuntimeError("restored model does not reproduce the base generations")
    score(model, tok, base_rows, device, routed=False)
    score(model, tok, zero_rows, device, routed=True)

    meta = {"seed": args.seed, "edit": args.edit, "max_new_tokens": args.max_new_tokens,
            "model_path": str(args.model_path), "attn_implementation": args.attn_implementation,
            "edited_parameters": names, "relative_weight_change": changed,
            "apply_seconds": apply_seconds, "restore_check_passed": restore_ok,
            "forget_case_ids": [int(r["case_id"]) for r in forget_records]}
    base_summary = {**leak_summary(base_rows), **meta, "model": "base"}
    zero_summary = {**leak_summary(zero_rows), **meta, "model": f"{args.edit}-edited"}
    write_generation(out / "base" / "routed_generation.json", base_rows, base_summary)
    write_generation(out / "zerounlearn" / "routed_generation.json", zero_rows, zero_summary)
    print(json.dumps({"base": leak_summary(base_rows), "edited": leak_summary(zero_rows)}, indent=2))
    print("\nSamples (base | edited):")
    for b, z in list(zip(base_rows, zero_rows))[:args.show]:
        print(f"- {b['prompt']!r}  answer={b['answer']!r}\n    base : {b['continuation']!r}"
              f"\n    edit : {z['continuation']!r}  leak={z['answer_in_continuation']}")


# -------------------------------------------------------------------------- compare

def _load(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else None


def _uniform(rows):
    """Recompute the leak flags for routed-generate files that predate them."""
    for r in rows:
        answer = r["answer"].casefold()
        r.setdefault("answer_in_continuation", answer in r["continuation"].casefold())
        r.setdefault("answer_at_start", r["continuation"].strip().casefold().startswith(answer))
        r.setdefault("answer_in_prompt", answer in r["prompt"].casefold())
    return rows


def _fmt(value, pct=False):
    if value is None:
        return "–"
    if pct:
        return f"{100 * value:.1f}%"
    return f"{value:.4g}" if isinstance(value, float) else str(value)


def compare(args):
    zero_dir = Path(args.zero_dir)
    sweep = Path(args.sweep_root)
    entries = [("base (this run)", zero_dir / "base" / "routed_generation.json", None)]
    if (sweep / "base" / "routed_generation.json").is_file():
        entries.append(("base (router, α-sweep run)", sweep / "base" / "routed_generation.json",
                        sweep / "base" / "linear_global" / "official_mcf_eval.json"))
    if (sweep / "reference_delta_e" / "routed_generation.json").is_file():
        ref = SEMANTIC_ROOT / "outputs" / "mcf_multiseed_regular_v1" / "seed1" / "L19" / "linear_global"
        entries.append(("Δe (setting A)", sweep / "reference_delta_e" / "routed_generation.json",
                        ref / "official_mcf_eval.json"))
    for alpha in args.alphas:
        arm = sweep / f"a{alpha}"
        if (arm / "routed_generation.json").is_file():
            entries.append((f"projection α={alpha}", arm / "routed_generation.json",
                            arm / "linear_global" / "official_mcf_eval.json"))
    zero_eval = zero_dir / "zerounlearn" / "official_mcf_eval.json"
    if not zero_eval.is_file() and args.zero_official:
        zero_eval = Path(args.zero_official)
    entries.append(("ZeroUnlearn (target_true parity)",
                    zero_dir / "zerounlearn" / "routed_generation.json", zero_eval))

    reference_ids = None
    lines = ["| model | Eff ↓ | Gen ↓ | Spe | PPL | prompts | leak@start ↓ | leak@24 ↓ "
             "| leak@24 excl. answer-in-prompt ↓ | gen NLL ↓ | repeat-3gram ↓ |",
             "|" + "---|" * 11]
    per_model = {}
    for label, gen_path, eval_path in entries:
        gen = _load(gen_path)
        if gen is None:
            print(f"skip {label}: {gen_path} missing")
            continue
        rows = _uniform(gen["examples"])
        ids = [r["id"] for r in rows]
        if reference_ids is None:
            reference_ids = ids
        elif ids != reference_ids:
            print(f"WARNING {label}: prompt set differs from the first row")
        s = leak_summary(rows)
        official = _load(eval_path) if eval_path else None
        forget = (official or {}).get("forget") or {}
        if official and official.get("metric_version") != "zerounlearn_answer_probability_v2":
            # Raw evaluate_loaded_model_official output: forget.Eff/Gen there are the legacy
            # "prefers target_true over target_new" rates. Re-score exactly as our static
            # evaluator does (Eq. 16 answer probability; Spe = neighbourhood accuracy).
            from mcf_zero_unlearn_metric_parity import summarize_probability_metrics
            forget = summarize_probability_metrics(official["forget"], official["forget_raw"])
        lines.append("| " + " | ".join([
            label, _fmt(forget.get("Eff")), _fmt(forget.get("Gen")), _fmt(forget.get("Spe")),
            _fmt((official or {}).get("forget_PPL")), str(s["prompts"]),
            _fmt(s["leak_start"], True), _fmt(s["leak_any"], True),
            _fmt(s["leak_any_excl"], True) + f" (n={s['prompts_answer_not_in_prompt']})",
            _fmt(s["base_nll_per_token_mean"]), _fmt(s["repeated_trigram"], True)]) + " |")
        per_model[label] = {r["id"]: r for r in rows}

    text = (f"# Free-generation leak on MCF seed {args.seed} forget prompts "
            "(canonical + development views)\n\n"
            "Eff/Gen/Spe/PPL: official MCF eval (Eff/Gen = Eq. 16 complete-answer probability %, "
            "lower is better). leak@start: greedy continuation begins with the true answer; "
            "leak@24: true answer anywhere in the 24-token greedy continuation; excl.: prompts "
            "whose own text does not already contain the answer. Δe/projection are only active "
            "where the router fires; ZeroUnlearn edits weights, so it is active everywhere. "
            "gen NLL = base-model NLL per continuation token (fluency).\n\n"
            + "\n".join(lines) + "\n")
    zero = per_model.get("ZeroUnlearn (target_true parity)")
    base = per_model.get("base (this run)")
    other = per_model.get("base (router, α-sweep run)")
    if base and other:
        same = sum(base[i]["continuation"] == other[i]["continuation"] for i in base if i in other)
        text += (f"\nSanity: the two base runs give identical continuations on "
                 f"{same}/{len(base)} prompts.\n")
    if zero and base:
        leaks = [i for i in zero if zero[i]["answer_in_continuation"]
                 and not zero[i]["answer_in_prompt"]][:args.show]
        text += "\n## ZeroUnlearn continuations that still contain the answer (answer not in prompt)\n\n"
        for i in leaks:
            text += (f"- **{zero[i]['prompt']}** → answer *{zero[i]['answer']}*\n"
                     f"  - base: `{base[i]['continuation'].strip()}`\n"
                     f"  - ZeroUnlearn: `{zero[i]['continuation'].strip()}`\n")
    (zero_dir / "comparison.md").write_text(text)
    print(text)


def _defaults():
    """Default paths without importing the (heavy) ZeroUnlearn runner."""
    model = ("/scratch/yl258/kp759/hf/models--meta-llama--Llama-3.2-3B-Instruct/snapshots/"
             "0cb88a4f764b7a12671c53f0838cd831a0843b95")
    zero_root = SEMANTIC_ROOT.parent / "ZeroUnlearn"
    return {
        "model": model,
        "zero_root": str(zero_root),
        "hparams": str(zero_root / "hparams" / "ZeroUnlearn" / "Llama-3.2-3B-Instruct.json"),
        "mcf": str(SEMANTIC_ROOT / "data" / "multi_counterfact.json"),
        "wikidata": str(SEMANTIC_ROOT / "data" / "wikidata"),
    }


def main(argv=None):
    d = _defaults()
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--output-dir", required=True)
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--model-path", default=d["model"])
    r.add_argument("--zero-unlearn-root", default=d["zero_root"])
    r.add_argument("--hparams-path", default=d["hparams"])
    r.add_argument("--mcf-path", default=d["mcf"])
    r.add_argument("--wikidata-dir", default=d["wikidata"])
    r.add_argument("--edit", choices=("zerounlearn", "none"), default="zerounlearn",
                   help="'none' = no edit (smoke test of the generation/scoring path)")
    r.add_argument("--official-eval", action="store_true",
                   help="also run the official MCF eval on the edited model (~extra 20-40 min)")
    r.add_argument("--max-new-tokens", type=int, default=24)
    r.add_argument("--attn-implementation", default="eager")
    r.add_argument("--device", default="cuda")
    r.add_argument("--show", type=int, default=8)
    r.add_argument("--local-files-only", action="store_true")
    c = sub.add_parser("compare")
    c.add_argument("--zero-dir", required=True)
    c.add_argument("--sweep-root", default=str(SEMANTIC_ROOT / "outputs" / "mcf_projection_alpha_sweep_seed1"))
    c.add_argument("--alphas", nargs="*", default=["2.5", "3"])
    c.add_argument("--zero-official", default=None,
                   help="ZeroUnlearn official eval JSON if this run skipped --official-eval")
    c.add_argument("--seed", type=int, default=1)
    c.add_argument("--show", type=int, default=8)
    args = parser.parse_args(argv)
    if args.command == "run":
        run(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()
