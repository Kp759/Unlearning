#!/usr/bin/env python3
"""Cache last-token hidden states at every layer, plus the L19 router's outputs.

One forward pass per probe on the frozen base model (no bank attached), with a hook on
every decoder block that keeps the block output at the prompt's last token: the exact
tensor the router reads (`linear_router.LinearClassifierAssociationBank._hook`: block
`layer` output at prefix_length - 1). Everything downstream (scripts/router_diag_report.py,
scripts/router_layer_sweep_cached.py) runs on CPU from this cache.

    python -u scripts/cache_router_states.py --dataset mcf --seed 1 --device cuda --local-files-only
    # -> outputs/router_diag_v1/mcf/seed1/

Sections (scripts/router_diag_probes.py):
    eval       every prompt the official evaluator routes (forget / paraphrase /
               neighborhood / retain), with genie ground truth           -> eval dtype
    null       prompts with no association: generic text, PPL prefixes, unseen
               subjects in the forget templates, protected subjects asked about
               other relations                                           -> eval dtype
    perturbed  typo / case / partial-name / lead-in variants of the forget and
               paraphrase prompts (test only)                            -> eval dtype
    train      the router's own fit / calibration / audit prompts       -> train dtype
    aug        training-side augmentation and hard negatives            -> train dtype
eval dtype defaults to bfloat16 (what every official evaluator runs), train dtype to
float32 (what fit_linear_router.py extracted features in), so the cached L19 routes are
the evaluation's routes and a refit at L19 is the shipped fit.

Files in --out-dir:
    probes_<section>.json   the probes (prompt, split, group, should_fire, true_facts, ...)
    states_<section>.pt     {"layers": [...], "states": [L, P, d] in the compute dtype}
    router_<section>.pt     L19 router on the cached states: deployed logits [P, N],
                            stage-1 logits (before the calibrated bias), subject
                            eligibility [P, N], and the runtime decision (decide_routes)
    manifest.json           provenance, counts, hook parity, router-pass timing,
                            optional generations on fired null prompts
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import torch

from linear_router import decide_routes, is_linear_router_artifact, load_router_artifact, score_queries
import router_diag_probes as probes_lib

DEFAULT_ROUTER_DIRS = {
    "mcf": "outputs/mcf_multiseed_regular_v1/seed{seed}/L19/router",
    "zsre": "outputs/zsre_multiseed_reworded_v2/seed{seed}/L19/router",
    "mquake": "outputs/mquake_multiseed_regular_v1/seed{seed}/L19/router",
    "rwku": "outputs/rwku_multiseed_subject_v1/seed{seed}/L19/router",
}
EVAL_SECTIONS = ("eval", "null", "perturbed")
TRAIN_SECTIONS = ("train", "aug")
IDK_MARKERS = ("i don't know", "i do not know", "not sure", "unknown", "i cannot", "i can't",
               "no information", "i'm not able", "i am not able", "unable to")


class _Stop(Exception):
    pass


def probe_ids(tok, probe):
    if probe["input_ids"] is not None:
        return list(probe["input_ids"])
    return [int(t) for t in tok(probe["prompt"])["input_ids"]]


def eligibility_from_ids(ids_list, subject_patterns):
    """[P, N] bool, the runtime rule: a complete subject-token pattern occurs in the prompt."""
    widths = sorted({len(p) for patterns in subject_patterns for p in patterns})
    out = torch.zeros((len(ids_list), len(subject_patterns)), dtype=torch.bool)
    for row, ids in enumerate(ids_list):
        grams = {tuple(ids[i:i + w]) for w in widths for i in range(len(ids) - w + 1)}
        for fact, patterns in enumerate(subject_patterns):
            if any(tuple(p) in grams for p in patterns):
                out[row, fact] = True
    return out


@torch.no_grad()
def capture_last_token_states(model, ids_list, layers, *, batch_size, pad_id, store_dtype=None):
    """[len(layers), P, d] block outputs at each prompt's last token (right padding)."""
    device = next(model.parameters()).device
    blocks = model.model.layers
    layers = [int(layer) for layer in layers]
    stop_at = max(layers)
    hidden = model.config.hidden_size
    store_dtype = store_dtype or next(model.parameters()).dtype
    out = torch.empty((len(layers), len(ids_list), hidden), dtype=store_dtype)
    order = sorted(range(len(ids_list)), key=lambda i: len(ids_list[i]))
    slot = {layer: k for k, layer in enumerate(layers)}
    for start in range(0, len(order), int(batch_size)):
        batch = order[start:start + int(batch_size)]
        width = max(len(ids_list[i]) for i in batch)
        input_ids = torch.full((len(batch), width), int(pad_id), dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for r, i in enumerate(batch):
            input_ids[r, :len(ids_list[i])] = torch.tensor(ids_list[i])
            mask[r, :len(ids_list[i])] = 1
        last = (mask.sum(dim=1) - 1).to(device)
        rows = torch.arange(len(batch), device=device)
        index = torch.tensor(batch)
        captured = {}

        def make_hook(layer):
            def hook(_module, _args, output):
                h = output[0] if isinstance(output, tuple) else output
                captured[layer] = h[rows, last].detach()
                if layer == stop_at:
                    raise _Stop
            return hook

        handles = [blocks[layer].register_forward_hook(make_hook(layer)) for layer in layers]
        try:
            model(input_ids=input_ids.to(device), attention_mask=mask.to(device), use_cache=False)
        except _Stop:
            pass
        finally:
            for handle in handles:
                handle.remove()
        for layer in layers:
            out[slot[layer], index] = captured[layer].to("cpu", store_dtype)
    return out


def router_logits(states_l, artifact):
    """Deployed logits [P, N] (and stage-1 logits) from block-`layer` states, as the hook."""
    def score(bias):
        logits = score_queries(states_l.float(), artifact["router_weight"], bias,
                               artifact["feature_mean"], artifact.get("feature_components"))
        head_index = artifact.get("head_index")
        return logits if head_index is None else logits[:, torch.as_tensor(head_index)]

    deployed = score(artifact["router_bias"])
    calibration = artifact.get("bias_calibration") or {}
    stage1 = None if calibration.get("stage1_bias") is None else score(calibration["stage1_bias"])
    return deployed, stage1


def runtime_rule(artifact):
    gate = str(artifact.get("gate_mode", "threshold"))
    if gate == "subject":
        return float("-inf"), 0.0
    per_head = artifact.get("per_head_thresholds")
    threshold = torch.as_tensor(per_head).float() if per_head is not None else float(artifact["threshold"])
    return threshold, float(artifact.get("ambiguity_margin", 0.5))


def route_section(states_l, ids_list, artifact):
    logits, stage1 = router_logits(states_l, artifact)
    eligible = eligibility_from_ids(ids_list, artifact["subject_patterns"])
    threshold, margin = runtime_rule(artifact)
    decision = decide_routes(logits, eligible, threshold, margin)
    return {"logits": logits, "stage1_logits": stage1, "eligible": eligible,
            **{k: v.cpu() for k, v in decision.items()}}


@torch.no_grad()
def hook_parity(model, artifact, tok, ids_list, offline, sample, batch_size, pad_id, seed=0):
    """Route a sample through the REAL bank hook, batched (as cached) and one at a time
    unpadded (as the RWKU evaluator), and compare with the offline decision."""
    device = next(model.parameters()).device
    lm, bank = load_router_artifact(model, artifact)
    g = torch.Generator().manual_seed(seed)
    picked = torch.randperm(len(ids_list), generator=g)[:int(sample)].tolist()
    report = {}
    try:
        for mode, size in (("batched", int(batch_size)), ("single", 1)):
            mismatches = []
            for start in range(0, len(picked), size):
                rows = picked[start:start + size]
                width = max(len(ids_list[i]) for i in rows)
                input_ids = torch.full((len(rows), width), int(pad_id), dtype=torch.long)
                mask = torch.zeros_like(input_ids)
                for r, i in enumerate(rows):
                    input_ids[r, :len(ids_list[i])] = torch.tensor(ids_list[i])
                    mask[r, :len(ids_list[i])] = 1
                lm(input_ids=input_ids.to(device), attention_mask=mask.to(device), use_cache=False)
                for r, i in enumerate(rows):
                    hook = bank.last_active_fact_indices[r]
                    expected = [int(offline["fact"][i])] if bool(offline["active"][i]) else []
                    if hook != expected:
                        mismatches.append({"index": i, "hook": hook, "offline": expected})
            report[mode] = {"prompts": len(picked), "route_mismatches": len(mismatches),
                            "examples": mismatches[:10]}
    finally:
        bank.close()
    return report


@torch.no_grad()
def time_router_pass(model, ids_list, layers, batch_size, pad_id, repeats=3):
    """ms per prompt: full forward vs forward stopped after block L (the routing pass only
    needs the router layer; scripts/early_write.py currently runs it in full)."""
    device = next(model.parameters()).device
    sample = ids_list[:min(len(ids_list), 8 * int(batch_size))]
    out = {}
    for name, stop in [("full_forward", None)] + [(f"stop_after_L{l}", int(l)) for l in layers]:
        best = math.inf
        for _ in range(int(repeats)):
            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            if stop is None:
                for s in range(0, len(sample), int(batch_size)):
                    chunk = sample[s:s + int(batch_size)]
                    width = max(len(x) for x in chunk)
                    ids = torch.full((len(chunk), width), int(pad_id), dtype=torch.long)
                    mask = torch.zeros_like(ids)
                    for r, x in enumerate(chunk):
                        ids[r, :len(x)] = torch.tensor(x)
                        mask[r, :len(x)] = 1
                    model(input_ids=ids.to(device), attention_mask=mask.to(device), use_cache=False)
            else:
                capture_last_token_states(model, sample, [stop], batch_size=batch_size, pad_id=pad_id)
            if device.type == "cuda":
                torch.cuda.synchronize()
            best = min(best, time.perf_counter() - started)
        out[name] = 1000.0 * best / len(sample)
    full = out["full_forward"]
    out["fraction_of_full"] = {k: v / full for k, v in out.items() if k != "full_forward"}
    out["blocks"] = len(model.model.layers)
    out["prompts_timed"] = len(sample)
    return out


@torch.no_grad()
def null_generations(base_model, tok, ours_dir, null_probes, ids_list, routes, max_prompts,
                     max_new_tokens):
    """Base vs ours greedy outputs on the null prompts the router fires on."""
    fired = [i for i in range(len(null_probes)) if bool(routes["active"][i])
             and not null_probes[i]["should_fire"]][:int(max_prompts)]
    if not fired:
        return {"fired_null_prompts": 0, "rows": []}
    device = next(base_model.parameters()).device
    eos = base_model.config.eos_token_id
    eos_ids = set(eos) if isinstance(eos, (list, tuple)) else {eos}

    def greedy(lm, ids):
        x = torch.tensor([ids], device=device)
        if hasattr(lm, "generate_uncached_fixed_boundary"):
            out = lm.generate_uncached_fixed_boundary(x, torch.ones_like(x),
                                                      max_new_tokens=max_new_tokens)
        else:
            out = x
            for _ in range(int(max_new_tokens)):
                nxt = lm(input_ids=out, use_cache=False).logits[:, -1].argmax(-1, keepdim=True)
                out = torch.cat([out, nxt], dim=1)
                if int(nxt) in eos_ids:
                    break
        return tok.decode(out[0, len(ids):], skip_special_tokens=True)

    base = {i: greedy(base_model, ids_list[i]) for i in fired}
    artifact = torch.load(Path(ours_dir) / "fact_association_embeddings.pt", map_location="cpu",
                          weights_only=False)
    lm, bank = load_router_artifact(base_model, artifact)
    try:
        rows = []
        for i in fired:
            edited = greedy(lm, ids_list[i])
            rows.append({"index": i, "group": null_probes[i]["group"],
                         "prompt": null_probes[i]["prompt"], "base": base[i], "ours": edited,
                         "changed": edited.strip() != base[i].strip(),
                         "idk_like": any(m in edited.casefold() for m in IDK_MARKERS)})
    finally:
        if hasattr(bank, "close"):
            bank.close()
    return {"fired_null_prompts": len(fired), "ours_dir": str(ours_dir),
            "changed": sum(r["changed"] for r in rows), "idk_like": sum(r["idk_like"] for r in rows),
            "rows": rows}


def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def _counts(probes):
    out = {}
    for p in probes:
        key = f"{p['split']}/{p['group']}"
        row = out.setdefault(key, {"n": 0, "should_fire": 0})
        row["n"] += 1
        row["should_fire"] += int(p["should_fire"])
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=sorted(DEFAULT_ROUTER_DIRS))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--router-dir", default=None, help="default: the paper's L19 router")
    parser.add_argument("--out-dir", default=None, help="default: outputs/router_diag_v1/<ds>/seed<S>")
    parser.add_argument("--model-path", default=None, help="default: the router manifest's")
    parser.add_argument("--mcf-path", default=None)
    parser.add_argument("--zsre-path", default=probes_lib.DEFAULT_PATHS["zsre"])
    parser.add_argument("--mquake-path", default=probes_lib.DEFAULT_PATHS["mquake"])
    parser.add_argument("--rwku-root", default=probes_lib.DEFAULT_PATHS["rwku"])
    parser.add_argument("--wikidata-dir", default=probes_lib.DEFAULT_PATHS["wikidata"])
    parser.add_argument("--sections", default=",".join(probes_lib.SECTIONS))
    parser.add_argument("--layers", default="all", help="'all' or comma-separated block indices")
    parser.add_argument("--retain-num", type=int, default=probes_lib.OFFICIAL_RETAIN_NUM,
                        help="first N official retain records (default: all 1000)")
    parser.add_argument("--null-per-fact", type=int, default=3)
    parser.add_argument("--perturb-per-probe", type=int, default=3)
    parser.add_argument("--aug-per-probe", type=int, default=2)
    parser.add_argument("--hardneg-per-fact", type=int, default=3)
    parser.add_argument("--no-ppl", action="store_true")
    parser.add_argument("--eval-dtype", default="bfloat16", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--train-dtype", default="float32", choices=("bfloat16", "float16", "float32"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--parity-sample", type=int, default=256,
                        help="eval probes routed through the real hook (0 = skip)")
    parser.add_argument("--time-layers", default="19,26", help="router-pass timing ('' = skip)")
    parser.add_argument("--ours-dir", default=None,
                        help="an ours run dir: generate base vs ours on the fired null prompts")
    parser.add_argument("--max-null-generations", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)

    router_dir = Path(args.router_dir or DEFAULT_ROUTER_DIRS[args.dataset].format(seed=args.seed)).resolve()
    out = Path(args.out_dir or f"outputs/router_diag_v1/{args.dataset}/seed{args.seed}").resolve()
    if (out / "manifest.json").exists() and not args.overwrite:
        print(f"{out}/manifest.json exists (--overwrite to redo)")
        return 0
    out.mkdir(parents=True, exist_ok=True)
    artifact = torch.load(router_dir / "fact_association_embeddings.pt", map_location="cpu",
                          weights_only=False)
    manifest = json.loads((router_dir / "association_manifest.json").read_text())
    if not is_linear_router_artifact(artifact) or artifact.get("early_write"):
        raise ValueError(f"{router_dir} is not a linear-router dir (use the L19 router dir)")
    if artifact.get("learned_subject") is not None:
        raise ValueError("hybrid (learned subject) routers are not supported: eligibility "
                         "here is the exact subject-token rule")
    sections = [s for s in args.sections.split(",") if s]
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = Path(args.model_path or manifest["model_path"]).resolve()
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=args.local_files_only)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    first_dtype = args.train_dtype if any(s in TRAIN_SECTIONS for s in sections) else args.eval_dtype
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=getattr(torch, first_dtype), local_files_only=args.local_files_only,
        attn_implementation="eager").to(args.device).eval()
    model.requires_grad_(False)
    n_layers = len(model.model.layers)
    layers = list(range(n_layers)) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    router_layer = int(artifact["layer"])
    if router_layer not in layers:
        layers = sorted(set(layers) | {router_layer})

    from mcf_zero_unlearn_official_eval import is_llama_like

    paths = {"mcf": args.mcf_path or manifest.get("mcf_path") or probes_lib.DEFAULT_PATHS["mcf"],
             "zsre": args.zsre_path, "mquake": args.mquake_path, "rwku": args.rwku_root,
             "wikidata": None if args.no_ppl else args.wikidata_dir}
    started = time.time()
    built, notes = probes_lib.build_all(
        args.dataset, tok=tok, artifact=artifact, router_dir=router_dir, seed=args.seed,
        paths=paths, retain_num=args.retain_num, llama_like=is_llama_like(model, tok),
        sections=sections, null_per_fact=args.null_per_fact,
        perturb_per_probe=args.perturb_per_probe, aug_per_probe=args.aug_per_probe,
        hardneg_per_fact=args.hardneg_per_fact, with_ppl=not args.no_ppl)
    print(json.dumps({"phase": "probes_built", "seconds": round(time.time() - started, 1),
                      **{s: len(v) for s, v in built.items()}}), flush=True)

    report = {"dataset": args.dataset, "seed": args.seed, "router_dir": str(router_dir),
              "model_path": str(model_path), "router_layer": router_layer, "layers": layers,
              "n_layers": n_layers, "gate_mode": str(artifact.get("gate_mode", "threshold")),
              "threshold": artifact.get("threshold"),
              "ambiguity_margin": float(artifact.get("ambiguity_margin", 0.5)),
              "router_fit": artifact.get("router_fit", {}),
              "facts": [{"id": f["id"], "subject": f.get("subject"), "relation": f.get("relation")}
                        for f in artifact["facts"]],
              "subject_patterns": artifact["subject_patterns"],
              "notes": notes, "git_commit": _git_commit(), "command": sys.argv, "sections": {}}
    ids_by_section = {}
    order = [s for s in TRAIN_SECTIONS if s in built] + [s for s in EVAL_SECTIONS if s in built]
    for section in order:
        dtype = args.train_dtype if section in TRAIN_SECTIONS else args.eval_dtype
        model.to(getattr(torch, dtype))
        section_probes = built[section]
        ids = [probe_ids(tok, p) for p in section_probes]
        for p, x in zip(section_probes, ids):
            p["n_tokens"] = len(x)
        ids_by_section[section] = ids
        t0 = time.time()
        states = capture_last_token_states(model, ids, layers, batch_size=args.batch_size,
                                           pad_id=tok.pad_token_id)
        routes = route_section(states[layers.index(router_layer)], ids, artifact)
        torch.save({"layers": layers, "states": states, "dtype": dtype}, out / f"states_{section}.pt")
        torch.save(routes, out / f"router_{section}.pt")
        (out / f"probes_{section}.json").write_text(json.dumps(section_probes) + "\n")
        report["sections"][section] = {"probes": len(section_probes), "dtype": dtype,
                                       "seconds": round(time.time() - t0, 1),
                                       "fired": int(routes["active"].sum()),
                                       "counts": _counts(section_probes)}
        print(json.dumps({"phase": "cached", "section": section, "probes": len(section_probes),
                          "fired": int(routes["active"].sum()),
                          "seconds": report["sections"][section]["seconds"]}), flush=True)
        if section == "eval" and args.parity_sample:
            report["hook_parity"] = hook_parity(model, artifact, tok, ids, routes, args.parity_sample,
                                                args.batch_size, tok.pad_token_id)
            print(json.dumps({"phase": "hook_parity", **{k: v["route_mismatches"] for k, v in
                                                       report["hook_parity"].items()}}), flush=True)
        del states
    model.to(getattr(torch, args.eval_dtype))
    if args.time_layers and "eval" in ids_by_section:
        report["router_pass_timing_ms_per_prompt"] = time_router_pass(
            model, ids_by_section["eval"], [int(x) for x in args.time_layers.split(",")],
            args.batch_size, tok.pad_token_id)
    if args.ours_dir and "null" in built:
        routes = torch.load(out / "router_null.pt", weights_only=False)
        report["null_generations"] = null_generations(
            model, tok, args.ours_dir, built["null"], ids_by_section["null"], routes,
            args.max_null_generations, args.max_new_tokens)
    report["seconds"] = round(time.time() - started, 1)
    (out / "manifest.json").write_text(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({"phase": "done", "out_dir": str(out), "seconds": report["seconds"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
