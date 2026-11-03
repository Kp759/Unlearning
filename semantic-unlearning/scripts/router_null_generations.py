#!/usr/bin/env python3
"""Base vs ours greedy outputs on the no-association prompts the router fires on, for an
existing router_diag cache, without rebuilding it.

cache_router_states.py --ours-dir does the same while it builds a cache. This reruns only that
step with another set of rows, e.g. the paper's method (outputs/l16_ours_v1: router at 19, Δe
at 16) for a cache built when the generations came from outputs/onepass_l19_v1 (Δe at 19). The
routes are the cached ones (router_null.pt), so only the rows differ. The result replaces
manifest["null_generations"]; re-run router_diag_report.py afterwards to put it in the report.

    python -u scripts/router_null_generations.py \
        --cache-dir outputs/router_diag_ours_control_v1/mcf/seed1 --device cuda --local-files-only
    # rows default to outputs/l16_ours_v1/<dataset>/seed<seed>/ours
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cache_router_states import null_generations, probe_ids


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--ours-dir", default=None,
                    help="run dir with fact_association_embeddings.pt (default: outputs/l16_ours_v1/<ds>/seed<seed>/ours)")
    ap.add_argument("--max-null-generations", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--dtype", default="bfloat16", help="what every official evaluator runs")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--local-files-only", action="store_true")
    args = ap.parse_args(argv)

    cache = Path(args.cache_dir)
    manifest = json.loads((cache / "manifest.json").read_text())
    ours = Path(args.ours_dir or f"outputs/l16_ours_v1/{manifest['dataset']}/seed{manifest['seed']}/ours")
    if not (ours / "fact_association_embeddings.pt").is_file():
        raise SystemExit(f"no finished ours run at {ours}")
    artifact = torch.load(ours / "fact_association_embeddings.pt", map_location="cpu", weights_only=False)
    router = torch.load(Path(manifest["router_dir"]) / "fact_association_embeddings.pt", map_location="cpu",
                        weights_only=False)
    if not torch.equal(torch.as_tensor(artifact["router_weight"]), torch.as_tensor(router["router_weight"])):
        raise SystemExit(f"{ours} routes with other heads than the cached router {manifest['router_dir']}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(manifest["model_path"], use_fast=True, local_files_only=args.local_files_only)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        manifest["model_path"], dtype=getattr(torch, args.dtype), local_files_only=args.local_files_only,
        attn_implementation="eager").to(args.device).eval()
    model.requires_grad_(False)
    probes = json.loads((cache / "probes_null.json").read_text())
    routes = torch.load(cache / "router_null.pt", weights_only=False)
    ids = [probe_ids(tok, p) for p in probes]
    result = null_generations(model, tok, ours, probes, ids, routes, args.max_null_generations, args.max_new_tokens)
    result["write_layer"] = (artifact.get("early_write") or {}).get("layer", artifact.get("layer"))
    manifest["null_generations"] = result
    tmp = cache / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    tmp.replace(cache / "manifest.json")
    print(json.dumps({"phase": "null_generations", "cache": str(cache), "ours_dir": str(ours),
                      **{k: v for k, v in result.items() if k != "rows"}}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
