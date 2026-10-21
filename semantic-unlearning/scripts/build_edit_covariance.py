#!/usr/bin/env python3
"""Covariance statistics for the locate-and-edit baselines (ZeroUnlearn, ROME, MEMIT) from a real
Wikipedia corpus, computed once in a single forward pass and written exactly where and how the
vendored ZeroUnlearn code caches them, so every baseline task loads them instead of recomputing.

Why: the vendored loaders read data/wikidata, the 200-sentence PPL fixture (~3.5k tokens). The
down_proj input of Llama-3.2-3B is 8192-d, so that covariance is rank-deficient: ROME inverts it,
MEMIT solves (lambda C + K K^T)^-1, and the first ROME / MEMIT runs collapsed (PPL 1e5-1e6).

Per layer L (block L's mlp.down_proj), under STATS/<model>/<dataset-name>_stats/:
  model.layers.L.mlp.down_proj_float32_mom2_<N>.npz   CombinedStat(mom2=SecondMoment()) of the
        down_proj inputs K: ROME compute_u, MEMIT get_cov, ZeroUnlearn layer_stats_retain
  vkt_model.layers.L.mlp.down_proj_float32_<N>.npz    VKT() of (down_proj output, input):
        ZeroUnlearn compute_vkt_cross_covariance
<model> = config._name_or_path.rsplit('/')[-1] and STATS = semantic-unlearning/data/stats, as at
edit time; N = the hparams' mom2_n_samples (100000). The sample is the one tally(sample_size=N,
random_sample=1) draws (FixedRandomSubsetSampler, seed 1); texts are tokenized by
rome.tok_dataset.TokenizedDataset with maxlen 4096 (as ZeroUnlearn's loader). Documents are packed
into batches of <= --batch-tokens tokens (sums over tokens do not depend on batching), the
forward runs in bfloat16 and stops after the last needed layer, and accumulation is float32
(mom2_dtype).

    python -u scripts/build_edit_covariance.py --text-dir /project/yl258/kp759/datasets/wikipedia_20231101_100k \\
        --dataset-name wikipedia_20231101_100k --layers 16 17 18 --local-files-only
Then run the baselines with --mom2-text <text-dir> --mom2-dataset <dataset-name>
(run_l16_paper_one.sh: BASELINE_MOM2_TEXT / BASELINE_MOM2_DATASET).
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import sys
import time

import torch

SEMANTIC_ROOT = Path(__file__).resolve().parents[1]
ZERO_ROOT = SEMANTIC_ROOT.parent / "ZeroUnlearn"
DEFAULT_REF = SEMANTIC_ROOT / "outputs" / "mcf_multiseed_regular_v1" / "seed1" / "L19" / "router" / "association_manifest.json"


@contextlib.contextmanager
def _cwd(path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def vendored():
    """util.runningstats / rome.tok_dataset from ZeroUnlearn/ (globals.yml is read from its cwd)."""
    root = str(ZERO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    with _cwd(ZERO_ROOT):
        import util.runningstats as rs
        from rome.tok_dataset import TokenizedDataset, dict_to_, flatten_masked_batch, length_collation
    return rs, TokenizedDataset, dict_to_, flatten_masked_batch, length_collation


def cache_paths(stats_dir, model_name, dataset_name, layer, n, template="model.layers.{}.mlp.down_proj"):
    name = template.format(layer)
    base = Path(stats_dir) / model_name / f"{dataset_name}_stats"
    return base / f"{name}_float32_mom2_{n}.npz", base / f"vkt_{name}_float32_{n}.npz"


class _Stop(Exception):
    pass


def build(model, tok, texts, layers, *, stats_dir, dataset_name, sample_size=100000, max_len=4096,
          batch_tokens=12288, batch_size=16, device="cuda", force=False, num_workers=2, log_every=500):
    rs, TokenizedDataset, dict_to_, flatten_masked_batch, length_collation = vendored()
    model_name = str(model.config._name_or_path).rsplit("/")[-1]
    todo = []
    for L in layers:
        k_path, v_path = cache_paths(stats_dir, model_name, dataset_name, L, sample_size)
        if force or not (k_path.is_file() and v_path.is_file()):
            todo.append(L)
    report = {"model_name": model_name, "dataset_name": dataset_name, "sample_size": sample_size,
              "layers": list(layers), "computed": list(todo), "skipped_existing": sorted(set(layers) - set(todo))}
    if not todo:
        return report
    ds = TokenizedDataset([{"text": t} for t in texts], tokenizer=tok, maxlen=max_len)
    loader = rs.make_loader(ds, sample_size=sample_size, random_sample=1, batch_size=batch_size,
                            collate_fn=length_collation(batch_tokens), pin_memory=(device != "cpu"),
                            num_workers=num_workers)
    K = {L: rs.CombinedStat(mom2=rs.SecondMoment()) for L in todo}
    V = {L: rs.VKT() for L in todo}
    buf, handles, last = {}, [], max(todo)

    def hook(L):
        def fn(module, inputs, output):
            buf[L] = (inputs[0], output)
            if L == last:
                raise _Stop()
        return fn

    for L in todo:
        handles.append(model.model.layers[L].mlp.down_proj.register_forward_hook(hook(L)))
    t0, tokens, batches = time.time(), 0, 0
    try:
        with torch.no_grad():
            for group in loader:
                for batch in group:
                    batch = dict_to_(batch, device)
                    try:
                        model(**batch)
                    except _Stop:
                        pass
                    mask = batch["attention_mask"]
                    for L in todo:
                        k = flatten_masked_batch(buf[L][0], mask).float()
                        v = flatten_masked_batch(buf[L][1], mask).float()
                        K[L].add(k)
                        V[L].add(v, k)
                    tokens += int(mask.sum())
                    batches += 1
                    buf.clear()
                    if log_every and batches % log_every == 0:
                        print(json.dumps({"phase": "progress", "batches": batches, "tokens": tokens,
                                          "seconds": round(time.time() - t0, 1)}), flush=True)
    finally:
        for h in handles:
            h.remove()
    for L in todo:
        k_path, v_path = cache_paths(stats_dir, model_name, dataset_name, L, sample_size)
        for stat, path in ((K[L], k_path), (V[L], v_path)):
            stat.to_("cpu")
            tmp = path.with_name(path.stem + ".tmp.npz")
            rs.save_cached_state(str(tmp), stat, {"sample_size": sample_size})
            os.replace(tmp, path)
    report.update({"tokens": tokens, "batches": batches, "seconds": round(time.time() - t0, 1),
                   "documents": min(sample_size, len(texts)),
                   "files": [str(p) for L in todo for p in cache_paths(stats_dir, model_name, dataset_name, L, sample_size)]})
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--text-dir", required=True, help="DatasetDict with a 'train' split and a 'text' field")
    p.add_argument("--dataset-name", required=True, help="the baselines' --mom2-dataset (cache sub-directory)")
    p.add_argument("--model-path", default=None, help="default: model_path of the MCF seed-1 L19 router")
    p.add_argument("--layers", type=int, nargs="+", default=[16, 17, 18])
    p.add_argument("--sample-size", type=int, default=100000, help="= hparams mom2_n_samples")
    p.add_argument("--max-len", type=int, default=4096)
    p.add_argument("--batch-tokens", type=int, default=12288)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--stats-dir", default=str(SEMANTIC_ROOT / "data" / "stats"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--force", action="store_true")
    p.add_argument("--local-files-only", action="store_true")
    args = p.parse_args(argv)

    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = args.model_path or json.loads(DEFAULT_REF.read_text())["model_path"]
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True, local_files_only=args.local_files_only)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16 if args.device != "cpu" else torch.float32,
        local_files_only=args.local_files_only).to(args.device).eval()
    model.requires_grad_(False)
    texts = load_from_disk(args.text_dir)["train"]["text"]
    texts = [str(t) for t in texts]
    print(json.dumps({"phase": "start", "model_path": model_path, "documents": len(texts),
                      "layers": args.layers, "sample_size": args.sample_size}), flush=True)
    report = build(model, tok, texts, args.layers, stats_dir=args.stats_dir, dataset_name=args.dataset_name,
                   sample_size=args.sample_size, max_len=args.max_len, batch_tokens=args.batch_tokens,
                   batch_size=args.batch_size, device=args.device, force=args.force)
    report["text_dir"] = str(args.text_dir)
    out = Path(args.stats_dir) / report["model_name"] / f"{args.dataset_name}_stats" / "build_report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"phase": "done", **report}), flush=True)


if __name__ == "__main__":
    main()
