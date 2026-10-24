"""Deletion requests that arrive over time (the dynamic setting).

The split's requests are cut into T batches. After batch t the method has seen batches
1..t (through Method.unlearn_sequential) and is scored on:
    forget  direct / paraphrase / question probes of every batch so far (must stay gone)
    retain  every retain probe (locality must not erode as edits accumulate)
Reported per step and as the mean over steps (area under the curve).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import metrics as M
from .methods import make
from .methods.base import UnlearnContext
from .model_api import ModelHandle
from .runner import EvalConfig, base_reference, renderer_for, score_core
from .schema import Split

SEQ_FAMILIES = ("direct", "paraphrase", "question")


def run_sequential(split: Split, method_name: str, method_args: Dict[str, Any], model_loader: Callable,
                   cfg: EvalConfig, out_dir: Path, steps: int = 5, cache_dir: Optional[Path] = None, log=print) -> Dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hf, tok = model_loader()
    handle = ModelHandle(model=hf, tokenizer=tok, hf=hf)
    renderer = renderer_for(split, tok)
    ref = base_reference(split, handle, renderer, cfg, cache_dir=cache_dir, log=log)
    method = make(method_name, **method_args)
    reqs = list(split.requests)
    size = max(1, -(-len(reqs) // steps))
    batches = [reqs[i:i + size] for i in range(0, len(reqs), size)]
    history: List = []
    curve = []
    for t, batch in enumerate(batches, 1):
        ctx = UnlearnContext(seed=split.seed + t, renderer=renderer,
                             generic_prompts=split.manifest.get("generic_prompts", ()),
                             max_seconds=cfg.max_unlearn_seconds, log=log)
        t0 = time.time()
        handle = method.unlearn_sequential(handle, batch, list(history), split.allowed_retain, ctx)
        secs = time.time() - t0
        history += batch
        seen = {r.request_id for r in history}
        probes = [p for p in split.probes if (p.role == "retain") or
                  (p.family in SEQ_FAMILIES and p.request_id in seen)]
        core = score_core(split, handle, renderer, cfg, ref, probes=probes, gen=False)
        agg = M.aggregates(core["families_tf"], None, None, min_n=3)
        fam = core["families_tf"]
        direct = fam.get("forget/direct", {}).get("score")
        curve.append({"step": t, "requests_so_far": len(history), "unlearn_seconds": secs,
                      "E": agg["E"], "L": agg["L"], "residual_direct": direct, "SCALPEL_EL": M.harmonic(
                          [x for x in (agg["E"], agg["L"]) if x is not None]) if agg["E"] is not None else None})
        log(f"[seq] step {t}/{len(batches)}: E={agg['E']} L={agg['L']}")
    vals = [c["SCALPEL_EL"] for c in curve if c["SCALPEL_EL"] is not None]
    res = {"split": split.name, "method": method.card.to_dict(), "method_args": method_args, "steps": len(batches),
           "curve": curve, "auc_SCALPEL_EL": sum(vals) / len(vals) if vals else None,
           "final_L": curve[-1]["L"] if curve else None}
    (out_dir / "sequential.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    return res
