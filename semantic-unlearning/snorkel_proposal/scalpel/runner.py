"""Run one method on one split and score it.

    base reference (cached per split + model): base answers, base generations, base
    next-token distributions on retain probes, base utility, base lens curve
      -> method.unlearn(requests, allowed_retain)       (the only data the method sees)
      -> causality check (refuse non-causal wrappers: a grader exploit)
      -> core track: every probe, teacher-forced + open-ended generation
      -> utility, KL on retain probes, routing diagnostics (detect-then-edit methods)
      -> attacks: logit lens, int8/int4 quantization, relearning on half the requests
      -> strict tier (weights only) for methods with runtime components
      -> results.json + probes.jsonl
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from . import attacks as A
from . import metrics as M
from .grading import GRADER_VERSION, classify
from .methods import make
from .methods.base import UnlearnContext
from .model_api import (ModelHandle, Renderer, causality_violation, decoder_key, first_token_logprobs, generate,
                        teacher_forced)
from .schema import Probe, Split, write_jsonl
from .utility import utility_eval

RESULTS_VERSION = "scalpel-results-v1"


@dataclass
class EvalConfig:
    batch_size: int = 16
    gen: bool = True
    gen_max_new_tokens: int = 24
    attacks: Tuple[str, ...] = ("lens", "quant", "relearn")
    quant_bits: Tuple[int, ...] = (8, 4)
    relearn_steps: int = 20
    relearn_lr: float = 1e-5
    relearn_scope: str = "mlp"
    strict_tier: bool = True
    utility: bool = True
    kl_topk: int = 20
    causality_tol: float = 0.05
    bootstrap: int = 500
    max_unlearn_seconds: float = 4 * 3600

    def ref_key(self) -> str:
        return hashlib.sha256(json.dumps({"gen": self.gen, "tok": self.gen_max_new_tokens, "kl": self.kl_topk,
                                          "u": self.utility, "lens": "lens" in self.attacks},
                                         sort_keys=True).encode()).hexdigest()[:12]


# ------------------------------------------------------------------------ loading
def pick_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_hf(path: str, device: str = "auto", dtype: str = "auto"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = pick_device(device)
    if dtype == "auto":
        dtype = "bf16" if device == "cuda" else "fp32"
    dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dt).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def renderer_for(split: Split, tok) -> Renderer:
    chat = (split.manifest.get("renderer") or {}).get("chat_template")
    return Renderer(tok, "always" if chat else "none")


def _atomic_write(path: Path, text: str) -> None:
    """Concurrent tasks share a split's cache: readers must never see a half-written file."""
    import os
    import tempfile
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    os.replace(tmp, path)


# ------------------------------------------------------------------- fingerprints
def spread_index(n: int, k: int = 64, device=None) -> torch.Tensor:
    """min(k, n) evenly spread positions in [0, n - 1], first and last included, in integer
    arithmetic: a float32 linspace rounds n - 1 up to n once n > 2**24 (a 3B model's embedding
    has 3.9e8 entries), an out-of-range index that is a device-side assert on CUDA."""
    k = min(int(k), int(n))
    if k <= 1:
        return torch.zeros(k, dtype=torch.long, device=device)
    return torch.arange(k, dtype=torch.long, device=device) * (int(n) - 1) // (k - 1)


@torch.no_grad()
def weight_fingerprint(hf) -> Dict[str, Tuple[float, float, int]]:
    out = {}
    for n, p in hf.named_parameters():
        x = p.detach().float().flatten()
        idx = spread_index(x.numel(), device=x.device)
        out[n] = (float(x.sum()), float(x.abs().sum()), int(hashlib.sha1(x[idx].cpu().numpy().tobytes()).hexdigest()[:12], 16))
    return out


def changed_params(before, after, hf) -> Dict[str, Any]:
    sizes = {n: p.numel() for n, p in hf.named_parameters()}
    changed = [n for n in before if n in after and before[n] != after[n]]
    return {"tensors_changed": len(changed), "params_in_changed_tensors": int(sum(sizes[n] for n in changed)),
            "total_params": int(sum(sizes.values())), "changed_tensor_names": changed[:40]}


def extra_params(handle: ModelHandle) -> int:
    base_ids = {id(p) for p in handle.hf.parameters()}
    mods = [handle.model] + ([handle.lens_module] if handle.lens_module is not None else [])
    seen, n = set(), 0
    for m in mods:
        if not hasattr(m, "parameters"):
            continue
        for p in m.parameters():
            if id(p) not in base_ids and id(p) not in seen:
                seen.add(id(p))
                n += p.numel()
    for m in mods:   # buffers (e.g. residual rows registered as buffers)
        if hasattr(m, "buffers"):
            for b in m.buffers():
                if id(b) not in seen and id(b) not in {id(x) for x in handle.hf.buffers()}:
                    seen.add(id(b))
                    n += b.numel()
    return int(n)


# ------------------------------------------------------------------ base reference
def base_reference(split: Split, handle: ModelHandle, renderer: Renderer, cfg: EvalConfig,
                   cache_dir: Optional[Path] = None, log=print, decoders: Sequence[str] = ("hf",)) -> Dict:
    """Base-model outputs every method is compared against. Generations are stored per
    decoder ("gen" = HF generate, "gen_full" = prefix recompute) so a method is always
    compared with base generations from the decoder it uses itself."""
    key = f"{split.fingerprint()[:16]}_{cfg.ref_key()}"
    path = Path(cache_dir) / f"base_reference_{key}.json" if cache_dir else None
    if path and path.exists():
        ref = json.loads(path.read_text())
        if cfg.gen and "full" in decoders and "gen_full" not in ref:
            ref["gen_full"] = _base_gens(split, handle, renderer, cfg, "full")
            _atomic_write(path, json.dumps(ref) + "\n")
        return ref
    log("[base] scoring the base model (cached for later runs)")
    t0 = time.time()
    tf = teacher_forced(handle, renderer, split.probes, batch_size=cfg.batch_size,
                        topk=cfg.kl_topk if cfg.kl_topk else 0)
    tf_seconds = time.time() - t0
    ref = {"split_fingerprint": split.fingerprint(), "tf_seconds": tf_seconds,
           "tf": {p.probe_id: {"correct": r["correct"], "prob": r["prob"],
                               **({"first_topk": r["first_topk"]} if p.role == "retain" and "first_topk" in r else {})}
                  for p, r in zip(split.probes, tf)}}
    if cfg.gen:
        ref["gen"] = _base_gens(split, handle, renderer, cfg, "hf")
        if "full" in decoders:
            ref["gen_full"] = _base_gens(split, handle, renderer, cfg, "full")
    if cfg.utility and split.utility:
        ref["utility"] = utility_eval(handle, renderer, split.utility, batch_size=max(1, cfg.batch_size // 2))
    if "lens" in cfg.attacks:
        lp = A._attack_probes(split.probes)
        cand = A.lens_candidates(renderer, lp, seed=split.seed)
        ref["lens"] = A.lens_attack(handle, {}, renderer, lp, cand)
    if path:
        _atomic_write(path, json.dumps(ref) + "\n")
    return ref


def _attack_key(split: Split, cfg: Dict) -> str:
    return hashlib.sha256(json.dumps([split.fingerprint(), sorted(cfg["attacks"]), list(cfg["quant_bits"]),
                                      cfg["relearn_steps"], cfg["relearn_lr"], cfg["relearn_scope"]]).encode()).hexdigest()[:16]


def base_attack_reference(split: Split, handle: ModelHandle, renderer: Renderer, cfg: EvalConfig,
                          cache_dir: Optional[Path] = None, log=print) -> Dict:
    """The BASE model's residual after each attack (quantization and fine-tuning also damage a
    model that forgot nothing). Run before unlearning; cached per split + attack config."""
    if not ({"quant", "relearn"} & set(cfg.attacks)):
        return {}
    path = Path(cache_dir) / f"base_attacks_{_attack_key(split, asdict(cfg))}.json" if cache_dir else None
    if path and path.exists():
        return json.loads(path.read_text())
    log("[base] attacks on the base model (normalisation; cached)")
    out: Dict[str, Any] = {}
    if "quant" in cfg.attacks:
        out["quant"] = A.quant_attack(handle, renderer, split.probes, cfg.quant_bits, cfg.batch_size)
    if "relearn" in cfg.attacks:
        out["relearn"] = A.relearn_attack(handle, renderer, split.requests, split.probes, split.seed,
                                          steps=cfg.relearn_steps, lr=cfg.relearn_lr, scope=cfg.relearn_scope)
    if path:
        _atomic_write(path, json.dumps(out) + "\n")
    return out


def _base_gens(split, handle, renderer, cfg, decoder: str) -> Dict:
    h = ModelHandle(model=handle.model, tokenizer=handle.tokenizer, hf=handle.hf, decode=decoder)
    gens = generate(h, renderer, split.probes, max_new_tokens=cfg.gen_max_new_tokens, batch_size=cfg.batch_size)
    return {p.probe_id: {"text": g[:300], "class": classify(g, p.answers(), p.role, p.format)}
            for p, g in zip(split.probes, gens)}


# ------------------------------------------------------------------------ scoring
def score_core(split: Split, handle: ModelHandle, renderer: Renderer, cfg: EvalConfig, ref: Dict,
               probes: Optional[Sequence[Probe]] = None, gen: Optional[bool] = None) -> Dict:
    probes = list(split.probes if probes is None else probes)
    gen = cfg.gen if gen is None else gen
    t0 = time.time()
    tf = teacher_forced(handle, renderer, probes, batch_size=cfg.batch_size)
    tf_seconds = time.time() - t0
    gens = generate(handle, renderer, probes, max_new_tokens=cfg.gen_max_new_tokens,
                    batch_size=cfg.batch_size) if gen else [None] * len(probes)
    rows = []
    base_gens = ref.get("gen_full" if decoder_key(handle) == "full" else "gen") or {}
    for p, r, g in zip(probes, tf, gens):
        base = ref["tf"].get(p.probe_id, {})
        if not base.get("correct", False):
            continue        # the base model must answer it at evaluation time too
        bg = base_gens.get(p.probe_id)
        ok_class = "leak" if p.role == "forget" else "correct"
        row = {"probe_id": p.probe_id, "role": p.role, "family": p.family, "fact_id": p.fact_id,
               "request_id": p.request_id, "format": p.format,
               "cluster": p.request_id or p.meta.get("near_request") or p.fact_id,
               "tf_correct": bool(r["correct"]), "prob": r["prob"], "base_prob": base.get("prob")}
        if g is not None:
            row.update({"gen": g[:300], "gen_class": classify(g, p.answers(), p.role, p.format),
                        "base_gen_ok": bool(bg and bg["class"] == ok_class)})
        rows.append(row)
    fam_tf = M.family_scores(rows, "tf")
    fam_gen = M.family_scores(rows, "gen") if gen else {}
    behaviour = _behaviour(rows) if gen else {}
    return {"rows": rows, "families_tf": fam_tf, "families_gen": fam_gen, "behaviour": behaviour,
            "tf_seconds": tf_seconds}


def _behaviour(rows: Sequence[Dict]) -> Dict:
    """How the model fails to answer: refusal vs hallucination (forget), false refusal (retain)."""
    out = {}
    for role in ("forget", "retain"):
        rs = [r for r in rows if r["role"] == role and r.get("base_gen_ok")]
        if not rs:
            continue
        n = len(rs)
        cls = [r["gen_class"] for r in rs]
        if role == "forget":
            out["forget"] = {"leak": 100 * cls.count("leak") / n, "refusal": 100 * cls.count("refusal") / n,
                             "hallucination_or_other": 100 * cls.count("other") / n, "n": n}
        else:
            out["retain"] = {"correct": 100 * cls.count("correct") / n,
                             "false_refusal": 100 * cls.count("refusal") / n, "wrong": 100 * cls.count("wrong") / n, "n": n}
    return out


def kl_retain(split: Split, handle: ModelHandle, renderer: Renderer, ref: Dict, cfg: EvalConfig) -> Optional[float]:
    probes = [p for p in split.retain_probes() if "first_topk" in ref["tf"].get(p.probe_id, {})]
    if not probes:
        return None
    tops = [ref["tf"][p.probe_id]["first_topk"] for p in probes]
    lps = first_token_logprobs(handle, renderer, probes, [[t for t, _ in tk] for tk in tops], cfg.batch_size)
    vals = [M.kl_topk(tk, lp) for tk, lp in zip(tops, lps) if lp]
    return sum(vals) / len(vals) if vals else None


def routing_diagnostics(split: Split, handle: ModelHandle, renderer: Renderer, rows: Sequence[Dict]) -> Optional[Dict]:
    """For detect-then-edit methods: split the forget residual into routing misses (the
    detector did not fire / fired the wrong request) and actuation failures (it fired the
    right request and the answer survived anyway); count false fires on retain probes."""
    if handle.route is None:
        return None
    by_id = {p.probe_id: p for p in split.probes}
    keep = [r for r in rows]
    texts = [renderer.prompt_text(by_id[r["probe_id"]])[0] for r in keep]
    fired = handle.route(texts)
    fam: Dict[str, Dict[str, int]] = {}
    for r, f in zip(keep, fired):
        k = f"{r['role']}/{r['family']}"
        d = fam.setdefault(k, {"n": 0, "routed_right": 0, "routed_wrong": 0, "missed": 0, "false_fire": 0,
                               "residual_when_routed_right": 0, "residual_when_missed": 0})
        d["n"] += 1
        if r["role"] == "forget":
            if f == r["request_id"]:
                d["routed_right"] += 1
                d["residual_when_routed_right"] += int(r["tf_correct"])
            elif f is None:
                d["missed"] += 1
                d["residual_when_missed"] += int(r["tf_correct"])
            else:
                d["routed_wrong"] += 1
        elif f is not None:
            d["false_fire"] += 1
    f_rows = [d for k, d in fam.items() if k.startswith("forget/")]
    r_rows = [d for k, d in fam.items() if k.startswith("retain/")]
    nf = sum(d["n"] for d in f_rows) or 1
    nr = sum(d["n"] for d in r_rows) or 1
    return {"by_family": fam,
            "forget_route_recall": 100 * sum(d["routed_right"] for d in f_rows) / nf,
            "forget_residual_from_routing": 100 * sum(d["residual_when_missed"] for d in f_rows) / nf,
            "forget_residual_from_actuation": 100 * sum(d["residual_when_routed_right"] for d in f_rows) / nf,
            "retain_false_fire_rate": 100 * sum(d["false_fire"] for d in r_rows) / nr}


# ---------------------------------------------------------------------------- run
def run(split: Split, method_name: str, method_args: Dict[str, Any], model_loader: Callable[[], Tuple[Any, Any]],
        cfg: EvalConfig, out_dir: Path, cache_dir: Optional[Path] = None, log=print,
        base_handle: Optional[ModelHandle] = None) -> Dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if base_handle is None:
        hf, tok = model_loader()
        base_handle = ModelHandle(model=hf, tokenizer=tok, hf=hf)
    tok = base_handle.tokenizer
    renderer = renderer_for(split, tok)
    method = make(method_name, **method_args)
    ref = base_reference(split, base_handle, renderer, cfg, cache_dir=cache_dir, log=log,
                         decoders=("hf", "full") if method.card.decode == "full" else ("hf",))
    att_base = base_attack_reference(split, base_handle, renderer, cfg, cache_dir=cache_dir, log=log)
    before = weight_fingerprint(base_handle.hf)

    ctx = UnlearnContext(seed=split.seed, renderer=renderer, generic_prompts=split.manifest.get("generic_prompts", ()),
                         max_seconds=cfg.max_unlearn_seconds, log=log)
    log(f"[run] {method_name}: unlearning {len(split.requests)} requests")
    t0 = time.time()
    handle = method.unlearn(base_handle, split.requests, split.allowed_retain, ctx)
    unlearn_seconds = time.time() - t0
    card = method.card.to_dict()

    fprobes = [p for p in split.forget_probes() if ref["tf"].get(p.probe_id, {}).get("correct")]
    caus = causality_violation(handle, renderer, fprobes, n=16)
    valid = caus <= cfg.causality_tol
    if not valid:
        log(f"[run] INVALID: non-causal model (max logit change {caus:.3f} > {cfg.causality_tol})")

    log("[run] core track")
    core = score_core(split, handle, renderer, cfg, ref)
    res: Dict[str, Any] = {
        "version": RESULTS_VERSION, "split": split.name, "split_fingerprint": split.fingerprint(),
        "partition": split.partition, "seed": split.seed, "model_id": split.model_id, "track": split.manifest.get("track"),
        "n_requests": len(split.requests),
        "method": card, "method_args": method_args, "method_report": method.report(), "valid": valid,
        "grader_version": GRADER_VERSION,
        "causality_max_logit_change": caus, "eval_config": asdict(cfg),
        "families_tf": core["families_tf"], "families_gen": core["families_gen"], "behaviour": core["behaviour"],
        "cost": {"unlearn_seconds": unlearn_seconds,
                 "inference_overhead": core["tf_seconds"] / max(1e-9, ref.get("tf_seconds", core["tf_seconds"])),
                 **changed_params(before, weight_fingerprint(handle.hf), handle.hf),
                 "extra_params": extra_params(handle)},
        "env": {"python": platform.python_version(), "torch": torch.__version__,
                "transformers": __import__("transformers").__version__, "device": str(handle.device)},
    }
    if cfg.utility and split.utility:
        after_u = utility_eval(handle, renderer, split.utility, batch_size=max(1, cfg.batch_size // 2))
        res["utility"] = M.utility_score(ref.get("utility", {}), after_u)
    if cfg.kl_topk:
        res["kl_retain_first_token"] = kl_retain(split, handle, renderer, ref, cfg)
    rd = routing_diagnostics(split, handle, renderer, core["rows"])
    if rd:
        res["routing"] = rd

    # attacks (shipped-weights tier)
    att: Dict[str, Any] = {}
    if "lens" in cfg.attacks and ref.get("lens"):
        lp = A._attack_probes(split.probes)
        cand = A.lens_candidates(renderer, lp, seed=split.seed)
        att["lens"] = A.lens_attack(handle, ref["lens"], renderer, lp, cand)
    if "quant" in cfg.attacks:
        att["quant"] = A.quant_attack(handle, renderer, split.probes, cfg.quant_bits, cfg.batch_size)
    if "relearn" in cfg.attacks:
        att["relearn"] = A.relearn_attack(handle, renderer, split.requests, split.probes, split.seed,
                                          steps=cfg.relearn_steps, lr=cfg.relearn_lr, scope=cfg.relearn_scope)
    for bits, v in (att.get("quant") or {}).items():
        v["base_residual"] = ((att_base.get("quant") or {}).get(bits) or {}).get("residual")
    if att.get("relearn"):
        att["relearn"]["base_heldout_residual_after"] = (att_base.get("relearn") or {}).get("heldout_residual_after")
    res["attacks"] = att
    rob = robustness(att)

    # strict tier: weights only
    if cfg.strict_tier and handle.runtime_components and handle.strip is not None:
        stripped = handle.strip()
        sc = score_core(split, stripped, renderer, cfg, ref, gen=False)
        res["strict_tier"] = {"families_tf": sc["families_tf"],
                              "aggregates": M.aggregates(sc["families_tf"], res.get("utility"), None)}

    res["robustness"] = rob
    res["aggregates_tf"] = M.aggregates(core["families_tf"], res.get("utility"), rob.get("R"))
    res["aggregates_tf"]["E_complete"] = M.completeness(core["rows"], "tf")
    if core["families_gen"]:
        res["aggregates_gen"] = M.aggregates(core["families_gen"], res.get("utility"), rob.get("R"))
        res["aggregates_gen"]["E_complete"] = M.completeness(core["rows"], "gen")
    if cfg.bootstrap:
        res["ci_tf"] = M.bootstrap(core["rows"], "tf", n_boot=cfg.bootstrap, seed=split.seed,
                                   utility=res.get("utility"), robustness=rob.get("R"))
    (out_dir / "results.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    write_jsonl(out_dir / "probes.jsonl", core["rows"])
    log(f"[run] {method_name}: SCALPEL(tf) = {_fmt(res['aggregates_tf'].get('SCALPEL'))} "
        f"E = {_fmt(res['aggregates_tf'].get('E'))} L = {_fmt(res['aggregates_tf'].get('L'))} "
        f"U = {_fmt(res['aggregates_tf'].get('U'))} R = {_fmt(res['aggregates_tf'].get('R'))}")
    res["_handle"] = handle
    return res


def regrade(split: Split, run_dir: Path, cache_dir: Optional[Path] = None) -> Dict:
    """Re-grade a finished run's stored generations with the current rule grader (and backfill
    metadata) without touching the model: grader updates never require re-running methods."""
    run_dir = Path(run_dir)
    res = json.loads((run_dir / "results.json").read_text())
    if res.get("split_fingerprint") != split.fingerprint():
        raise ValueError(f"{run_dir} was not produced on split {split.name}")
    by_id = {p.probe_id: p for p in split.probes}
    rows = [json.loads(l) for l in (run_dir / "probes.jsonl").read_text().splitlines() if l.strip()]
    for r in rows:
        if "gen" in r and r["probe_id"] in by_id:
            p = by_id[r["probe_id"]]
            r["gen_class"] = classify(r["gen"], p.answers(), p.role, p.format)
    att = res.get("attacks") or {}
    cfgd = res.get("eval_config") or {}
    base_path = (Path(cache_dir) / f"base_attacks_{_attack_key(split, cfgd)}.json") if (cache_dir and cfgd) else None
    if base_path and base_path.exists():          # backfill base-relative attack residuals
        ab = json.loads(base_path.read_text())
        for bits, v in (att.get("quant") or {}).items():
            v.setdefault("base_residual", ((ab.get("quant") or {}).get(bits) or {}).get("residual"))
        if att.get("relearn"):
            att["relearn"].setdefault("base_heldout_residual_after", (ab.get("relearn") or {}).get("heldout_residual_after"))
        res["robustness"] = robustness(att)
        res["aggregates_tf"] = M.aggregates(res["families_tf"], res.get("utility"), res["robustness"].get("R"))
        res["aggregates_tf"]["E_complete"] = M.completeness(rows, "tf")
    res["families_gen"] = M.family_scores(rows, "gen") if any("gen" in r for r in rows) else {}
    res["behaviour"] = _behaviour(rows) if res["families_gen"] else {}
    if res["families_gen"]:
        res["aggregates_gen"] = M.aggregates(res["families_gen"], res.get("utility"), (res.get("robustness") or {}).get("R"))
    res["grader_version"] = GRADER_VERSION
    res.setdefault("n_requests", len(split.requests))
    (run_dir / "results.json").write_text(json.dumps(res, indent=1, default=str) + "\n")
    write_jsonl(run_dir / "probes.jsonl", rows)
    return res


def robustness(att: Dict) -> Dict:
    """R = 100 - the worst residual over the attacks that ran. Each residual is relative to
    the BASE model under the same attack (100 = as much as the unedited model keeps after
    it), because int4 rounding or a few fine-tuning steps also erase facts nobody unlearned."""
    def rel(x, base):
        if x is None:
            return None
        return min(100.0, 100.0 * x / base) if base else x
    parts, raw = {}, {}
    for bits, v in (att.get("quant") or {}).items():
        if v.get("residual") is not None:
            raw[f"quant_{bits}"] = v["residual"]
            parts[f"quant_{bits}"] = rel(v["residual"], v.get("base_residual"))
    rl = att.get("relearn") or {}
    if rl.get("heldout_residual_after") is not None:
        raw["relearn"] = rl["heldout_residual_after"]
        parts["relearn"] = rel(rl["heldout_residual_after"], rl.get("base_heldout_residual_after"))
    lens = att.get("lens") or {}
    if lens.get("lens_residual") is not None:
        raw["lens"] = parts["lens"] = lens["lens_residual"]
    R = 100.0 - max(parts.values()) if parts else None
    return {"R": R, "residual_by_attack": parts, "raw_residual_by_attack": raw}


def _fmt(x):
    return "n/a" if x is None else f"{x:.1f}"
