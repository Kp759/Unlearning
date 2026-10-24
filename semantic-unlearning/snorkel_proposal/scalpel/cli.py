"""Command line.

    python -m scalpel toy        --out results/toy                       # everything, tiny model, CPU
    python -m scalpel build      --model M --out splits/dev_s1.json [--sources mcf,zsre,mquake] [--track entity]
    python -m scalpel run        --split S --model M --method ga [--method-arg lr=1e-5] --out runs/ga_s1
    python -m scalpel suite      --split S --model M --methods noop,idk_all,ga,npo --out runs/
    python -m scalpel sequential --split S --model M --method ga --steps 5 --out runs/seq_ga
    python -m scalpel redteam    --split S --model M --out runs/redteam
    python -m scalpel validate   --split S
    python -m scalpel report     --runs 'runs/*/results.json' --out leaderboard/
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

from .schema import Split

PKG_ROOT = Path(__file__).resolve().parents[1]          # snorkel_proposal/
REPO_DATA = PKG_ROOT.parent / "data"                    # semantic-unlearning/data
MQUAKE_CACHE = PKG_ROOT / "data_cache" / "mquake"


def _log(msg: str):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def _method_args(pairs: List[str]) -> Dict:
    out = {}
    for p in pairs or []:
        k, _, v = p.partition("=")
        out[k] = v
    return out


def _eval_cfg(a):
    from .runner import EvalConfig
    attacks = tuple(x for x in (a.attacks or "").split(",") if x)
    return EvalConfig(batch_size=a.batch_size, gen=not a.no_gen, gen_max_new_tokens=a.gen_tokens, attacks=attacks,
                      relearn_steps=a.relearn_steps, relearn_lr=a.relearn_lr, relearn_scope=a.relearn_scope,
                      bootstrap=a.bootstrap,
                      utility=not a.no_utility, max_unlearn_seconds=a.max_unlearn_seconds)


def _add_eval_args(p):
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--no-gen", action="store_true", help="skip the open-ended (generation) scorer")
    p.add_argument("--gen-tokens", type=int, default=24)
    p.add_argument("--attacks", default="lens,quant,relearn")
    p.add_argument("--relearn-steps", type=int, default=20)
    p.add_argument("--relearn-lr", type=float, default=1e-5)
    p.add_argument("--relearn-scope", choices=("mlp", "mlp_down", "all"), default="mlp",
                   help="parameters the relearning attacker fine-tunes (mlp_down fits a 16 GB laptop)")
    p.add_argument("--bootstrap", type=int, default=500)
    p.add_argument("--no-utility", action="store_true")
    p.add_argument("--max-unlearn-seconds", type=float, default=4 * 3600)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto")


def _loader(a):
    from .runner import load_hf
    return lambda: load_hf(a.model, a.device, a.dtype)


# --------------------------------------------------------------------------- build
def cmd_build(a):
    from .build import BuildConfig, build_entity_split, build_fact_split, utility_sets
    from .model_api import ModelHandle, Renderer
    from .runner import load_hf
    from . import sources as S

    hf, tok = load_hf(a.model, a.device, a.dtype)
    handle = ModelHandle(model=hf, tokenizer=tok, hf=hf)
    renderer = Renderer(tok, a.chat_template)
    cfg = BuildConfig(name=a.name, seed=a.seed, partition=a.partition, salt=a.salt, n_forget=a.n_forget,
                      n_disjoint=a.n_disjoint, batch_size=a.batch_size, use_chat_template=a.chat_template,
                      screen_max=a.screen_max,
                      private_templates=a.private_templates)
    rwku_root = Path(a.rwku or REPO_DATA / "rwku")
    rwku = S.load_rwku(rwku_root) if rwku_root.exists() else []
    ppl = _ppl_texts(Path(a.ppl_texts) if a.ppl_texts else REPO_DATA / "wikidata")
    util = utility_sets(rwku, ppl, cfg.n_mmlu, cfg.n_trivia, a.seed)
    if a.track == "entity":
        targets = rwku[:a.n_entities]
        split = build_entity_split(targets, handle, renderer, cfg, model_id=a.model_id or a.model, utility=util, log=_log)
    else:
        groups, srcman = [], {}
        for name in a.sources.split(","):
            if name == "mcf":
                p = Path(a.mcf or REPO_DATA / "multi_counterfact.json")
                groups.append(S.load_mcf(p)); srcman["mcf"] = {"path": str(p), "sha256": S.file_sha256(p)}
            elif name == "zsre":
                p = Path(a.zsre or REPO_DATA / "zsre_mend_eval.json")
                groups.append(S.load_zsre(p)); srcman["zsre"] = {"path": str(p), "sha256": S.file_sha256(p)}
            elif name == "mquake":
                paths = [Path(x) for x in (a.mquake.split(",") if a.mquake else
                                           [MQUAKE_CACHE / "MQuAKE-CF-3k-v2.json", MQUAKE_CACHE / "MQuAKE-CF.json"])]
                paths = [p for p in paths if p.exists()]
                groups.append(S.load_mquake(paths))
                srcman["mquake"] = [{"path": str(p), "sha256": S.file_sha256(p)} for p in paths]
            elif name == "toy":
                from .toy import load_toy_world
                facts, util = load_toy_world(Path(a.model))
                groups.append(facts); srcman["toy"] = {"model_dir": a.model}
            else:
                raise SystemExit(f"unknown source {name}")
        facts, stats = S.merge_facts(groups)
        srcman["merge"] = stats
        _log(f"[build] {stats['facts']} facts after merging ({json.dumps(stats)})")
        split = build_fact_split(facts, handle, renderer, cfg, model_id=a.model_id or a.model, utility=util,
                                 source_manifest=srcman, log=_log)
    out = split.save(Path(a.out))
    from .report import split_summary_md
    (out.parent / "split_summary.md").write_text(split_summary_md(split))
    _log(f"[build] wrote {out}: {len(split.requests)} requests, {len(split.probes)} probes")
    print(json.dumps(split.manifest["counts"], indent=1))


def _ppl_texts(path: Path) -> List[str]:
    if path.is_dir():
        arrow = next(path.rglob("*.arrow"), None)
        if arrow is None:
            return []
        import pyarrow as pa
        with pa.memory_map(str(arrow)) as f:
            t = pa.ipc.open_stream(f).read_all()
        return [x for x in t.column("text").to_pylist() if x][:200]
    return [l.strip() for l in path.read_text().splitlines() if l.strip()][:200]


# ----------------------------------------------------------------------------- run
def cmd_run(a):
    from .runner import run
    split = Split.load(Path(a.split))
    res = run(split, a.method, _method_args(a.method_arg), _loader(a), _eval_cfg(a), Path(a.out),
              cache_dir=Path(a.split).parent / "cache", log=_log)
    res.pop("_handle", None)


def cmd_suite(a):
    """Several methods on one split, reloading the base model only after a method edits it."""
    from .model_api import ModelHandle
    from .runner import load_hf, run, weight_fingerprint
    split = Split.load(Path(a.split))
    cfg = _eval_cfg(a)
    hf, tok = load_hf(a.model, a.device, a.dtype)
    ref_fp = weight_fingerprint(hf)
    for name in a.methods.split(","):
        out = Path(a.out) / name
        if (out / "results.json").exists() and not a.overwrite:
            _log(f"[suite] {name}: done, skipping")
            continue
        args = _method_args([x for x in (a.method_arg or []) if x.startswith(name + ".")])
        args = {k.split(".", 1)[1]: v for k, v in args.items()}
        base = ModelHandle(model=hf, tokenizer=tok, hf=hf)
        res = run(split, name, args, None, cfg, out, cache_dir=Path(a.split).parent / "cache", log=_log, base_handle=base)
        res.pop("_handle", None)
        if weight_fingerprint(hf) != ref_fp:
            _log("[suite] weights changed; reloading the base model")
            del hf
            hf, tok = load_hf(a.model, a.device, a.dtype)


def cmd_sequential(a):
    from .sequential import run_sequential
    split = Split.load(Path(a.split))
    run_sequential(split, a.method, _method_args(a.method_arg), _loader(a), _eval_cfg(a), Path(a.out),
                   steps=a.steps, cache_dir=Path(a.split).parent / "cache", log=_log)


def cmd_validate(a):
    from .redteam import validate_split
    v = validate_split(Split.load(Path(a.split)))
    print(json.dumps(v, indent=1))
    sys.exit(0 if v["ok"] else 1)


def cmd_redteam(a):
    from .methods import REFERENCE_CHEATS
    from .redteam import cheat_gate, markdown, validate_split
    a.methods = ",".join(REFERENCE_CHEATS)
    cmd_suite(a)
    split = Split.load(Path(a.split))
    results = [json.loads((Path(a.out) / m / "results.json").read_text()) for m in REFERENCE_CHEATS]
    v, g = validate_split(split), cheat_gate(results)
    md = markdown(v, g, split.name)
    (Path(a.out) / "RED_TEAM_REPORT.md").write_text(md)
    (Path(a.out) / "red_team.json").write_text(json.dumps({"validation": v, "cheat_gate": g}, indent=1))
    print(md)


def cmd_regrade(a):
    from .runner import regrade
    split = Split.load(Path(a.split))
    for pat in a.runs:
        for d in sorted(glob.glob(pat)):
            if (Path(d) / "results.json").exists():
                regrade(split, Path(d), cache_dir=Path(a.split).parent / "cache")
                _log(f"[regrade] {d}")


def cmd_gate(a):
    """Split validation + reference-cheat gate from results already on disk (cluster tasks run
    each cheat as its own job; this collects them)."""
    from .methods import REFERENCE_CHEATS
    from .redteam import cheat_gate, markdown, validate_split
    split = Split.load(Path(a.split))
    results = []
    for m in REFERENCE_CHEATS:
        f = Path(a.runs) / m / "results.json"
        if f.exists():
            results.append(json.loads(f.read_text()))
        else:
            _log(f"[gate] missing {f}")
    v, g = validate_split(split), cheat_gate(results)
    if len(results) < len(REFERENCE_CHEATS):
        g["ok"] = False
    md = markdown(v, g, split.name)
    (Path(a.runs) / "RED_TEAM_REPORT.md").write_text(md)
    (Path(a.runs) / "red_team.json").write_text(json.dumps({"validation": v, "cheat_gate": g}, indent=1))
    print(md)


def cmd_report(a):
    from .report import write_report
    paths = sorted({p for pat in a.runs for p in glob.glob(pat)})
    table = write_report([Path(p) for p in paths], Path(a.out), a.title)
    _log(f"[report] {len(paths)} runs -> {a.out}/leaderboard.md ({len(table)} rows)")


# ----------------------------------------------------------------------------- toy
def cmd_toy(a):
    """Train the toy model, build a split, run cheats + baselines, report, red-team."""
    from .runner import pick_device
    from .toy import train_toy_model
    out = Path(a.out)
    mdir = train_toy_model(out / "model", n_people=a.people, epochs=a.epochs, device=pick_device("auto"),
                           batch_size=256, lr=3e-3, log=_log)
    split_path = out / "split_dev_s1.json"
    common = ["--model", str(mdir), "--device", "cpu", "--dtype", "fp32", "--batch-size", "32"]
    if not split_path.exists():
        main(["build", *common, "--sources", "toy", "--out", str(split_path), "--n-forget", str(a.n_forget),
              "--n-disjoint", "60", "--chat-template", "none", "--name", "scalpel-toy", "--model-id", "toy-llama"])
    main(["validate", "--split", str(split_path)], exit_ok=True)
    toy_args = ["--method-arg", "ga.lr=2e-3", "--method-arg", "gd.lr=2e-3", "--method-arg", "npo.lr=2e-3",
                "--method-arg", "rmu.lr=3e-3", "--method-arg", "rmu.layer=1", "--method-arg", "rmu.steps=300", "--method-arg", "rmu.alpha=100"]
    eval_args = ["--relearn-lr", "5e-4", "--relearn-steps", "20", "--bootstrap", "200"]
    main(["redteam", "--split", str(split_path), *common, *eval_args, "--out", str(out / "runs")])
    main(["suite", "--split", str(split_path), *common, *eval_args, "--methods", "ga,gd,npo,rmu",
          "--out", str(out / "runs"), *toy_args])
    main(["sequential", "--split", str(split_path), *common, "--method", "gd", "--method-arg", "lr=2e-3",
          "--steps", "4", "--out", str(out / "sequential_gd"), "--no-gen", "--attacks", ""])
    main(["report", "--runs", str(out / "runs" / "*" / "results.json"), "--out", str(out / "leaderboard"),
          "--title", "SCALPEL leaderboard — toy world (pipeline check, not a result)"])


# ---------------------------------------------------------------------------- main
def main(argv=None, exit_ok: bool = False):
    ap = argparse.ArgumentParser(prog="scalpel", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--model", required=True)
    b.add_argument("--model-id", default=None)
    b.add_argument("--out", required=True)
    b.add_argument("--track", choices=("fact", "entity"), default="fact")
    b.add_argument("--sources", default="mcf,zsre,mquake")
    b.add_argument("--mcf"); b.add_argument("--zsre"); b.add_argument("--mquake"); b.add_argument("--rwku")
    b.add_argument("--ppl-texts")
    b.add_argument("--name", default="scalpel-fact")
    b.add_argument("--seed", type=int, default=1)
    b.add_argument("--partition", choices=("dev", "test"), default="dev")
    b.add_argument("--salt", default="scalpel-v1")
    b.add_argument("--private-templates", default=None)
    b.add_argument("--n-forget", type=int, default=50)
    b.add_argument("--n-disjoint", type=int, default=150)
    b.add_argument("--screen-max", type=int, default=8000, help="max candidate facts scored for the knowledge filter")
    b.add_argument("--n-entities", type=int, default=10)
    b.add_argument("--batch-size", type=int, default=16)
    b.add_argument("--chat-template", choices=("auto", "always", "none"), default="auto")
    b.add_argument("--device", default="auto"); b.add_argument("--dtype", default="auto")
    b.set_defaults(fn=cmd_build)

    r = sub.add_parser("run")
    r.add_argument("--split", required=True); r.add_argument("--model", required=True)
    r.add_argument("--method", required=True); r.add_argument("--method-arg", action="append")
    r.add_argument("--out", required=True)
    _add_eval_args(r)
    r.set_defaults(fn=cmd_run)

    s = sub.add_parser("suite")
    s.add_argument("--split", required=True); s.add_argument("--model", required=True)
    s.add_argument("--methods", required=True)
    s.add_argument("--method-arg", action="append", help="METHOD.key=value")
    s.add_argument("--out", required=True); s.add_argument("--overwrite", action="store_true")
    _add_eval_args(s)
    s.set_defaults(fn=cmd_suite)

    q = sub.add_parser("sequential")
    q.add_argument("--split", required=True); q.add_argument("--model", required=True)
    q.add_argument("--method", required=True); q.add_argument("--method-arg", action="append")
    q.add_argument("--steps", type=int, default=5); q.add_argument("--out", required=True)
    _add_eval_args(q)
    q.set_defaults(fn=cmd_sequential)

    t = sub.add_parser("redteam")
    t.add_argument("--split", required=True); t.add_argument("--model", required=True)
    t.add_argument("--out", required=True); t.add_argument("--overwrite", action="store_true")
    t.add_argument("--method-arg", action="append")
    _add_eval_args(t)
    t.set_defaults(fn=cmd_redteam)

    v = sub.add_parser("validate")
    v.add_argument("--split", required=True)
    v.set_defaults(fn=cmd_validate)

    gt = sub.add_parser("gate", help="split validation + cheat gate from finished runs")
    gt.add_argument("--split", required=True); gt.add_argument("--runs", required=True)
    gt.set_defaults(fn=cmd_gate)

    rg = sub.add_parser("regrade", help="re-grade stored generations with the current grader")
    rg.add_argument("--split", required=True); rg.add_argument("--runs", nargs="+", required=True)
    rg.set_defaults(fn=cmd_regrade)

    rp = sub.add_parser("report")
    rp.add_argument("--runs", nargs="+", required=True); rp.add_argument("--out", required=True)
    rp.add_argument("--title", default="SCALPEL leaderboard")
    rp.set_defaults(fn=cmd_report)

    y = sub.add_parser("toy")
    y.add_argument("--out", default=str(PKG_ROOT / "results" / "toy"))
    y.add_argument("--people", type=int, default=80)
    y.add_argument("--epochs", type=int, default=30)
    y.add_argument("--n-forget", type=int, default=24)
    y.set_defaults(fn=cmd_toy)

    args = ap.parse_args(argv)
    try:
        args.fn(args)
    except SystemExit as e:
        if not exit_ok:
            raise
        return e.code
    return 0
