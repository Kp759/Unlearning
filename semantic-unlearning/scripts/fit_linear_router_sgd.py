#!/usr/bin/env python3
"""Optimizer ablation for the linear-classifier router: minibatch SGD vs L-BFGS.

This runs fit_linear_router.py unchanged (same dataset, splits, negatives,
feature map, CV folds, calibration, bias folding, artifact, parity checks).
Exactly one function is replaced: linear_router._fit_heads, the stage-1
solver for

    min_{W,b}  sum_h sum_i pw_ih * BCE(w_h . phi_i + b_h, y_ih)
               + (l2 / 2) ||W||^2 + (1e-6 / 2) ||b||^2

(pw = the shipped masked, class-balanced pair weights; each head's sum to 1).
L-BFGS (shipped): full batch, strong-Wolfe line search, float64.
SGD (this script): torch.optim.SGD on minibatches of prompts, with the
unbiased estimate (n / |B|) * sum_{i in B} of the data term, same objective,
same zero init, same float64.

    # SGD, hyperparameters and every fit_linear_router.py flag copied from an
    # existing L-BFGS router (L2 and PCA pinned to its CV selection):
    python -u scripts/fit_linear_router_sgd.py \
        --like outputs/mcf_multiseed_regular_v1/seed1/L19/router \
        --output-dir outputs/optimizer_ablation_v1/mcf/seed1/L19/router_sgd

    # the same command with --optimizer lbfgs is the noise-floor control
    # (a second L-BFGS fit through the identical code path).

Flags not listed below are passed through to fit_linear_router.py and
override the copied ones. `--cv-grid full` re-runs the full (L2, PCA) grid
with SGD instead of pinning, i.e. "what would SGD have selected".

Outputs: the normal fit_linear_router.py run dir, plus optimizer_ablation.json
(optimizer settings, effective argv, and an in-process L-BFGS twin fit on the
same features for the final model: objective gap, weight cosine, logit gap).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))

import linear_router  # noqa: E402  (the module fit_linear_router.py imports)

LBFGS_FIT_HEADS = linear_router._fit_heads


# ---------------------------------------------------------------------------
# The SGD solver (same signature and info contract as linear_router._fit_heads)
# ---------------------------------------------------------------------------

def _objective(phi, target, pair_weight, weight, bias, l2, index=None):
    """The shipped objective; on a minibatch `index`, an unbiased estimate."""
    if index is None:
        rows, t, pw, scale = phi, target, pair_weight, 1.0
    else:
        rows, t, pw = phi[index], target[index], pair_weight[index]
        scale = phi.shape[0] / max(int(index.numel()), 1)
    logits = rows @ weight.T + bias
    loss = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
    data = (loss * pw).sum() * scale
    penalty = 0.5 * float(l2) * (weight * weight).sum()
    return data + penalty + 0.5 * linear_router._BIAS_PENALTY * (bias * bias).sum()


def smoothness_bounds(phi, pair_weight, l2, batch_size):
    """Upper bounds on the Lipschitz constant of the gradient.

    BCE'' <= 1/4, so head h's Hessian is <= (1/4) sum_i pw_ih phi~_i phi~_i^T
    (phi~ = [phi, 1]); its trace bounds lambda_max. `full`: the full-batch
    objective. `minibatch`: the worst minibatch of `batch_size` prompts for
    the (n/|B|)-scaled estimate (top-|B| terms per head). A step of 1/L_mb is
    stable for every batch SGD can draw.
    """
    n = phi.shape[0]
    sq = (phi * phi).sum(dim=1) + 1.0                      # ||phi~_i||^2  [n]
    terms = pair_weight * sq[:, None]                       # [n, H]
    full = 0.25 * float(terms.sum(dim=0).max()) + float(l2)
    k = min(int(batch_size), n)
    top = terms.topk(k, dim=0).values.sum(dim=0)            # worst batch per head
    minibatch = 0.25 * (n / k) * float(top.max()) + float(l2)
    return {"full_batch": full, "minibatch_worst_case": minibatch}


def sgd_fit_heads(phi, labels, eligible, l2, balance, max_iter, tolerance, *, config):
    """Drop-in for linear_router._fit_heads: (weight, bias, info)."""
    with torch.enable_grad():
        return _sgd_fit_heads(phi, labels, eligible, l2, balance, max_iter, tolerance, config)


def _sgd_fit_heads(phi, labels, eligible, l2, balance, max_iter, tolerance, config):
    started = time.time()
    dtype = torch.float64 if config["dtype"] == "float64" else torch.float32
    phi = phi.to(dtype)
    device = phi.device
    n, d = phi.shape
    n_heads = labels.shape[1]
    pair_weight = linear_router._pair_weights(labels, eligible, balance).to(dtype)
    target = labels.to(dtype)
    weight = torch.zeros((n_heads, d), dtype=dtype, device=device, requires_grad=True)
    bias = torch.zeros(n_heads, dtype=dtype, device=device, requires_grad=True)

    batch = max(1, min(int(config["batch_size"]), n))
    bounds = smoothness_bounds(phi.detach(), pair_weight, l2, batch)
    if config["lr"] == "auto":
        lr = float(config["lr_scale"]) / bounds["minibatch_worst_case"]
        lr_rule = f"{config['lr_scale']} / L_minibatch_worst_case"
    else:
        lr, lr_rule = float(config["lr"]), "fixed"
    momentum = float(config["momentum"])
    optimizer = torch.optim.SGD(
        [weight, bias], lr=lr, momentum=momentum,
        nesterov=bool(config["nesterov"]) and momentum > 0,
    )
    steps_per_epoch = math.ceil(n / batch)
    total_steps = int(config["epochs"]) * steps_per_epoch
    schedule = config["schedule"]
    if schedule == "cosine":
        factor = lambda s: 0.5 * (1.0 + math.cos(math.pi * min(s, total_steps) / total_steps))
    elif schedule == "step":
        factor = lambda s: 0.1 ** ((s >= 0.5 * total_steps) + (s >= 0.75 * total_steps))
    else:
        factor = lambda s: 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    generator = torch.Generator().manual_seed(int(config["seed"]))

    def full_state():
        weight.grad, bias.grad = None, None
        value = _objective(phi, target, pair_weight, weight, bias, l2)
        value.backward()
        grad = float(torch.cat([weight.grad.flatten(), bias.grad.flatten()]).abs().max())
        weight.grad, bias.grad = None, None
        return float(value.detach()), grad

    history, log_every = [], max(1, int(config["log_every"]))
    for epoch in range(int(config["epochs"])):
        order = torch.randperm(n, generator=generator).to(device)
        for start in range(0, n, batch):
            index = order[start:start + batch]
            optimizer.zero_grad(set_to_none=True)
            _objective(phi, target, pair_weight, weight, bias, l2, index).backward()
            optimizer.step()
            scheduler.step()
        if (epoch + 1) % log_every == 0 or epoch + 1 == int(config["epochs"]):
            value, grad = full_state()
            history.append({"epoch": epoch + 1, "objective": value, "max_abs_gradient": grad,
                            "lr": optimizer.param_groups[0]["lr"]})
            if not math.isfinite(value):
                raise FloatingPointError(
                    f"SGD diverged at epoch {epoch + 1} (lr={lr:.3g}); lower --sgd-lr-scale"
                )
    optimizer.zero_grad(set_to_none=True)
    value, grad = full_state()
    info = {
        "optimizer": "sgd",
        "objective": value,
        "max_abs_gradient": grad,
        # Same criterion as the L-BFGS fit, so the flag means the same thing.
        "converged": grad <= max(float(tolerance) * 100, 1e-6),
        "sgd": {
            "lr": lr,
            "lr_rule": lr_rule,
            "smoothness_bounds": bounds,
            "momentum": momentum,
            "nesterov": bool(config["nesterov"]) and momentum > 0,
            "batch_size": batch,
            "epochs": int(config["epochs"]),
            "steps": total_steps,
            "schedule": schedule,
            "seed": int(config["seed"]),
            "dtype": config["dtype"],
            "fit_rows": n,
            "seconds": round(time.time() - started, 3),
            "objective_history": history,
        },
    }
    if config.get("twin", True):
        info["lbfgs_twin_same_features"] = lbfgs_twin(
            phi.double(), labels, eligible, l2, balance, max_iter, tolerance,
            weight.detach().double(), bias.detach().double(), value,
        )
    return weight.detach(), bias.detach(), info


@torch.no_grad()
def _twin_logits(phi, weight, bias):
    return phi @ weight.T + bias


def lbfgs_twin(phi, labels, eligible, l2, balance, max_iter, tolerance, w_sgd, b_sgd, obj_sgd):
    """The shipped L-BFGS fit on the SAME features: optimizer-only difference.

    Unaffected by query-extraction noise, so any gap here is the optimizer.
    """
    with torch.enable_grad():
        w_ref, b_ref, info_ref = LBFGS_FIT_HEADS(phi, labels, eligible, l2, balance,
                                                 max_iter, tolerance)
    w_ref, b_ref = w_ref.double(), b_ref.double()
    elig = eligible.bool()
    lab = labels.bool() & elig
    z_ref, z_sgd = _twin_logits(phi, w_ref, b_ref), _twin_logits(phi, w_sgd, b_sgd)
    cos = F.cosine_similarity(w_sgd, w_ref, dim=1)
    rel = (w_sgd - w_ref).norm(dim=1) / w_ref.norm(dim=1).clamp_min(1e-12)
    gap = obj_sgd - float(info_ref["objective"])
    return {
        "lbfgs_objective": float(info_ref["objective"]),
        "lbfgs_max_abs_gradient": float(info_ref["max_abs_gradient"]),
        "lbfgs_converged": bool(info_ref["converged"]),
        "objective_gap_sgd_minus_lbfgs": gap,
        "relative_objective_gap": gap / max(abs(float(info_ref["objective"])), 1e-12),
        "weight_cosine": {"min": float(cos.min()), "median": float(cos.median()),
                          "mean": float(cos.mean())},
        "relative_weight_l2_diff": {"max": float(rel.max()), "median": float(rel.median())},
        "weight_norm_ratio_sgd_over_lbfgs": float(w_sgd.norm() / w_ref.norm().clamp_min(1e-12)),
        "bias_abs_diff_max": float((b_sgd - b_ref).abs().max()),
        "fit_logit_abs_diff": {
            "max": float((z_sgd - z_ref)[elig].abs().max()) if bool(elig.any()) else 0.0,
            "mean": float((z_sgd - z_ref)[elig].abs().mean()) if bool(elig.any()) else 0.0,
        },
        "fit_pairs_sign_disagreement_at_zero": int(((z_sgd >= 0) != (z_ref >= 0))[elig].sum()),
        "fit_errors_at_zero": {
            "sgd": int(((z_sgd >= 0) != lab)[elig].sum()),
            "lbfgs": int(((z_ref >= 0) != lab)[elig].sum()),
        },
        "note": ("Stage-1 logits on the fit split only; routes after calibration are "
                 "compared by compare_router_optimizers.py."),
    }


# ---------------------------------------------------------------------------
# CLI: rebuild the reference fit command, pin its hyperparameters, run it
# ---------------------------------------------------------------------------

def _strip_option(argv, name):
    out, skip = [], False
    for token in argv:
        if skip:
            skip = False
            continue
        if token == name:
            skip = True
            continue
        if token.startswith(name + "="):
            continue
        out.append(token)
    return out


def reference_argv(like_dir):
    """fit_linear_router.py flags of an existing run, minus its output dir."""
    report_path = Path(like_dir) / "linear_router_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"Missing {report_path}")
    report = json.loads(report_path.read_text())
    command = list(report.get("command") or [])
    if not command:
        raise ValueError(f"{report_path} has no recorded command; pass the flags explicitly")
    if command[0].endswith(".py"):
        command = command[1:]
    command = _strip_option(command, "--output-dir")
    fit = report["router_fit"]
    return command, {"l2": float(fit["selected_l2"]), "pca_dim": int(fit["selected_pca_dim"])}


def build_parser():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,  # unknown flags pass through to fit_linear_router.py
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument("--like", default=None,
                   help="existing L-BFGS router dir: copy its fit_linear_router.py flags")
    p.add_argument("--cv-grid", choices=("pinned", "full"), default="pinned",
                   help="pinned: L2/PCA = the --like run's CV selection (optimizer-only "
                        "comparison); full: rerun the CV grid with this optimizer")
    p.add_argument("--optimizer", choices=("sgd", "lbfgs"), default="sgd",
                   help="lbfgs = the shipped solver through this same path (control)")
    p.add_argument("--sgd-lr", default="auto",
                   help="'auto' = lr-scale / worst-case minibatch smoothness, or a number")
    p.add_argument("--sgd-lr-scale", type=float, default=1.0)
    p.add_argument("--sgd-momentum", type=float, default=0.9)
    p.add_argument("--sgd-nesterov", action="store_true")
    p.add_argument("--sgd-batch-size", type=int, default=64)
    p.add_argument("--sgd-epochs", type=int, default=300)
    p.add_argument("--sgd-schedule", choices=("cosine", "step", "constant"), default="cosine")
    p.add_argument("--sgd-seed", type=int, default=0)
    p.add_argument("--sgd-dtype", choices=("float64", "float32"), default="float64")
    p.add_argument("--sgd-log-every", type=int, default=10)
    p.add_argument("--no-lbfgs-twin", action="store_true",
                   help="skip the in-process L-BFGS fit on the same features")
    return p


def build_fit_argv(like, cv_grid, output_dir, passthrough):
    """(argv for fit_linear_router.main, pinned hyperparameters or None)."""
    base, pinned = [], None
    if like:
        base, pinned = reference_argv(like)
    fit_argv = list(base) + ["--output-dir", str(output_dir)]
    if pinned is not None and cv_grid == "pinned":
        fit_argv = _strip_option(_strip_option(fit_argv, "--lambdas"), "--pca-dims")
        fit_argv += ["--lambdas", repr(pinned["l2"]), "--pca-dims", str(pinned["pca_dim"])]
    fit_argv += list(passthrough)
    return fit_argv, pinned


def main(argv=None):
    parser = build_parser()
    args, passthrough = parser.parse_known_args(argv)
    if args.sgd_lr != "auto":
        float(args.sgd_lr)  # fail early on a typo

    fit_argv, pinned = build_fit_argv(args.like, args.cv_grid, args.output_dir, passthrough)
    if "--run-dir" not in fit_argv and not any(t.startswith("--run-dir=") for t in fit_argv):
        parser.error("no --run-dir: pass --like <router dir> or --run-dir <prep dir>")

    config = {
        "lr": args.sgd_lr, "lr_scale": args.sgd_lr_scale, "momentum": args.sgd_momentum,
        "nesterov": args.sgd_nesterov, "batch_size": args.sgd_batch_size,
        "epochs": args.sgd_epochs, "schedule": args.sgd_schedule, "seed": args.sgd_seed,
        "dtype": args.sgd_dtype, "log_every": args.sgd_log_every,
        "twin": not args.no_lbfgs_twin,
    }
    if args.optimizer == "sgd":
        linear_router._fit_heads = (
            lambda phi, labels, eligible, l2, balance, max_iter, tolerance:
            sgd_fit_heads(phi, labels, eligible, l2, balance, max_iter, tolerance, config=config)
        )
    print(json.dumps({"phase": "optimizer_ablation", "optimizer": args.optimizer,
                      "cv_grid": args.cv_grid, "pinned": pinned,
                      "fit_argv": fit_argv}), flush=True)

    import fit_linear_router as cli  # after the patch; resolves _fit_heads at call time
    started = time.time()
    try:
        status = cli.main(fit_argv)
    finally:
        linear_router._fit_heads = LBFGS_FIT_HEADS

    output = Path(args.output_dir).resolve()
    report_path = output / "linear_router_report.json"
    report = json.loads(report_path.read_text())
    fit_info = report["router_fit"]["fit_info"]
    record = {
        "schema_version": "router_optimizer_ablation_v1",
        "optimizer": args.optimizer,
        "sgd_config": config if args.optimizer == "sgd" else None,
        "cv_grid": args.cv_grid,
        "like": None if args.like is None else str(Path(args.like).resolve()),
        "pinned_hyperparameters": pinned if args.cv_grid == "pinned" else None,
        "selected_l2": report["router_fit"]["selected_l2"],
        "selected_pca_dim": report["router_fit"]["selected_pca_dim"],
        "fit_argv": fit_argv,
        "final_fit": {k: v for k, v in fit_info.items()
                      if k in ("optimizer", "objective", "max_abs_gradient", "converged",
                               "lbfgs_iterations", "lbfgs_function_evaluations")},
        "sgd": fit_info.get("sgd"),
        "lbfgs_twin_same_features": fit_info.get("lbfgs_twin_same_features"),
        "seconds_total": round(time.time() - started, 1),
    }
    record["final_fit"].setdefault("optimizer", "lbfgs")
    report["optimizer_ablation"] = record
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (output / "optimizer_ablation.json").write_text(
        json.dumps(record, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({"phase": "optimizer_ablation_done", **{
        k: record[k] for k in ("optimizer", "selected_l2", "selected_pca_dim", "final_fit")
    }, "twin": record["lbfgs_twin_same_features"] and {
        k: record["lbfgs_twin_same_features"][k] for k in (
            "relative_objective_gap", "weight_cosine", "fit_pairs_sign_disagreement_at_zero")
    }}, indent=2), flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
