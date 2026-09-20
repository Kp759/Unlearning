#!/usr/bin/env python3
"""Step 3: replace max-cosine with a learned, calibrated scorer.

The shipped score d_i = max_p cos(q,p) - max_n cos(q,n) is a 1-NN rule: its
value is set entirely by whichever single prototype happens to be nearest, so
one badly chosen negative moves the decision for a whole association. There is
no learned boundary, no feature weighting, and no notion of which directions
in h_19 carry the relation. It also emits a bare margin, so there is no ROC,
no operating point that can be moved after the fact, and no way to say "fire
only when at least 99% confident".

Two replacements, in the order worth trying them:

  logistic   one linear probe per association over q(x), or a shared probe
             over [q; q*k_i]. A small change from what exists, and it already
             buys a probability, an ROC, a tunable operating point, and a
             boundary that weights directions instead of trusting one nearest
             prototype. It sits in the existing research idiom -- probes on
             hidden states.

  bilinear   s_i(x) = q(x)^T W k_i with W = U V^T shared across associations,
             trained by InfoNCE with a learnable null key. No per-association
             parameters, so an unseen association scores from its key vector
             alone -- the answer to both "how do you set tau for association
             N+1?" and "what happens at N = 10,000?".

             *** Gated on Step 1's data pipeline. Do not report it yet. ***
             On a synthetic task with 20 associations, 12 positives and 40
             negatives each, it reaches training AUC 0.99+ and held-out AUC
             ~0.52 at every rank from 32 down to 1, with weight decay from
             1e-4 to 1e-1. Even rank 1 carries 2*d projection parameters
             against a few hundred prompts. At the real d=3072, rank 64 is
             393K parameters against roughly 650 prompts. It will fit the
             fitting split perfectly and tell you nothing, and the training
             curve looks healthy while it happens. The arm is kept here
             because it is the right architecture once there are enough
             prompts per association; it is not runnable evidence today.

`samples_per_parameter` and `overfitting_gap` are reported for every arm for
this reason, and a warning fires below --min-samples-per-parameter.

An MLP head is deliberately not offered at all: it fails the same way with
less to recommend it.

Calibration is not optional and is not a postprocessing step here: the
operating point is chosen on a held-out split to hit a target false-activation
rate, and reported with a Wilson interval, exactly as in
calibrate_router_operating_point.py.

Input: a torch file with
  queries      [P, d] float, the pre-intervention q(x) per prompt
  fact_index   [P] long, the association each prompt belongs to
  polarity     [P] long, 1 positive / 0 negative
  keys         [N, d] float, optional; association keys for the bilinear arm
               (mean of that association's positive prototypes if omitted)

Usage
-----
python -u scripts/train_router_scorer.py \
  --queries outputs/<run>/probes/queries.pt \
  --output-dir outputs/<run>/scorer \
  --arms logistic,bilinear --target-fpr 0.001 --rank 64
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def wilson(successes, total, z=1.96):
    if total == 0:
        return {"rate": None, "low": None, "high": None, "n": 0}
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    spread = (
        z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
    ) / denominator
    return {
        "rate": p,
        "low": max(0.0, centre - spread),
        "high": min(1.0, centre + spread),
        "n": int(total),
    }


def roc_auc(positive, negative):
    if not len(positive) or not len(negative):
        return None
    pos = torch.as_tensor(positive, dtype=torch.float64).flatten()
    neg = torch.as_tensor(negative, dtype=torch.float64).flatten()
    comparison = pos[:, None] - neg[None, :]
    wins = (comparison > 0).sum() + 0.5 * (comparison == 0).sum()
    return float(wins / (pos.numel() * neg.numel()))


def stratified_split(fact_index, polarity, holdout_fraction, seed):
    """Split per (association, polarity) so every cell appears on both sides."""
    generator = torch.Generator().manual_seed(int(seed))
    fit, held = [], []
    for fact in fact_index.unique().tolist():
        for label in (0, 1):
            rows = torch.nonzero(
                (fact_index == fact) & (polarity == label), as_tuple=False
            ).flatten()
            if rows.numel() == 0:
                continue
            shuffled = rows[torch.randperm(rows.numel(), generator=generator)]
            if shuffled.numel() == 1:
                fit.append(shuffled)
                continue
            cut = int(round(shuffled.numel() * (1.0 - float(holdout_fraction))))
            cut = max(1, min(shuffled.numel() - 1, cut))
            fit.append(shuffled[:cut])
            held.append(shuffled[cut:])
    return torch.cat(fit), (torch.cat(held) if held else torch.empty(0, dtype=torch.long))


def similarity_features(queries, keys, fact_index):
    """Low-dimensional, association-invariant features for the logistic arm.

    A probe over raw q does not work at this data scale and fails silently: on
    a synthetic task that is cosine-separable by construction, a probe over
    [q ; q*k_i] drives training loss from 1.07 to 0.23 while held-out AUC stays
    at 0.57 against a max-cosine baseline of 0.91. With 2*d free weights and a
    few hundred prompts it memorizes through the raw-q half instead of finding
    the q.k interaction, and nothing in the training curve reveals that.

    So the probe learns a boundary over the interpretable scores the router
    already computes, rather than over the hidden state. It calibrates the
    existing geometry instead of trying to relearn it, which is both the
    data-efficient choice and the more defensible one.

      cos_own        cosine to this association's key
      cos_best_other largest cosine to any other association's key
      margin         cos_own - cos_best_other
      cos_mean       mean cosine over all keys, a per-prompt difficulty offset
      cos_rank       fraction of keys this association outranks
      norm_gap       cos_own minus the mean, in units of the spread over keys
    """
    similarity = queries @ keys.T
    rows = torch.arange(similarity.shape[0], device=similarity.device)
    own = similarity[rows, fact_index]
    masked = similarity.clone()
    masked[rows, fact_index] = float("-inf")
    best_other = masked.max(dim=-1).values
    mean = similarity.mean(dim=-1)
    std = similarity.std(dim=-1).clamp_min(1e-6)
    rank = (similarity <= own[:, None]).float().mean(dim=-1)
    return torch.stack(
        [own, best_other, own - best_other, mean, rank, (own - mean) / std],
        dim=-1,
    )


FEATURE_NAMES = (
    "cos_own", "cos_best_other", "margin", "cos_mean", "cos_rank", "norm_gap",
)


class SharedLogisticScorer(nn.Module):
    """Logistic probe over similarity features, shared across associations.

    Six weights plus one shared bias. A per-association bias is deliberately
    omitted: with a handful of positives each it fits base rates rather than
    the boundary, and it would also break generalization to association N+1.
    """

    def __init__(self, hidden_size=None, num_facts=None):
        super().__init__()
        self.weight = nn.Linear(len(FEATURE_NAMES), 1)
        nn.init.zeros_(self.weight.weight)
        nn.init.zeros_(self.weight.bias)

    def forward(self, queries, keys, fact_index):
        features = similarity_features(queries, keys, fact_index)
        return self.weight(features).squeeze(-1)

    def coefficients(self):
        return {
            name: float(value)
            for name, value in zip(
                FEATURE_NAMES, self.weight.weight.detach().flatten()
            )
        }


class BilinearScorer(nn.Module):
    """s_i(x) = q^T (U V^T) k_i, low-rank and shared across associations.

    No per-association parameters at all, so an unseen association scores
    correctly from its key vector alone. That is the property that answers
    both the N+1 question and the scaling question.
    """

    def __init__(self, hidden_size, rank=64, scale=14.0, max_scale=100.0):
        super().__init__()
        hidden_size, rank = int(hidden_size), int(rank)
        self.left = nn.Parameter(torch.randn(hidden_size, rank) / math.sqrt(hidden_size))
        self.right = nn.Parameter(torch.randn(hidden_size, rank) / math.sqrt(hidden_size))
        # CLIP-style learnable temperature. It is the ONLY scaling applied:
        # dividing these scores by a separate temperature as well puts logits
        # around 140 at initialization, which saturates the softmax and leaves
        # InfoNCE with no usable gradient.
        self.logit_scale = nn.Parameter(torch.tensor(float(scale)).log())
        self.max_scale = float(max_scale)

        # A learnable "none of the above" key. Negatives are trained to select
        # it, which folds hard negatives into the same cross-entropy as the
        # positives. Scoring negatives with a separate BCE-toward-zero term
        # instead sets up a tug of war -- InfoNCE needs large-magnitude logits,
        # BCE-toward-zero wants them near zero -- and the temperature collapses.
        self.null_key = nn.Parameter(torch.randn(rank) / math.sqrt(rank))

    def scale(self):
        return self.logit_scale.exp().clamp(max=self.max_scale)

    def all_scores(self, queries, keys, include_null=False):
        left = F.normalize(queries @ self.left, dim=-1)
        right = F.normalize(keys @ self.right, dim=-1)
        if include_null:
            right = torch.cat(
                [right, F.normalize(self.null_key, dim=-1)[None, :]], dim=0
            )
        return self.scale() * (left @ right.T)

    def forward(self, queries, keys, fact_index):
        scores = self.all_scores(queries, keys)
        return scores[torch.arange(scores.shape[0], device=scores.device), fact_index]


def train_logistic(queries, keys, fact_index, polarity, rows, epochs, lr, weight_decay,
                   device):
    model = SharedLogisticScorer(queries.shape[1], keys.shape[0]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(lr), weight_decay=float(weight_decay)
    )
    q, k = queries[rows].to(device), keys.to(device)
    f, y = fact_index[rows].to(device), polarity[rows].float().to(device)
    # Rare-positive correction: without it the probe can reach high accuracy by
    # predicting "negative" everywhere, which is exactly the degenerate router.
    positive_weight = torch.tensor(
        max((y == 0).sum().item(), 1) / max((y == 1).sum().item(), 1), device=device
    )
    for _ in range(int(epochs)):
        optimizer.zero_grad()
        logits = model(q, k, f)
        loss = F.binary_cross_entropy_with_logits(
            logits, y, pos_weight=positive_weight
        )
        loss.backward()
        optimizer.step()
    return model, float(loss.detach())


def train_bilinear(queries, keys, fact_index, polarity, rows, epochs, lr, weight_decay,
                   rank, temperature, device):
    model = BilinearScorer(queries.shape[1], rank=rank).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(lr), weight_decay=float(weight_decay)
    )
    positive_rows = rows[polarity[rows] == 1]
    if positive_rows.numel() == 0:
        raise SystemExit("InfoNCE needs positive (query, association) pairs")
    q_pos = queries[positive_rows].to(device)
    f_pos = fact_index[positive_rows].to(device)
    negative_rows = rows[polarity[rows] == 0]
    q_neg = queries[negative_rows].to(device)
    f_neg = fact_index[negative_rows].to(device)
    k = keys.to(device)

    null_index = int(k.shape[0])
    for _ in range(int(epochs)):
        optimizer.zero_grad()
        # One cross-entropy over [associations, null]. Positives must select
        # their own key; negatives must select the null key. all_scores already
        # applies the learnable temperature, so no further division here.
        queries_all = q_pos if q_neg.numel() == 0 else torch.cat([q_pos, q_neg])
        targets = (
            f_pos
            if q_neg.numel() == 0
            else torch.cat([
                f_pos,
                torch.full_like(f_neg, null_index),
            ])
        )
        logits = model.all_scores(queries_all, k, include_null=True)
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()
    return model, float(loss.detach())


def evaluate(model, queries, keys, fact_index, polarity, rows, target_fpr, device):
    with torch.no_grad():
        scores = model(
            queries[rows].to(device), keys.to(device), fact_index[rows].to(device)
        ).cpu()
    labels = polarity[rows]
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    ordered = torch.sort(negatives, descending=True).values
    allowed = int(math.floor(float(target_fpr) * ordered.numel()))
    if ordered.numel() == 0:
        threshold = float("-inf")
    elif allowed >= ordered.numel():
        threshold = float(ordered.min()) - 1e-6
    else:
        threshold = float(ordered[allowed]) + 1e-9
    return {
        "roc_auc": roc_auc(positives, negatives),
        "operating_point": threshold,
        "target_fpr": float(target_fpr),
        "recall_at_operating_point": wilson(
            int((positives >= threshold).sum()), int(positives.numel())
        ),
        "false_activation_at_operating_point": wilson(
            int((negatives >= threshold).sum()), int(negatives.numel())
        ),
        "positive_count": int(positives.numel()),
        "negative_count": int(negatives.numel()),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--arms", default="maxcos,logistic,bilinear")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--target-fpr", type=float, default=0.001)
    parser.add_argument("--holdout-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--min-samples-per-parameter",
        type=float,
        default=10.0,
        help="warn below this; a scorer under it can memorize the fitting split",
    )
    args = parser.parse_args(argv)

    torch.manual_seed(int(args.seed))
    payload = torch.load(args.queries, map_location="cpu", weights_only=False)
    queries = F.normalize(payload["queries"].float(), dim=-1)
    fact_index = payload["fact_index"].long()
    polarity = payload["polarity"].long()
    if "keys" in payload:
        keys = F.normalize(payload["keys"].float(), dim=-1)
    else:
        num_facts = int(fact_index.max()) + 1
        keys = torch.zeros(num_facts, queries.shape[1])
        for fact in range(num_facts):
            rows = (fact_index == fact) & (polarity == 1)
            if rows.any():
                keys[fact] = queries[rows].mean(dim=0)
        keys = F.normalize(keys, dim=-1)

    fit_rows, held_rows = stratified_split(
        fact_index, polarity, args.holdout_fraction, args.seed
    )
    if held_rows.numel() == 0:
        raise SystemExit("Held-out split is empty; lower --holdout-fraction")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    report = {
        "schema_version": "router_scorer_v1",
        "queries": str(Path(args.queries).resolve()),
        "prompt_count": int(queries.shape[0]),
        "hidden_size": int(queries.shape[1]),
        "association_count": int(keys.shape[0]),
        "fitting_rows": int(fit_rows.numel()),
        "held_out_rows": int(held_rows.numel()),
        "target_fpr": args.target_fpr,
        "arms": {},
    }

    for arm in arms:
        if arm == "maxcos":
            # The shipped rule as a baseline: cosine to the association key,
            # with no learning. Present so every other arm has something to
            # beat on the same held-out split.
            class _Cosine(nn.Module):
                def forward(self, q, k, f):
                    return (q * k[f]).sum(dim=-1)

            model, loss = _Cosine(), None
            parameters = 0
        elif arm == "logistic":
            model, loss = train_logistic(
                queries, keys, fact_index, polarity, fit_rows,
                args.epochs, args.lr, args.weight_decay, args.device,
            )
            parameters = sum(p.numel() for p in model.parameters())
        elif arm == "bilinear":
            model, loss = train_bilinear(
                queries, keys, fact_index, polarity, fit_rows,
                args.epochs, args.lr, args.weight_decay, args.rank,
                args.temperature, args.device,
            )
            parameters = sum(p.numel() for p in model.parameters())
        else:
            raise SystemExit(f"Unknown arm: {arm}")

        model = model.to(args.device).eval()
        fitting = evaluate(
            model, queries, keys, fact_index, polarity, fit_rows,
            args.target_fpr, args.device,
        )
        held = evaluate(
            model, queries, keys, fact_index, polarity, held_rows,
            args.target_fpr, args.device,
        )
        # None rather than inf for the zero-parameter baseline: inf is not
        # JSON-serializable and "unbounded" is the honest reading anyway.
        samples_per_parameter = (
            float(fit_rows.numel()) / parameters if parameters else None
        )
        # The gap between fitting and held-out AUC is the only thing that makes
        # memorization visible: a scorer that has memorized shows a healthy
        # falling loss and a near-perfect fitting AUC the whole way down.
        overfitting_gap = (
            fitting["roc_auc"] - held["roc_auc"]
            if fitting["roc_auc"] is not None and held["roc_auc"] is not None
            else None
        )
        warnings = []
        if (
            samples_per_parameter is not None
            and samples_per_parameter < float(args.min_samples_per_parameter)
        ):
            warnings.append(
                f"{samples_per_parameter:.3f} fitting samples per parameter is "
                f"below {args.min_samples_per_parameter}; this arm can fit the "
                f"fitting split without learning anything transferable. Do not "
                f"report it until Step 1 raises the prompt count."
            )
        if overfitting_gap is not None and overfitting_gap > 0.15:
            warnings.append(
                f"fitting AUC exceeds held-out AUC by {overfitting_gap:.3f}: "
                f"memorization, not generalization."
            )
        report["arms"][arm] = {
            "final_loss": loss,
            "trainable_parameters": parameters,
            "parameters_per_association": (
                parameters / max(int(keys.shape[0]), 1)
            ),
            "samples_per_parameter": samples_per_parameter,
            "overfitting_gap_auc": overfitting_gap,
            "warnings": warnings,
            "reportable": not warnings,
            "generalizes_to_unseen_association": arm in ("maxcos", "bilinear"),
            "fitting": fitting,
            "held_out": held,
        }
        if arm == "logistic":
            report["arms"][arm]["coefficients"] = model.coefficients()
        for warning in warnings:
            print(f"  WARNING [{arm}]: {warning}", flush=True)
        if arm != "maxcos":
            torch.save(
                {
                    "arm": arm,
                    "state_dict": model.state_dict(),
                    "keys": keys,
                    "hidden_size": int(queries.shape[1]),
                    "rank": args.rank if arm == "bilinear" else None,
                    # The operating point is chosen on held-out data and
                    # travels with the weights: a scorer without its threshold
                    # is not deployable.
                    "operating_point": report["arms"][arm]["held_out"][
                        "operating_point"
                    ],
                    "target_fpr": args.target_fpr,
                },
                output / f"scorer_{arm}.pt",
            )

    (output / "router_scorer.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps({
        "status": "scorer_complete",
        "held_out": {
            arm: {
                "roc_auc": block["held_out"]["roc_auc"],
                "recall_at_target_fpr": block["held_out"][
                    "recall_at_operating_point"
                ]["rate"],
                "false_activation": block["held_out"][
                    "false_activation_at_operating_point"
                ]["rate"],
                "parameters": block["trainable_parameters"],
                "samples_per_parameter": block["samples_per_parameter"],
                "overfitting_gap_auc": block["overfitting_gap_auc"],
                "reportable": block["reportable"],
                "handles_new_association": block[
                    "generalizes_to_unseen_association"
                ],
            }
            for arm, block in report["arms"].items()
        },
        "output": str(output / "router_scorer.json"),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
