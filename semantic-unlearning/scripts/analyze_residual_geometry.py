#!/usr/bin/env python3
"""Geometry of the trained residual bank: PCA, UMAP, norms, small-norm arms.

The bank is N residual vectors of width d (N=50, d=3072 for MCF/zsRE/RWKU;
N=105 for MQuAKE). Three questions this answers:

1. Do the residuals occupy a low-dimensional shared subspace, or is each one
   an independent direction? PCA spectrum answers this. A steep spectrum means
   the N rows are really k << N directions, which is an argument for a shared
   low-rank actuator instead of a per-fact row -- and a parameter-count result.

2. Do rows cluster by relation? Both PCA and UMAP place the N rows in 2D;
   colouring by relation shows whether the actuator learned relation-generic
   suppression directions or fact-specific ones. UMAP is the better view here
   because PCA can only show the two highest-variance directions, and with
   N=50 points in 3072 dimensions those two directions explain very little.
   Read them together, never UMAP alone -- UMAP distances between clusters are
   not meaningful, only the grouping is.

3. How large is the intervention? ||dE_i|| is reported both absolutely and
   relative to the typical ||h_19|| at the intervention site, because a
   residual that is large compared with the hidden state it is added to is a
   weak claim to "minimal intervention". `--emit-scaled` writes artifacts with
   rows rescaled to a norm budget so the existing evaluators produce a
   suppression-vs-norm curve with no retraining.

UMAP settings note: with N=50, `n_neighbors` must be small (5-15) or UMAP
degenerates to a global embedding. The default of 15 is roughly N/3, and
`--umap-neighbors` sweeps it. Cosine is the right metric -- the router scores
cosines, so Euclidean geometry here would not match the decision rule.

Usage
-----
python -u scripts/analyze_residual_geometry.py \
  --artifact outputs/<run>/fact_association_embeddings.pt \
  --output-dir outputs/<run>/geometry \
  --umap-neighbors 5,15,30 --emit-scaled 0.25,0.5,0.75
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def _load_rows(artifact_path):
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    rows = artifact["rows"].float()
    if rows.ndim != 2:
        raise ValueError(f"Expected a [N, d] residual bank, got {tuple(rows.shape)}")
    facts = artifact.get("facts", [])
    if facts and len(facts) != rows.shape[0]:
        raise ValueError("Fact metadata does not match the residual bank")
    return artifact, rows, facts


def _labels(facts, field):
    if not facts:
        return None
    return [str(fact.get(field, "")) for fact in facts]


def pca_2d(rows, center=True):
    """Deterministic full SVD; no randomized solver, so this is reproducible."""
    matrix = rows - rows.mean(dim=0, keepdim=True) if center else rows
    u, s, _ = torch.linalg.svd(matrix, full_matrices=False)
    coordinates = (u[:, :2] * s[:2]).numpy()
    variance = (s ** 2)
    ratio = (variance / variance.sum().clamp_min(1e-30)).numpy()
    return coordinates, ratio, s.numpy()


def umap_2d(rows, n_neighbors, min_dist=0.1, seed=0, metric="cosine"):
    try:
        import umap
    except ImportError as error:
        raise SystemExit(
            "umap-learn is not installed. pip install umap-learn"
        ) from error
    n_neighbors = int(min(int(n_neighbors), rows.shape[0] - 1))
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=float(min_dist),
        metric=str(metric),
        random_state=int(seed),
    )
    return reducer.fit_transform(rows.numpy()), n_neighbors


def norm_report(rows):
    norms = rows.norm(dim=-1)
    unit = torch.nn.functional.normalize(rows, dim=-1)
    cosine = unit @ unit.T
    off_diagonal = cosine[~torch.eye(rows.shape[0], dtype=torch.bool)]
    return {
        "row_count": int(rows.shape[0]),
        "hidden_size": int(rows.shape[1]),
        "norm_min": float(norms.min()),
        "norm_median": float(norms.median()),
        "norm_mean": float(norms.mean()),
        "norm_max": float(norms.max()),
        "norm_std": float(norms.std()),
        "zero_rows": int((norms < 1e-8).sum()),
        "pairwise_cosine_mean": float(off_diagonal.mean()),
        "pairwise_cosine_max": float(off_diagonal.max()),
        "pairwise_cosine_min": float(off_diagonal.min()),
        "effective_rank_participation_ratio": float(
            _participation_ratio(rows)
        ),
    }


def _participation_ratio(rows):
    """(sum s^2)^2 / sum s^4 -- a continuous stand-in for rank."""
    s = torch.linalg.svdvals(rows)
    squared = s ** 2
    return (squared.sum() ** 2) / (squared ** 2).sum().clamp_min(1e-30)


def scree(ratio, target=0.9):
    cumulative = np.cumsum(ratio)
    index = int(np.searchsorted(cumulative, float(target)) + 1)
    return min(index, len(ratio))


def plot(coordinates, labels, title, path, annotate=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(figsize=(7.0, 6.0), dpi=160)
    if labels is None:
        axes.scatter(coordinates[:, 0], coordinates[:, 1], s=42)
    else:
        unique = sorted(set(labels))
        palette = plt.get_cmap("tab20")(np.linspace(0, 1, max(len(unique), 2)))
        for index, label in enumerate(unique):
            selection = [i for i, value in enumerate(labels) if value == label]
            axes.scatter(
                coordinates[selection, 0],
                coordinates[selection, 1],
                s=42,
                color=palette[index],
                label=label if len(unique) <= 16 else None,
            )
        if len(unique) <= 16:
            axes.legend(fontsize=7, loc="best", frameon=False)
    if annotate is not None:
        for index, text in enumerate(annotate):
            axes.annotate(
                str(text),
                (coordinates[index, 0], coordinates[index, 1]),
                fontsize=5,
                alpha=0.6,
            )
    axes.set_title(title, fontsize=10)
    axes.set_xlabel("component 1")
    axes.set_ylabel("component 2")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def emit_scaled(artifact, rows, budgets, output_dir):
    """Write artifacts whose rows are clipped to a fraction of the median norm.

    This is the cheap version of the small-residual ablation: no retraining,
    it measures how much of the suppression survives a norm budget. The
    expensive version adds a norm penalty to the row objective; run this first
    to find out whether the penalty is worth a training sweep.
    """
    norms = rows.norm(dim=-1)
    reference = float(norms.median())
    emitted = []
    for budget in budgets:
        ceiling = float(budget) * reference
        scale = (ceiling / norms.clamp_min(1e-30)).clamp(max=1.0)
        scaled = rows * scale[:, None]
        record = dict(artifact)
        record["rows"] = scaled
        record["residual_norm_budget_fraction_of_median"] = float(budget)
        record["residual_norm_ceiling"] = ceiling
        record["residual_rows_clipped"] = int((scale < 1.0).sum())
        path = Path(output_dir) / f"rows_scaled_{budget:g}.pt"
        torch.save(record, path)
        emitted.append({
            "budget_fraction_of_median_norm": float(budget),
            "norm_ceiling": ceiling,
            "rows_clipped": int((scale < 1.0).sum()),
            "artifact": str(path),
        })
    return {"median_norm": reference, "arms": emitted}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label-field", default="relation")
    parser.add_argument("--umap-neighbors", default="5,15,30")
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--umap-metric", default="cosine")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-center", action="store_true")
    parser.add_argument("--annotate", action="store_true")
    parser.add_argument(
        "--emit-scaled",
        default="",
        help="comma-separated norm budgets as a fraction of the median row norm",
    )
    args = parser.parse_args(argv)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    artifact, rows, facts = _load_rows(args.artifact)
    labels = _labels(facts, args.label_field)
    annotate = _labels(facts, "subject") if args.annotate else None

    report = {
        "artifact": str(Path(args.artifact).resolve()),
        "architecture": artifact.get("architecture"),
        "layer": artifact.get("layer"),
        "norms": norm_report(rows),
    }

    coordinates, ratio, singular = pca_2d(rows, center=not args.no_center)
    report["pca"] = {
        "centered": not args.no_center,
        "explained_variance_ratio_top10": [float(x) for x in ratio[:10]],
        "explained_variance_top2": float(ratio[:2].sum()),
        "components_for_90pct_variance": scree(ratio, 0.9),
        "components_for_99pct_variance": scree(ratio, 0.99),
        "singular_values_top10": [float(x) for x in singular[:10]],
    }
    np.savetxt(output / "pca_coordinates.csv", coordinates, delimiter=",")
    plot(
        coordinates,
        labels,
        f"PCA of {rows.shape[0]} residual rows "
        f"({report['pca']['explained_variance_top2']:.1%} of variance)",
        output / "pca_residuals.png",
        annotate=annotate,
    )

    report["umap"] = []
    for value in [v for v in args.umap_neighbors.split(",") if v.strip()]:
        embedded, used = umap_2d(
            rows,
            int(value),
            min_dist=args.umap_min_dist,
            seed=args.seed,
            metric=args.umap_metric,
        )
        np.savetxt(
            output / f"umap_coordinates_k{used}.csv", embedded, delimiter=","
        )
        plot(
            embedded,
            labels,
            f"UMAP of {rows.shape[0]} residual rows "
            f"(k={used}, metric={args.umap_metric})",
            output / f"umap_residuals_k{used}.png",
            annotate=annotate,
        )
        report["umap"].append({
            "n_neighbors": used,
            "min_dist": args.umap_min_dist,
            "metric": args.umap_metric,
            "seed": args.seed,
            "coordinates": str(output / f"umap_coordinates_k{used}.csv"),
        })

    budgets = [float(v) for v in args.emit_scaled.split(",") if v.strip()]
    if budgets:
        report["scaled_arms"] = emit_scaled(artifact, rows, budgets, output)

    (output / "residual_geometry.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(
        {
            "status": "residual_geometry_complete",
            "output_dir": str(output),
            "row_count": report["norms"]["row_count"],
            "pca_top2_variance": report["pca"]["explained_variance_top2"],
            "components_for_90pct_variance": report["pca"][
                "components_for_90pct_variance"
            ],
            "effective_rank": report["norms"][
                "effective_rank_participation_ratio"
            ],
            "median_row_norm": report["norms"]["norm_median"],
        },
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
