#!/usr/bin/env python3
"""Raw logistic regression router: same heads, stage-1 bias, fire at p >= 0.5.

The shipped router is a two-stage fit: stage 1 learns w_i, b_i by BCE on the
training templates; stage 2 moves the calibrated cutoff t (chosen on held-out
calibration prompts) into the bias, b'_i = b_i - t, and fires at z' >= 0.
This script builds the one-stage counterpart from a folded router, with
nothing else changed:

    folded   z'_i = w_i.phi + (b_i - t)   fires at z' >= 0   (shipped)
    raw      z_i  = w_i.phi +  b_i        fires at z  >= 0   (plain logistic regression)

Same weights, feature map, subject eligibility, top-1 rule and ambiguity
margin, rows untouched (zero for a fresh router dir). The output loads in
every trainer and evaluator (decision rule `explicit_threshold` at 0.0).

    python -u scripts/make_raw_logistic_router.py \
        --router-dir outputs/mcf_multiseed_regular_v1/seed1/L19/router \
        --output-dir outputs/bias_rule_ablation_v1/lbfgs/mcf/seed1/L19/router_raw \
        --device cuda --local-files-only

With the base model (default) it also re-extracts every router prompt
(fit / calibration / audit) and scores BOTH rules on the same features:
route outcomes per split, positives the calibration rescues (own-head logit
in [t, 0)), negatives it adds, logit quantiles, and runtime parity of the raw
artifact. `--skip-routes` writes the artifact only (no GPU).
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import shutil
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from linear_router import (  # noqa: E402
    ARCHITECTURE,
    decide_routes,
    routing_policy_name,
    route_outcomes,
    score_queries,
)

RAW_RULE = "raw_logistic_stage1_bias_p_ge_0.5"


def _json_safe(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, torch.Tensor):
        return _json_safe(value.tolist())
    return value


def folded_cutoff(artifact):
    """(stage-1 bias [N], cutoff t as a float or [N] tensor) of a threshold-gate router."""
    if str(artifact.get("architecture")) != ARCHITECTURE:
        raise ValueError("not a linear-classifier router artifact")
    if str(artifact.get("gate_mode", "threshold")) != "threshold":
        raise ValueError("the subject gate has no cutoff; raw vs folded needs the threshold gate")
    if artifact.get("head_index") is not None:
        raise ValueError("shared (relation) heads are not supported here")
    calibration = artifact.get("bias_calibration")
    if calibration is not None:
        stage1 = torch.as_tensor(calibration["stage1_bias"]).float()
        shift = torch.as_tensor(calibration["shift"]).float()
        if not torch.allclose(stage1 - shift, artifact["router_bias"].float(), atol=1e-4):
            raise ValueError("bias_calibration does not reproduce the deployed bias")
        uniform = bool(torch.all(shift == shift[0]))
        return stage1, (float(shift[0]) if uniform else shift)
    stage1 = artifact["router_bias"].float()
    per_head = artifact.get("per_head_thresholds")
    return stage1, (float(artifact["threshold"]) if per_head is None
                    else torch.as_tensor(per_head).float())


def raw_artifact(folded):
    """The folded router with its calibration removed: b = stage-1 bias, cutoff 0."""
    stage1, cutoff = folded_cutoff(folded)
    raw = dict(folded)
    raw["router_bias"] = stage1.clone()
    raw["threshold"] = 0.0
    raw["per_head_thresholds"] = None
    raw["bias_calibration"] = None
    raw["threshold_policy"] = "global"
    raw["decision_rule"] = "explicit_threshold"
    raw["routing_policy"] = routing_policy_name("threshold", "explicit_threshold", "global")
    raw["bias_rule"] = {
        "rule": RAW_RULE,
        "runtime_rule": "fire the best subject-eligible head if sigmoid(w.phi + b) >= 0.5 "
                        "(z >= 0), b = stage-1 bias, no held-out calibration",
        "folded_cutoff_removed": _json_safe(cutoff if isinstance(cutoff, float)
                                            else cutoff.tolist()),
        "weights_changed": False,
        "rows_changed": False,
    }
    router_fit = copy.deepcopy(raw.get("router_fit") or {})
    router_fit.update({
        "decision_rule": "explicit_threshold",
        "bias_rule": RAW_RULE,
        "runtime_threshold_logit": 0.0,
        "calibrated_cutoff_folded_into_bias": False,
        "bias_shift": None,
        "folded_router_threshold_logit": router_fit.get("threshold_logit"),
        "threshold_logit": 0.0,
    })
    raw["router_fit"] = router_fit
    return raw, stage1, cutoff


def _quantiles(values):
    values = values.flatten().float()
    values = values[torch.isfinite(values)]
    if values.numel() == 0:
        return None
    q = torch.quantile(values, torch.tensor([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]))
    return dict(zip(("min", "p10", "p25", "median", "p75", "p90", "max"), q.tolist()))


def rule_diagnostics(z_raw, eligible, owner, splits, cutoff, margin, facts):
    """Both rules on the same stage-1 logits, per split.

    raw: z >= 0. folded: z >= t (identical to z - t >= 0). Positives are
    prompts with an owner; negatives are controls (owner -1).
    """
    eligible = eligible.bool()
    t = cutoff if isinstance(cutoff, float) else torch.as_tensor(cutoff).float()
    raw_d = decide_routes(z_raw, eligible, 0.0, margin)
    fold_d = decide_routes(z_raw, eligible, t, margin)
    positive = owner >= 0
    rows = torch.arange(z_raw.shape[0])
    own = torch.full((z_raw.shape[0],), float("nan"))
    own_eligible = torch.zeros_like(positive)
    own[positive] = z_raw[rows[positive], owner[positive]]
    own_eligible[positive] = eligible[rows[positive], owner[positive]]
    best_neg = z_raw.masked_fill(~eligible, float("-inf")).max(dim=-1).values
    t_own = t if isinstance(t, float) else torch.where(positive, t[owner.clamp_min(0)], 0.0)

    def correct(d):
        return positive & d["active"] & (d["fact"] == owner)

    out = {}
    for split in sorted(set(splits)):
        m = torch.tensor([s == split for s in splits])
        pos, neg = m & positive, m & ~positive
        c_raw, c_fold = correct(raw_d), correct(fold_d)
        own_pos = own[pos & own_eligible]
        t_pos = t_own[pos & own_eligible] if isinstance(t_own, torch.Tensor) else t_own
        out[split] = {
            "raw": route_outcomes(z_raw[m], eligible[m], owner[m], 0.0, margin, facts),
            "folded": route_outcomes(z_raw[m], eligible[m], owner[m], t, margin, facts),
            "positives": int(pos.sum()),
            "positives_correct": {"raw": int((c_raw & pos).sum()),
                                  "folded": int((c_fold & pos).sum())},
            "positives_rescued_by_calibration": int((pos & c_fold & ~c_raw).sum()),
            "positives_lost_by_calibration": int((pos & c_raw & ~c_fold).sum()),
            "positive_own_logit_below_zero": int((own_pos < 0).sum()),
            "positive_own_logit_in_cutoff_to_zero": int(((own_pos >= t_pos) & (own_pos < 0)).sum()),
            "negatives": int(neg.sum()),
            "negatives_firing": {"raw": int((neg & raw_d["active"]).sum()),
                                 "folded": int((neg & fold_d["active"]).sum())},
            "negatives_added_by_calibration": int((neg & fold_d["active"] & ~raw_d["active"]).sum()),
            "positive_own_stage1_logit": _quantiles(own_pos),
            "negative_best_eligible_stage1_logit": _quantiles(best_neg[neg]),
        }
    return out, raw_d, fold_d


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--router-dir", required=True, help="folded (calibrated-bias) router dir")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", default="float32", choices=("float32", "bfloat16", "float16"))
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--skip-routes", action="store_true",
                   help="write the artifact only (no model, no route recomputation)")
    a = p.parse_args(argv)

    router_dir, output = Path(a.router_dir).resolve(), Path(a.output_dir).resolve()
    folded = torch.load(router_dir / "fact_association_embeddings.pt", map_location="cpu",
                        weights_only=False)
    manifest = json.loads((router_dir / "association_manifest.json").read_text())
    report = json.loads((router_dir / "linear_router_report.json").read_text())
    raw, stage1, cutoff = raw_artifact(folded)
    output.mkdir(parents=True, exist_ok=False)
    for name in ("association_examples.json",):
        if (router_dir / name).is_file():
            shutil.copy2(router_dir / name, output / name)

    ablation = {
        "schema_version": "bias_rule_ablation_v1",
        "folded_router_dir": str(router_dir),
        "rule_raw": RAW_RULE,
        "rule_folded": "calibrated cutoff t folded into the bias, p >= 0.5",
        "folded_cutoff_t": _json_safe(cutoff if isinstance(cutoff, float) else cutoff.tolist()),
        "note": ("t < 0 means calibration lowered the cutoff (fires more than p >= 0.5 "
                 "on the stage-1 logits); raw fires at z >= 0."),
    }
    raw_report = copy.deepcopy(report)
    raw_report["router_fit"] = _json_safe(raw["router_fit"])
    dataset_rows = None

    if not a.skip_routes:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from fit_linear_router import NEUTRAL_PROMPT, runtime_parity
        from linear_router import eligibility_matrix, load_linear_classifier_artifact
        from static_overlap_fact_association_embeddings import (
            AssociationCausalLM,
            extract_prompt_queries,
        )

        dataset_rows = json.loads((router_dir / "linear_router_dataset.json").read_text())
        facts = list(folded["facts"])
        index = {str(f["id"]): i for i, f in enumerate(facts)}
        prompts = [r["prompt"] for r in dataset_rows]
        splits = [r["split"] for r in dataset_rows]
        owner = torch.tensor([index[r["owner_fact_id"]] if r.get("owner_fact_id") else -1
                              for r in dataset_rows])
        model_path = Path(manifest["model_path"]).resolve()
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True,
                                                  local_files_only=a.local_files_only)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=getattr(torch, a.dtype), local_files_only=a.local_files_only,
            attn_implementation="eager",
        ).to(a.device).eval()
        model.requires_grad_(False)
        layer = int(folded["layer"])
        queries = extract_prompt_queries(model, tokenizer, prompts, layer, batch_size=a.batch_size)
        eligible = eligibility_matrix(tokenizer, prompts, folded["subject_patterns"])
        z_raw = score_queries(queries, folded["router_weight"], stage1,
                              folded["feature_mean"], folded.get("feature_components")).cpu()
        margin = float(folded.get("ambiguity_margin", 0.5))
        by_split, raw_d, fold_d = rule_diagnostics(z_raw, eligible, owner, splits, cutoff,
                                                   margin, facts)
        stored = [r.get("linear_routes_to") for r in dataset_rows]
        recomputed = [facts[int(f)]["id"] if bool(act) else None
                      for act, f in zip(fold_d["active"].tolist(), fold_d["fact"].tolist())]
        ablation["folded_routes_recomputed_vs_stored_mismatches"] = sum(
            s != r for s, r in zip(stored, recomputed))
        ablation["by_split"] = by_split

        neutral = tokenizer(NEUTRAL_PROMPT, return_tensors="pt").to(a.device)
        with torch.no_grad():
            base_logits = model(**neutral, use_cache=False).logits.detach().clone()
        wrapped, bank = load_linear_classifier_artifact(model, raw)
        with torch.no_grad():
            if not torch.equal(base_logits, wrapped(**neutral, use_cache=False).logits):
                raise RuntimeError("raw router left the exact base path on a neutral prompt")
        ablation["runtime_parity_raw"] = runtime_parity(
            model, bank, tokenizer, prompts, raw_d, a.batch_size, a.device)
        raw_report["route_outcomes_by_split"] = {s: v["raw"] for s, v in by_split.items()}
        raw_report["runtime_parity"] = ablation["runtime_parity_raw"]
        for row, act, f, best in zip(dataset_rows, raw_d["active"].tolist(),
                                     raw_d["fact"].tolist(),
                                     raw_d["best_eligible_logit"].tolist()):
            row["folded_routes_to"] = row.get("linear_routes_to")
            row["linear_routes_to"] = facts[int(f)]["id"] if act else None
            row["linear_best_eligible_logit"] = best if math.isfinite(best) else None
        summary = {s: {"raw_correct": v["positives_correct"]["raw"],
                       "folded_correct": v["positives_correct"]["folded"],
                       "positives": v["positives"],
                       "raw_neg_fire": v["negatives_firing"]["raw"],
                       "folded_neg_fire": v["negatives_firing"]["folded"],
                       "negatives": v["negatives"]} for s, v in by_split.items()}
        print(json.dumps({"phase": "routes", "cutoff_t": ablation["folded_cutoff_t"],
                          "by_split": summary,
                          "raw_runtime_mismatches": ablation["runtime_parity_raw"]["route_mismatches"],
                          "folded_recompute_mismatches":
                              ablation["folded_routes_recomputed_vs_stored_mismatches"]},
                         indent=2), flush=True)

    raw_report["bias_rule_ablation"] = _json_safe(ablation)
    (output / "linear_router_report.json").write_text(
        json.dumps(_json_safe(raw_report), indent=2, allow_nan=False) + "\n")
    (output / "bias_rule_ablation.json").write_text(
        json.dumps(_json_safe(ablation), indent=2, allow_nan=False) + "\n")
    if dataset_rows is not None:
        (output / "linear_router_dataset.json").write_text(
            json.dumps(_json_safe(dataset_rows), indent=2, allow_nan=False) + "\n")
    new_manifest = dict(manifest)
    new_manifest.update({
        "decision_rule": "explicit_threshold",
        "routing_policy": raw["routing_policy"],
        "bias_calibration": None,
        "router": _json_safe(raw["router_fit"]),
        "bias_rule": raw["bias_rule"],
        "bias_rule_source_router_dir": str(router_dir),
        "runtime_trigger": "complete subject-token eligibility plus learned linear BCE head "
                           "at p >= 0.5 with the stage-1 bias (no calibration)",
    })
    (output / "association_manifest.json").write_text(
        json.dumps(_json_safe(new_manifest), indent=2, allow_nan=False) + "\n")
    # Artifact last: its presence marks a complete stage.
    torch.save(raw, output / "fact_association_embeddings.pt")
    print(f"{output}: raw logistic router (cutoff removed: {ablation['folded_cutoff_t']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
