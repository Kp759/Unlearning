import json
import sys
from pathlib import Path

import pytest
import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import build_mcf_surrogate_paraphrases as builder  # noqa: E402
import gagd_active_case_repair as repair  # noqa: E402
import mcf_sure_h_then_genaware_lmhead_lora as M  # noqa: E402


def _inst(record_pos, kind, index):
    return repair.MCFPromptInstance(
        record_index=100 + record_pos,
        sampled_position=record_pos,
        prompt_type=kind,
        prompt_index=index,
        prompt=f"prompt-{record_pos}-{kind}-{index}",
        target_new="reference",
        target_true="sensitive",
    )


def test_builder_deterministic_surrogates_are_unique_and_subject_preserving():
    xs = builder.deterministic_surrogates("Ada Lovelace", "Ada Lovelace worked in")
    assert len(xs) >= 8
    assert len({builder._normalize_cmp(x) for x in xs}) == len(xs)
    assert all("ada lovelace" in builder._normalize_cmp(x) for x in xs)
    assert all(builder._normalize_cmp(x) != builder._normalize_cmp("Ada Lovelace worked in") for x in xs)


def test_builder_answer_filter_uses_token_boundaries():
    assert builder._contains_answer("The answer would be English", ["English"])
    assert not builder._contains_answer("An Englishman was mentioned", ["English"])
    assert not builder._contains_answer("business relation", ["US"])


def test_worst_per_record_selects_one_worst_failure_per_fact():
    instances = [
        _inst(0, "rewrite", 0),
        _inst(1, "rewrite", 0),
        _inst(0, "surrogate", 0),
        _inst(0, "surrogate", 1),
        _inst(1, "surrogate", 0),
    ]
    margins = torch.tensor([0.3, -0.2, -1.1, -0.5, -0.9])
    active = M.select_active_instances(instances, margins, 0.25, "worst_per_record")
    assert active == [2, 4]


def test_all_failures_policy_keeps_every_failing_prompt():
    instances = [_inst(0, "rewrite", 0), _inst(0, "surrogate", 0), _inst(1, "surrogate", 0)]
    margins = torch.tensor([0.3, -0.2, 0.1])
    assert M.select_active_instances(instances, margins, 0.25, "all_failures") == [1, 2]


def test_scale_key_prioritizes_direct_eff_before_surrogate():
    direct_bad_surrogate_good = {
        "scale": 0.5,
        "direct": {"failures": 1},
        "surrogate": {"failures": 0, "minimum_margin": 1.0},
        "combined": {"minimum_margin": -0.1, "mean_margin": 1.0},
    }
    direct_good_surrogate_bad = {
        "scale": 1.0,
        "direct": {"failures": 0},
        "surrogate": {"failures": 2, "minimum_margin": -2.0},
        "combined": {"minimum_margin": -2.0, "mean_margin": 0.0},
    }
    assert M._scale_key(direct_good_surrogate_bad) < M._scale_key(direct_bad_surrogate_good)


def test_build_instances_never_uses_official_probe_fields():
    records = [{
        "case_id": 7,
        "requested_rewrite": {
            "subject": "Ada",
            "prompt": "{} worked in",
            "target_true": {"str": "sensitive"},
            "target_new": {"str": "reference"},
        },
        # These keys are deliberately present in the dummy record. build_instances
        # must ignore them; production load_locked rejects them before this point.
        "paraphrase_prompts": ["OFFICIAL PARA"],
        "neighborhood_prompts": ["OFFICIAL NEIGHBOR"],
    }]
    direct, surrogate = M.build_instances(records, [["For Ada, the field of work was"]])
    assert [x.prompt for x in direct] == ["Ada worked in"]
    assert [x.prompt for x in surrogate] == ["For Ada, the field of work was"]
    assert "OFFICIAL PARA" not in [x.prompt for x in surrogate]
    assert "OFFICIAL NEIGHBOR" not in [x.prompt for x in surrogate]


def test_surrogate_artifact_validation_rejects_declared_official_access(tmp_path):
    records = [{
        "case_id": 7,
        "requested_rewrite": {
            "subject": "Ada",
            "prompt": "{} worked in",
            "target_true": {"str": "sensitive"},
            "target_new": {"str": "reference"},
        },
    }]
    payload = {
        "schema_version": 1,
        "protocol": M.SURROGATE_PROTOCOL,
        "seed": 1,
        "forget_num": 1,
        "data_access": {
            "official_paraphrase_seen": 1,
            "official_neighborhood_seen": 0,
            "benchmark_retain_seen": 0,
            "official_PPL_seen": False,
        },
        "records": [{
            "case_id": 7,
            "sampled_position": 0,
            "subject": "Ada",
            "direct_prompt": "Ada worked in",
            "surrogate_prompts": ["For Ada, the field of work was"],
        }],
    }
    p = tmp_path / "surrogates.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="official paraphrase"):
        M.load_surrogate_artifact(p, records, seed=1, forget_num=1)


def test_parser_defaults_to_rank16_worst_case_training():
    args = M.parse_args([
        "--stage1-model-path", "stage1",
        "--training-visible-path", "visible.json",
        "--split-manifest", "manifest.json",
        "--surrogate-prompts-path", "surrogate.json",
        "--output-dir", "out",
        "--utility-wikipedia-dir", "wiki",
    ])
    assert args.lora_rank == 16
    assert args.lora_alpha == 16.0
    assert args.active_policy == "worst_per_record"
    assert args.solver_margin == 0.25
    assert args.acceptance_margin == 0.05
