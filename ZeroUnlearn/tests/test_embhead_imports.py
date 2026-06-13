def test_embhead_get_algorithm_does_not_import_mend():
    import sys
    from pathlib import Path

    project_root = str(Path(__file__).resolve().parents[1])
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    sys.modules.pop("baselines.mend", None)

    import ZeroUnlearn_EmbHead
    from ZeroUnlearn_EmbHead import (
        ZeroUnlearnEmbHeadHyperParams,
        apply_emb_head_all_to_model,
        apply_emb_head_touched_rows_to_model,
    )
    import experiments.evaluate as evaluate

    assert ZeroUnlearn_EmbHead is not None
    assert ZeroUnlearnEmbHeadHyperParams is not None
    assert apply_emb_head_all_to_model is not None
    assert apply_emb_head_touched_rows_to_model is not None
    assert "baselines.mend" not in sys.modules

    params_class, apply_algo = evaluate.get_algorithm("ZeroUnlearn_EmbHead_TouchedRows")

    assert params_class is ZeroUnlearnEmbHeadHyperParams
    assert apply_algo is apply_emb_head_touched_rows_to_model
    assert "baselines.mend" not in sys.modules
