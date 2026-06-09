def test_embhead_imports_and_alg_dict():
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
    assert "ZeroUnlearn_EmbHead_All" in evaluate.ALG_DICT
    assert "ZeroUnlearn_EmbHead_TouchedRows" in evaluate.ALG_DICT
