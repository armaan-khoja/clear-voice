import pandas as pd


def test_validation_scores_from_drive_and_from_this_run_are_compared_together():
    from scripts.turbo_wrapup import merge_choice

    known = pd.DataFrame([
        {"model": "holdA_medium", "fold": "A", "system": "Clear Voice (medium)", "wer": 0.169, "cer": 0.1, "n": 650, "chosen": True},
        {"model": "holdB_medium", "fold": "B", "system": "Clear Voice (medium)", "wer": 0.133, "cer": 0.1, "n": 384, "chosen": True},
    ])
    new = pd.DataFrame([
        {"model": "holdA_large-v3-turbo", "fold": "A", "system": "x", "wer": 0.150, "cer": 0.1, "n": 650, "chosen": True},
        {"model": "holdB_large-v3-turbo", "fold": "B", "system": "x", "wer": 0.140, "cer": 0.1, "n": 384, "chosen": True},
    ])
    t = merge_choice(known, new)
    assert sorted(t[t.chosen].model) == ["holdA_large-v3-turbo", "holdB_medium"]
    assert len(t) == 4
