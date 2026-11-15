import pandas as pd

from scripts.video_data import large_ratio, opening_from_draw, stock_model


def test_the_stock_baseline_is_the_model_clear_voice_was_fine_tuned_from(tmp_path):
    pd.DataFrame([
        {"model": "holdA", "fold": "A", "wer": 0.180}, {"model": "holdB", "fold": "B", "wer": 0.164},
        {"model": "holdA_medium", "fold": "A", "wer": 0.169}, {"model": "holdB_medium", "fold": "B", "wer": 0.133},
    ]).to_csv(tmp_path / "model_choice.csv", index=False)
    assert stock_model(tmp_path) == "whisper-medium"
    assert large_ratio("whisper-medium") == 2 and large_ratio("whisper-small") == 6


def test_without_a_model_comparison_the_baseline_is_whisper_small(tmp_path):
    assert stock_model(tmp_path) == "whisper-small"


def test_opening_from_the_random_draw_takes_the_biggest_gain_and_prefers_an_exact_transcript():
    # Used only when per-clip predictions are not available: the rule must not depend on order in the file.
    draw = pd.DataFrame([
        {"row_id": 1, "severity": "mild", "ref": "rate", "whisper-medium": "great", "Clear Voice": "rate"},
        {"row_id": 2, "severity": "moderate", "ref": "meats", "whisper-medium": "me too", "Clear Voice": "meach"},
        {"row_id": 3, "severity": "severe", "ref": "dread", "whisper-medium": "dead", "Clear Voice": "dread"},
        {"row_id": 4, "severity": "severe", "ref": "down", "whisper-medium": "down", "Clear Voice": "dark"},
    ])
    assert int(opening_from_draw(draw, "whisper-medium")["row_id"]) == 3  # mild clips are not eligible
    assert int(opening_from_draw(draw, "whisper-medium", exclude=[3])["row_id"]) == 2


def test_opening_by_rule_is_the_75th_percentile_improvement_among_moderate_and_severe_sentences():
    from scripts.video_data import opening_by_rule

    refs = ["the cat sat down"] * 5
    ours = ["the cat sat down"] * 5
    stock = {10: "the", 11: "the cat", 12: "the cat sat", 13: "a cat sat down", 14: "the cat sat down"}
    ft = pd.DataFrame({"row_id": [10, 11, 12, 13, 14], "ref": refs, "hyp": ours, "part": "test_strict",
                       "is_dysarthric": True, "prompt_type": "sentence", "severity": "severe"})
    dur = pd.Series(3.0, index=[10, 11, 12, 13, 14])
    # gains 0.75, 0.5, 0.25, 0.25, 0 -> positive gains sorted 0.25, 0.25, 0.5, 0.75 -> 75th percentile index 2
    row = opening_by_rule(ft, pd.Series(stock), dur)
    assert int(row["row_id"]) == 11 and row["Clear Voice"] == "the cat sat down"
