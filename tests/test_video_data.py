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
