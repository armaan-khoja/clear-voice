import math

import pandas as pd
import pytest

from clearvoice.metrics import score, speaker_bootstrap, table


def test_identical_texts_score_zero():
    r = score(["hello world", "yes"], ["Hello, world!", "yes"])
    assert r["wer"] == 0 and r["cer"] == 0 and r["n"] == 2


def test_one_substitution_and_one_deletion_out_of_four_words_is_half():
    assert score(["a b c d"], ["a x c"])["wer"] == pytest.approx(0.5)


def test_empty_reference_is_dropped_and_counted():
    r = score(["", "hello there"], ["anything", "hello there"])
    assert r["n"] == 1 and r["n_dropped_empty_ref"] == 1 and r["wer"] == 0


def test_empty_output_scores_one_without_error():
    assert score(["hello world"], [""])["wer"] == pytest.approx(1.0)


def test_runaway_output_scores_above_one_without_error():
    assert score(["yes"], ["yes " + "thank you " * 10])["wer"] > 1


def test_no_scorable_rows_gives_nan():
    r = score([""], ["x"])
    assert r["n"] == 0 and math.isnan(r["wer"])


def _preds():
    return pd.DataFrame({
        "speaker": ["F01", "F01", "M03", "M03", "FC01"],
        "severity": ["severe", "severe", "mild", "mild", "typical"],
        "ref": ["the cat sat", "yes", "go left", "no", "alpha"],
        "hyp": ["the bat sat", "yes", "go left", "know", "alpha"],
    })


def test_table_counts_clips_and_speakers_per_group():
    t = table(_preds(), by=["severity"]).set_index("severity")
    assert t.loc["severe", "n_clips"] == 2 and t.loc["severe", "n_speakers"] == 1
    assert t.loc["mild", "wer"] == pytest.approx(1 / 3)
    assert set(t.columns) >= {"wer", "cer", "n_clips", "n_speakers"}


def test_bootstrap_interval_brackets_the_point_estimate():
    df = _preds()
    lo, hi = speaker_bootstrap(df, "wer", n_boot=200)
    point = score(list(df["ref"]), list(df["hyp"]))["wer"]
    assert lo <= point <= hi
