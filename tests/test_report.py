import pandas as pd
import pytest

from clearvoice import plots
from clearvoice.report import headline, leakage, per_speaker, pick_demo_clips, worst_clips

SEV = {"F01": "severe", "M03": "mild", "F03": "moderate", "FC01": "typical"}


def fake_preds(system, wrong_every=3):
    rows = []
    for i in range(48):
        spk = list(SEV)[i % 4]
        part = ["test_seen", "test_newsent", "test_strict"][i % 3]
        sentence = i % 2 == 0
        ref = f"the cat sat number {i}" if sentence else f"word{i}"
        hyp = ref if i % wrong_every else ("the bat" if sentence else f"ward{i}")
        rows.append({"row_id": i, "speaker": spk, "severity": SEV[spk], "is_dysarthric": "C" not in spk,
                     "prompt_type": "sentence" if sentence else "word", "part": part, "fold": "A",
                     "system": system, "ref": ref, "hyp": hyp})
    return pd.DataFrame(rows)


def systems():
    return {"whisper-small": fake_preds("whisper-small", 2), "Clear Voice": fake_preds("Clear Voice", 4)}


def test_headline_has_one_row_per_system_with_intervals():
    h = headline(systems())
    assert list(h["system"]) == ["whisper-small", "Clear Voice"]
    for col in ("sentence_wer", "sentence_wer_lo", "sentence_wer_hi", "word_cer", "n_clips", "n_speakers"):
        assert col in h.columns
    assert (h["n_speakers"] == 3).all()  # dysarthric speakers only


def test_leakage_reports_three_conditions_for_dysarthric_speakers():
    t = leakage(systems()["Clear Voice"])
    assert list(t["condition"]) == ["New speaker, seen sentence", "Seen speaker, new sentence",
                                    "New speaker, new sentence"]


def test_per_speaker_covers_every_strict_speaker_and_system():
    t = per_speaker(systems())
    assert set(t["speaker"]) == set(SEV) and set(t["system"]) == {"whisper-small", "Clear Voice"}


def test_worst_clips_are_strict_dysarthric_and_sorted():
    w = worst_clips(systems()["Clear Voice"], n=5)
    assert (w["part"] == "test_strict").all() and w["is_dysarthric"].all()
    assert list(w["cer"]) == sorted(w["cer"], reverse=True)
    assert len(w) <= 5


def test_demo_clips_come_only_from_the_strict_split():
    d = pick_demo_clips(systems()["Clear Voice"], per_severity=2)
    assert (d["part"] == "test_strict").all() and d["is_dysarthric"].all()
    assert set(d["severity"]) <= {"severe", "mild", "moderate"}


@pytest.mark.parametrize("fn", ["bar_systems", "leakage", "per_speaker"])
def test_plots_write_a_non_empty_png(tmp_path, fn):
    data = {"bar_systems": headline(systems()), "leakage": leakage(systems()["Clear Voice"]),
            "per_speaker": per_speaker(systems())}[fn]
    out = tmp_path / f"{fn}.png"
    getattr(plots, fn)(data, out)
    assert out.stat().st_size > 1000


def test_personalisation_plot_writes_a_non_empty_png(tmp_path):
    df = pd.DataFrame({"speaker": ["M01"] * 4 + ["F03"] * 4, "severity": ["severe"] * 4 + ["moderate"] * 4,
                       "minutes": [0, 2, 5, 10] * 2, "cer": [0.6, 0.5, 0.4, 0.3, 0.3, 0.25, 0.2, 0.18]})
    out = tmp_path / "p.png"
    plots.personalisation(df, out)
    assert out.stat().st_size > 1000
