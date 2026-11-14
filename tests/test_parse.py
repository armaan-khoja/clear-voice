import pytest

from clearvoice.data import parse_path


def test_parse_path_reads_all_fields():
    p = parse_path("FC02_3_headMic_0006.wav")
    assert p == {
        "speaker": "FC02",
        "session": "3",
        "mic": "headMic",
        "utt": "0006",
        "utt_key": "FC02_3_0006",
    }


def test_parse_path_ignores_directories():
    assert parse_path("some/dir/M04_2_arrayMic_0101.wav")["speaker"] == "M04"


def test_parse_path_rejects_unknown_names():
    with pytest.raises(ValueError, match="bad.wav"):
        parse_path("bad.wav")
