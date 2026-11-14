import numpy as np
import pytest

from clearvoice.data import check_mic_pairs, load_audio, load_metadata
from tests.conftest import write_parquet


def test_load_metadata_reads_every_row_in_file_order(tiny_parquets):
    meta = load_metadata(tiny_parquets)
    assert list(meta["row_id"]) == [0, 1, 2, 3, 4]
    assert list(meta["file_idx"]) == [0, 0, 0, 1, 1]
    assert list(meta["row_in_file"]) == [0, 1, 2, 0, 1]
    assert list(meta["speaker"]) == ["FC01", "FC01", "F01", "M03", "M03"]
    assert list(meta["mic"]) == ["headMic", "arrayMic", "headMic", "arrayMic", "headMic"]
    assert list(meta["is_dysarthric"]) == [False, False, True, True, True]
    assert meta.loc[2, "prompt_type"] == "sentence"
    assert meta.loc[0, "prompt_type"] == "word"
    assert meta.loc[0, "text_norm"] == meta.loc[1, "text_norm"]
    assert meta.loc[2, "n_words"] == 4
    assert meta.loc[3, "utt_key"] == "M03_1_0010"


def test_load_metadata_names_bad_files(tmp_path):
    bad = write_parquet(tmp_path / "bad.parquet",
                        [{"name": "recording7.wav", "text": "hi", "seconds": 1.0}])
    with pytest.raises(ValueError, match="recording7.wav"):
        load_metadata([bad])


def test_load_audio_returns_only_requested_rows_at_16k(tiny_parquets):
    meta = load_metadata(tiny_parquets)
    audio = load_audio(tiny_parquets, meta, [2, 4])
    assert set(audio) == {2, 4}
    for rid, x in audio.items():
        assert x.dtype == np.float32 and x.ndim == 1
        assert abs(len(x) - meta.loc[rid, "duration"] * 16000) <= 2


def test_check_mic_pairs_finds_disagreeing_copies(tiny_parquets):
    meta = load_metadata(tiny_parquets)
    bad = check_mic_pairs(meta)
    assert list(bad["utt_key"].unique()) == ["M03_1_0010"]
