import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from clearvoice.data import load_metadata
from clearvoice.pipeline import PRED_COLUMNS, librispeech_check, predict, rows_for
from clearvoice.splits import write_splits
from tests.conftest import wav_bytes


def test_rows_for_merges_split_rows_with_metadata_in_order(tmp_path, tiny_parquets):
    meta = load_metadata(tiny_parquets)
    meta["severity"] = ["typical", "typical", "severe", "mild", "mild"]
    write_splits({"test_strict": meta.iloc[[4, 2]]}, tmp_path, "holdA")
    rows = rows_for(meta, tmp_path, "holdA", "test_strict")
    assert list(rows["row_id"]) == [4, 2]
    assert list(rows["text"]) == ["Bravo", "The quick brown fox"]
    for col in ("severity", "is_dysarthric", "prompt_type", "speaker"):
        assert col in rows.columns


@pytest.mark.slow
def test_predict_returns_labelled_prediction_rows(tiny_parquets):
    from clearvoice.asr import load_model

    meta = load_metadata(tiny_parquets)
    model, processor = load_model("openai/whisper-tiny")
    out = predict(model, processor, tiny_parquets, meta, meta.iloc[[0, 2]], system="tiny", fold="A",
                  part="test_strict", batch_size=2)
    assert list(out.columns) == PRED_COLUMNS
    assert list(out["row_id"]) == [0, 2] and set(out["system"]) == {"tiny"} and set(out["part"]) == {"test_strict"}


@pytest.mark.slow
def test_librispeech_check_scores_a_local_parquet(tmp_path):
    from clearvoice.asr import load_model

    audio = pa.array([{"bytes": wav_bytes(1.0), "path": "a.flac"}, {"bytes": wav_bytes(1.5), "path": "b.flac"}],
                     type=pa.struct([("bytes", pa.binary()), ("path", pa.string())]))
    path = tmp_path / "ls.parquet"
    pq.write_table(pa.table({"audio": audio, "text": ["HELLO WORLD", "GOOD MORNING"]}), path)
    model, processor = load_model("openai/whisper-tiny")
    r = librispeech_check(model, processor, n=2, parquet_path=path)
    assert r["n"] == 2 and r["wer"] >= 0
