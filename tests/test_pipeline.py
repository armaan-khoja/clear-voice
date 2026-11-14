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


def test_read_preds_keeps_empty_and_null_looking_outputs_as_text(tmp_path):
    from clearvoice.pipeline import read_preds

    df = pd.DataFrame({"row_id": [1, 2, 3], "speaker": ["M01"] * 3, "is_dysarthric": [True] * 3,
                       "ref": ["yes", "no", "null"], "hyp": ["", "null", "NA"]})
    df.to_csv(tmp_path / "p.csv", index=False)
    back = read_preds(tmp_path / "p.csv")
    assert list(back["hyp"]) == ["", "null", "NA"] and list(back["ref"]) == ["yes", "no", "null"]
    assert back["is_dysarthric"].dtype == bool


def _fake_model_dir(path):
    path.mkdir(parents=True)
    (path / "config.json").write_text("{}")
    (path / "model.safetensors").write_text("weights")
    return path


def test_save_best_copies_only_the_kept_model_to_the_work_folder(tmp_path):
    from clearvoice.pipeline import kept_model, save_best

    local = _fake_model_dir(tmp_path / "local" / "holdA" / "best")
    (tmp_path / "local" / "holdA" / "checkpoints").mkdir()
    out = save_best(local, tmp_path / "drive", "holdA")
    assert out == str(tmp_path / "drive" / "models" / "holdA" / "best")
    assert (tmp_path / "drive" / "models" / "holdA" / "best" / "model.safetensors").exists()
    assert not (tmp_path / "drive" / "models" / "holdA" / "checkpoints").exists()
    assert kept_model(tmp_path / "drive", "holdA") == out


def test_kept_model_is_none_before_training(tmp_path):
    from clearvoice.pipeline import kept_model

    assert kept_model(tmp_path, "holdB") is None


def test_align_folds_drops_baseline_rows_for_folds_without_a_fine_tuned_model():
    from clearvoice.pipeline import align_folds

    zs = {"whisper-small": pd.DataFrame({"fold": ["A", "A", "B"], "row_id": [1, 2, 3]})}
    ft = pd.DataFrame({"fold": ["A"], "row_id": [1]})
    out = align_folds(zs, ft)
    assert list(out["whisper-small"]["fold"]) == ["A", "A"]
