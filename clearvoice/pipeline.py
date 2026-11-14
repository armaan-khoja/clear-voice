"""Glue the notebooks call: load TORGO once, turn split CSVs into rows, predict, and check general speech."""
from pathlib import Path

import pandas as pd

from .asr import transcribe, transcribe_rows
from .data import download_parquets, load_audio, load_metadata
from .metrics import score
from .splits import read_split

PRED_COLUMNS = ["row_id", "speaker", "severity", "is_dysarthric", "prompt_type", "part", "fold", "system",
                "ref", "hyp"]
LIBRISPEECH = ("openslr/librispeech_asr", "clean/test/0000.parquet", "71cacbfb7e2354c4226d01e70d77d5fca3d04ba1")
_META_KEEP = ["row_id", "speaker", "severity", "is_dysarthric", "prompt_type", "text", "gender"]


def read_preds(path) -> pd.DataFrame:
    """Read a predictions CSV keeping empty or 'null'-looking transcripts as text, not NaN."""
    return pd.read_csv(path, keep_default_na=False, na_values=[])


def load_torgo(work_dir) -> tuple:
    """Download the parquet files and load metadata, cached as a pickle in `work_dir`."""
    parquets = download_parquets()
    cache = Path(work_dir) / "meta.pkl"
    if cache.exists():
        meta = pd.read_pickle(cache)
    else:
        meta = load_metadata(parquets)
        cache.parent.mkdir(parents=True, exist_ok=True)
        meta.to_pickle(cache)
    return parquets, meta


def rows_for(meta: pd.DataFrame, splits_dir, name: str, part: str) -> pd.DataFrame:
    """The rows of one split part, in file order, joined with their metadata (transcript, severity...)."""
    ids = read_split(splits_dir, name, part)[["row_id"]]
    return ids.merge(meta[_META_KEEP], on="row_id", how="left", validate="one_to_one")


def predict(model, processor, parquets, meta, rows: pd.DataFrame, system: str, fold: str, part: str,
            batch_size: int = 16, audio: dict | None = None) -> pd.DataFrame:
    """Transcribe `rows` and return prediction rows (PRED_COLUMNS). Loads audio unless `audio` is given."""
    if "text" not in rows.columns:
        rows = rows[["row_id"]].merge(meta[_META_KEEP], on="row_id", how="left")
    if audio is None:
        audio = load_audio(parquets, meta, rows["row_id"])
    out = transcribe_rows(model, processor, rows, audio, batch_size=batch_size)
    return out.assign(part=part, fold=fold, system=system)[PRED_COLUMNS]


def predict_fold(model, processor, parquets, meta, splits_dir, fold: str, parts, system: str,
                 batch_size: int = 16, limit: int | None = None) -> pd.DataFrame:
    """Predict every requested part of the model that holds out `fold`. `limit` keeps the first N rows (smoke runs)."""
    frames = []
    for part in parts:
        rows = rows_for(meta, splits_dir, f"hold{fold}", part)
        if limit:
            rows = rows.head(limit)
        frames.append(predict(model, processor, parquets, meta, rows, system, fold, part, batch_size))
    return pd.concat(frames, ignore_index=True)


def librispeech_check(model, processor, n: int = 200, batch_size: int = 16, parquet_path=None) -> dict:
    """Everyday read English: WER on the first n LibriSpeech test-clean clips."""
    import pyarrow.parquet as pq

    from .audio import decode_wav

    if parquet_path is None:
        from huggingface_hub import hf_hub_download

        repo, file, rev = LIBRISPEECH
        parquet_path = hf_hub_download(repo, file, repo_type="dataset", revision=rev)
    table = pq.read_table(parquet_path, columns=["audio", "text"]).slice(0, n)
    audios = [decode_wav(b) for b in table.column("audio").combine_chunks().field("bytes").to_pylist()]
    refs = table.column("text").to_pylist()
    hyps = transcribe(model, processor, audios, batch_size=batch_size)
    return {**score(refs, hyps), "refs": refs, "hyps": hyps}


def kept_model(work_dir, name: str) -> str | None:
    """The trained model saved for `name` (e.g. 'holdA') in the work folder, or None if not trained yet."""
    best = Path(work_dir) / "models" / name / "best"
    return str(best) if (best / "config.json").exists() else None


def save_best(best_dir, work_dir, name: str) -> str:
    """Copy only the kept model (not checkpoints) into the work folder, e.g. on Google Drive."""
    import shutil

    dest = Path(work_dir) / "models" / name / "best"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(best_dir, dest)
    return str(dest)


def align_folds(zero_shot: dict, ft: pd.DataFrame) -> dict:
    """Keep baseline rows only for folds that have a fine-tuned model, so every system is scored on the same speakers."""
    folds = set(ft["fold"].unique())
    if folds != {"A", "B"}:
        print(f"Only fold(s) {sorted(folds)} have a fine-tuned model: comparing on those speakers only.")
    return {s: df[df["fold"].isin(folds)].reset_index(drop=True) for s, df in zero_shot.items()}
