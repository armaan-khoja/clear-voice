"""Loading TORGO from Hugging Face: metadata, speaker IDs and audio."""
import os
import re

DATASET_ID = "abnerh/TORGO-database"
REVISION = "6e9b33c1af30eb3878574a1c1f5e60a60bd6d9c5"
PARQUET_FILES = [f"data/train-0000{i}-of-00004.parquet" for i in range(4)]

DYSARTHRIC = ["F01", "F03", "F04", "M01", "M02", "M03", "M04", "M05"]
CONTROLS = ["FC01", "FC02", "FC03", "MC01", "MC02", "MC03", "MC04"]
SEVERITY = {
    "F01": "severe", "M01": "severe", "M02": "severe", "M04": "severe",
    "M05": "moderate-severe", "F03": "moderate", "F04": "mild", "M03": "mild",
}

# speakerNumber_sessionNumber_micType_utteranceNumber.wav, e.g. FC02_3_headMic_0006.wav
_NAME = re.compile(
    r"^(?P<speaker>[FM]C?\d{2})_(?P<session>[^_]+)_(?P<mic>[A-Za-z]+Mic)_(?P<utt>\d+)\.wav$"
)


def parse_path(path: str) -> dict:
    """Read speaker, session, microphone and utterance number from a TORGO filename."""
    name = os.path.basename(path)
    m = _NAME.match(name)
    if m is None:
        raise ValueError(f"Unrecognised TORGO filename: {name}")
    d = m.groupdict()
    d["utt_key"] = f"{d['speaker']}_{d['session']}_{d['utt']}"
    return d


def download_parquets(cache_dir: str | None = None) -> list:
    """Download the 4 pinned TORGO parquet files (about 1.6 GB) and return their local paths."""
    from pathlib import Path

    from huggingface_hub import hf_hub_download

    return [
        Path(hf_hub_download(DATASET_ID, f, repo_type="dataset", revision=REVISION, cache_dir=cache_dir))
        for f in PARQUET_FILES
    ]


_META_COLS = ["audio", "transcription", "speech_status", "gender", "duration", "severity"]


def load_metadata(parquet_paths) -> "pd.DataFrame":
    """One row per clip, in file order, without the audio. row_id is the position across all files."""
    import pandas as pd
    import pyarrow.parquet as pq

    from .text import normalize, prompt_type

    rows, bad = [], []
    for file_idx, p in enumerate(parquet_paths):
        row_in_file = 0
        for batch in pq.ParquetFile(p).iter_batches(batch_size=256, columns=_META_COLS):
            paths = batch.column("audio").field("path").to_pylist()
            cols = {c: batch.column(c).to_pylist() for c in _META_COLS[1:]}
            for i, path in enumerate(paths):
                try:
                    parsed = parse_path(path or "")
                except ValueError:
                    bad.append(os.path.basename(path or "<missing path>"))
                    row_in_file += 1
                    continue
                rows.append({
                    "file_idx": file_idx, "row_in_file": row_in_file, "path": os.path.basename(path),
                    **parsed, "text": cols["transcription"][i], "speech_status": cols["speech_status"][i],
                    "gender": cols["gender"][i], "duration": cols["duration"][i], "severity": cols["severity"][i],
                })
                row_in_file += 1
    if bad:
        raise ValueError(f"{len(bad)} unrecognised TORGO filenames, e.g. {bad[:5]}")
    meta = pd.DataFrame(rows)
    meta.insert(0, "row_id", range(len(meta)))
    meta["text_norm"] = [normalize(t or "") for t in meta["text"]]
    meta["n_words"] = meta["text_norm"].str.split().str.len().fillna(0).astype(int)
    meta["prompt_type"] = [prompt_type(t) for t in meta["text_norm"]]
    meta["is_dysarthric"] = ~meta["speaker"].str.contains("C")
    return meta


def load_audio(parquet_paths, meta, row_ids) -> dict:
    """Decode only the requested clips. Returns {row_id: 16 kHz mono float32 array}."""
    import pyarrow.parquet as pq

    from .audio import decode_wav

    wanted = meta.set_index("row_id").loc[sorted(set(int(r) for r in row_ids)), ["file_idx", "row_in_file"]]
    out = {}
    for file_idx, group in wanted.groupby("file_idx"):
        need = dict(zip(group["row_in_file"], group.index))
        pos = 0
        for batch in pq.ParquetFile(parquet_paths[file_idx]).iter_batches(batch_size=256, columns=["audio"]):
            n = batch.num_rows
            hits = [r for r in need if pos <= r < pos + n]
            if hits:
                blobs = batch.column("audio").field("bytes")
                for r in hits:
                    out[need[r]] = decode_wav(blobs[r - pos].as_py())
            pos += n
            if pos > max(need):
                break
    return out


def check_mic_pairs(meta) -> "pd.DataFrame":
    """Utterances whose microphone copies disagree on the normalised transcript."""
    n = meta.groupby("utt_key")["text_norm"].nunique()
    return meta[meta["utt_key"].isin(n[n > 1].index)].sort_values(["utt_key", "mic"])
