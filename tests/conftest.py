"""Tiny synthetic TORGO-like parquet files, same schema as abnerh/TORGO-database."""
import io

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf


def wav_bytes(seconds: float, sr: int = 16000, freq: float = 220.0) -> bytes:
    t = np.arange(int(seconds * sr)) / sr
    x = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, x, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def write_parquet(path, rows):
    """rows: dicts with name, text, seconds, and optional sr, status, gender, severity."""
    audio = pa.array(
        [{"bytes": wav_bytes(r["seconds"], r.get("sr", 16000)), "path": r["name"]} for r in rows],
        type=pa.struct([("bytes", pa.binary()), ("path", pa.string())]),
    )
    table = pa.table(
        {
            "audio": audio,
            "transcription": [r["text"] for r in rows],
            "speech_status": [r.get("status", "healthy") for r in rows],
            "gender": [r.get("gender", "female") for r in rows],
            "duration": [float(r["seconds"]) for r in rows],
            "severity": [r.get("severity", "typical") for r in rows],
        }
    )
    pq.write_table(table, path)
    return path


@pytest.fixture
def tiny_parquets(tmp_path):
    f0 = write_parquet(
        tmp_path / "f0.parquet",
        [
            {"name": "FC01_1_headMic_0001.wav", "text": "Yes", "seconds": 1.0},
            {"name": "FC01_1_arrayMic_0001.wav", "text": "yes", "seconds": 1.0},
            {"name": "F01_2_headMic_0005.wav", "text": "The quick brown fox", "seconds": 2.0,
             "status": "dysarthria", "severity": "severe", "sr": 8000},
        ],
    )
    f1 = write_parquet(
        tmp_path / "f1.parquet",
        [
            {"name": "M03_1_arrayMic_0010.wav", "text": "Alpha", "seconds": 1.5,
             "status": "dysarthria", "gender": "male", "severity": "mild"},
            {"name": "M03_1_headMic_0010.wav", "text": "Bravo", "seconds": 1.5,
             "status": "dysarthria", "gender": "male", "severity": "mild"},
        ],
    )
    return [f0, f1]
