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
