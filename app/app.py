"""Clear Voice demo: a Gradio app for a Hugging Face Space.

Tabs:
- Unseen speakers: clips from dysarthric speakers the model never heard, saying sentences it never heard,
  with the true text and each system's output (precomputed by the notebooks).
- Try it: record or upload speech; stock whisper-small and Clear Voice transcribe it live.
- Results: the write-up's tables and figures.
"""
import os
import sys
from functools import lru_cache
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
for root in (HERE, HERE.parent):  # the package sits next to app.py on the Space, one level up in the repo
    if (root / "clearvoice").is_dir():
        sys.path.insert(0, str(root))
        break
from clearvoice.audio import to_16k_mono  # noqa: E402

DISCLAIMER = "Research prototype, not a medical device. Uses the TORGO database (non-commercial research use)."
STOCK = "openai/whisper-small"
CLIPS_CSV = HERE / "demo_clips.csv"
FIGURES = HERE / "figures"
SUMMARY = HERE / "summary.md"
SYSTEMS = ["whisper-small", "whisper-large-v3", "Clear Voice"]


def prepare_mic(x):
    """Gradio gives (sample_rate, int16 array, maybe stereo). Return 16 kHz mono float32, or None if empty."""
    if x is None:
        return None
    sr, arr = x
    arr = np.asarray(arr)
    if arr.size == 0:
        return None
    return to_16k_mono(arr, sr)


@lru_cache(maxsize=2)
def _model(name: str):
    from clearvoice.asr import load_model

    return load_model(name)


def _transcribe_with(name: str, audio: np.ndarray) -> str:
    from clearvoice.asr import transcribe

    model, processor = _model(name)
    return transcribe(model, processor, [audio], batch_size=1)[0]


def transcribe_live(x):
    """Return (stock whisper-small, Clear Voice) transcripts for a recording."""
    audio = prepare_mic(x)
    if audio is None:
        return "", ""
    stock = _transcribe_with(STOCK, audio)
    cv_id = os.environ.get("CV_MODEL_A")
    cv = _transcribe_with(cv_id, audio) if cv_id else "(Clear Voice model not loaded: set the CV_MODEL_A variable)"
    return stock, cv


def load_clips() -> pd.DataFrame:
    if not Path(CLIPS_CSV).exists():
        return pd.DataFrame()
    return pd.read_csv(CLIPS_CSV, keep_default_na=False, na_values=[])  # an empty transcript stays empty


def clip_label(row) -> str:
    return f"{row['speaker']} ({row['severity']}, {row['prompt_type']}) #{row['row_id']}"


@lru_cache(maxsize=4)
def _parquet(file_idx: int):
    from huggingface_hub import hf_hub_download

    from clearvoice.data import DATASET_ID, PARQUET_FILES, REVISION

    return hf_hub_download(DATASET_ID, PARQUET_FILES[file_idx], repo_type="dataset", revision=REVISION)


def clip_audio(row):
    """Fetch one clip from the TORGO dataset on Hugging Face (audio is never stored in the app)."""
    from clearvoice.data import PARQUET_FILES, load_audio

    meta = pd.DataFrame([{"row_id": int(row["row_id"]), "file_idx": int(row["file_idx"]),
                          "row_in_file": int(row["row_in_file"])}])
    paths = [_parquet(i) if i == int(row["file_idx"]) else None for i in range(len(PARQUET_FILES))]
    return 16000, load_audio(paths, meta, [int(row["row_id"])])[int(row["row_id"])]


def show_clip(label: str):
    clips = load_clips()
    row = clips[[clip_label(r) == label for _, r in clips.iterrows()]].iloc[0]
    return (clip_audio(row), row["ref"], *[str(row.get(s, "")) for s in SYSTEMS])


def warm_up() -> None:
    """Load both models and the demo clips' dataset files before anyone clicks, so the first click is fast."""
    _model(STOCK)
    cv_id = os.environ.get("CV_MODEL_A")
    if cv_id:
        _model(cv_id)
    clips = load_clips()
    if not clips.empty:
        for i in sorted(set(int(x) for x in clips["file_idx"])):
            _parquet(i)


def build_ui() -> gr.Blocks:
    clips = load_clips()
    with gr.Blocks(title="Clear Voice") as demo:
        gr.Markdown(
            "# Clear Voice\n"
            "Speech recognition fine-tuned so people with dysarthria (speech affected by cerebral palsy, ALS, "
            "stroke or Parkinson's) are understood, tested on speakers and sentences it never heard.\n\n"
            f"> {DISCLAIMER}"
        )
        with gr.Tab("Unseen speakers"):
            if clips.empty:
                gr.Markdown("Demo clips have not been generated yet (run notebook 5).")
            else:
                gr.Markdown("Clips picked at random per severity from the strict test set: the model never heard "
                            "this speaker or this sentence. Transcripts were produced by the notebooks.")
                choice = gr.Dropdown([clip_label(r) for _, r in clips.iterrows()], label="Choose a clip")
                audio = gr.Audio(label="Recording", interactive=False)
                truth = gr.Textbox(label="What they said")
                outs = [gr.Textbox(label=s) for s in SYSTEMS]
                choice.change(show_clip, choice, [audio, truth, *outs])
        with gr.Tab("Try it"):
            gr.Markdown("Speak or upload a short clip. Both models run live on this server's CPU, so it takes a few seconds.")
            mic = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Your recording")
            button = gr.Button("Transcribe")
            stock = gr.Textbox(label="Stock whisper-small")
            cv = gr.Textbox(label="Clear Voice")
            button.click(transcribe_live, mic, [stock, cv])
        with gr.Tab("Results"):
            if Path(SUMMARY).exists():
                gr.Markdown(Path(SUMMARY).read_text())
            for png in sorted(Path(FIGURES).glob("*.png")):
                gr.Image(str(png), show_label=False)
    return demo


if __name__ == "__main__":
    import threading

    threading.Thread(target=warm_up, daemon=True).start()
    build_ui().launch()
