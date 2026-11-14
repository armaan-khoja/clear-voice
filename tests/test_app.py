import numpy as np
import pandas as pd
import pytest

gr = pytest.importorskip("gradio")
import app.app as demo_app  # noqa: E402


def test_prepare_mic_converts_stereo_48k_int16_to_16k_mono_float():
    x = (np.random.default_rng(0).uniform(-1, 1, (48000, 2)) * 30000).astype(np.int16)
    out = demo_app.prepare_mic((48000, x))
    assert out.dtype == np.float32 and out.ndim == 1 and abs(len(out) - 16000) <= 2


def test_prepare_mic_handles_nothing_recorded():
    assert demo_app.prepare_mic(None) is None
    assert demo_app.prepare_mic((16000, np.array([], dtype=np.int16))) is None


def _clips(tmp_path):
    df = pd.DataFrame({"row_id": [5, 9], "file_idx": [0, 1], "row_in_file": [5, 2], "speaker": ["M01", "F04"],
                       "severity": ["severe", "mild"], "prompt_type": ["word", "sentence"],
                       "ref": ["yes", "the dog ran"], "Clear Voice": ["yes", "the dog ran"],
                       "whisper-small": ["Thank you.", "The dark ran."], "whisper-large-v3": ["Yes.", "the dog run"]})
    path = tmp_path / "demo_clips.csv"
    df.to_csv(path, index=False)
    return path


def test_build_ui_does_not_load_any_model(monkeypatch, tmp_path):
    import clearvoice.asr
    monkeypatch.setattr(clearvoice.asr, "load_model", lambda *a, **k: pytest.fail("model loaded at build time"))
    monkeypatch.setattr(demo_app, "CLIPS_CSV", _clips(tmp_path))
    assert isinstance(demo_app.build_ui(), gr.Blocks)


def test_show_clip_returns_audio_and_all_transcripts(monkeypatch, tmp_path):
    monkeypatch.setattr(demo_app, "CLIPS_CSV", _clips(tmp_path))
    monkeypatch.setattr(demo_app, "clip_audio", lambda row: (16000, np.zeros(1600, dtype=np.float32)))
    clips = demo_app.load_clips()
    label = demo_app.clip_label(clips.iloc[1])
    audio, truth, small, large, cv = demo_app.show_clip(label)
    assert audio[0] == 16000
    assert (truth, small, large, cv) == ("the dog ran", "The dark ran.", "the dog run", "the dog ran")


def test_live_transcription_says_when_the_fine_tuned_model_is_missing(monkeypatch):
    monkeypatch.delenv("CV_MODEL_A", raising=False)
    monkeypatch.setattr(demo_app, "_transcribe_with", lambda name, audio: f"[{name}]")
    stock, cv = demo_app.transcribe_live((16000, np.zeros(16000, dtype=np.int16)))
    assert stock == "[openai/whisper-small]" and "not loaded" in cv
