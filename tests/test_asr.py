import numpy as np
import pandas as pd
import pytest

from clearvoice.asr import load_model, transcribe, transcribe_rows

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def tiny():
    return load_model("openai/whisper-tiny")


def _clips():
    t = np.arange(32000) / 16000
    return [np.zeros(16000, dtype=np.float32), (0.2 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)]


def test_transcribe_returns_one_string_per_clip(tiny):
    model, processor = tiny
    out = transcribe(model, processor, _clips(), batch_size=2)
    assert len(out) == 2 and all(isinstance(s, str) for s in out)


def test_transcribe_rows_keeps_row_order_and_adds_ref_and_hyp(tiny):
    model, processor = tiny
    rows = pd.DataFrame({"row_id": [7, 3], "text": ["silence", "tone"]})
    audio = dict(zip([7, 3], _clips()))
    out = transcribe_rows(model, processor, rows, audio, batch_size=1)
    assert list(out["row_id"]) == [7, 3]
    assert list(out["ref"]) == ["silence", "tone"]
    assert out["hyp"].map(type).eq(str).all()
