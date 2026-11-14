import numpy as np
import pandas as pd
import pytest

from clearvoice.asr import load_model, transcribe
from clearvoice.train import TrainConfig, finetune

pytestmark = pytest.mark.slow


def _data():
    rng = np.random.default_rng(0)
    audio = {i: (0.1 * rng.standard_normal(16000 + 4000 * i)).astype(np.float32) for i in range(4)}
    rows = pd.DataFrame({"row_id": [0, 1, 2, 3], "text": ["yes", "no", "go left", "the quick brown fox"]})
    return rows, audio


def test_finetune_writes_a_loadable_model(tmp_path):
    rows, audio = _data()
    cfg = TrainConfig(base_model="openai/whisper-tiny", max_steps=2, eval_steps=1, save_steps=1,
                      batch_size=2, eval_batch_size=2, num_workers=0)
    out = finetune(cfg, rows.iloc[:3], rows.iloc[3:], audio, str(tmp_path / "run"))
    model, processor = load_model(out)
    hyps = transcribe(model, processor, [audio[0]], batch_size=1)
    assert len(hyps) == 1 and isinstance(hyps[0], str)


def test_finetune_without_validation_trains_for_epochs(tmp_path):
    rows, audio = _data()
    cfg = TrainConfig(base_model="openai/whisper-tiny", max_steps=None, num_train_epochs=1, eval=False,
                      batch_size=2, num_workers=0)
    out = finetune(cfg, rows, None, audio, str(tmp_path / "personal"))
    model, processor = load_model(out)
    assert isinstance(transcribe(model, processor, [audio[1]])[0], str)
