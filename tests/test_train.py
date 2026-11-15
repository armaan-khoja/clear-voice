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


def test_finetuned_model_keeps_output_layer_tied_to_embeddings(tmp_path):
    """The best-checkpoint reload once warned 'missing keys: proj_out.weight'; the saved model must stay tied."""
    import torch

    rows, audio = _data()
    cfg = TrainConfig(base_model="openai/whisper-tiny", max_steps=2, eval_steps=1, save_steps=1,
                      batch_size=2, eval_batch_size=2, num_workers=0)
    out = finetune(cfg, rows.iloc[:3], rows.iloc[3:], audio, str(tmp_path / "run"))
    model, _ = load_model(out)
    assert torch.equal(model.proj_out.weight, model.model.decoder.embed_tokens.weight)


def test_finetune_removes_intermediate_checkpoints_to_save_drive_space(tmp_path):
    rows, audio = _data()
    cfg = TrainConfig(base_model="openai/whisper-tiny", max_steps=2, eval_steps=1, save_steps=1,
                      batch_size=2, eval_batch_size=2, num_workers=0)
    finetune(cfg, rows.iloc[:3], rows.iloc[3:], audio, str(tmp_path / "run"))
    assert not (tmp_path / "run" / "checkpoints").exists()
    assert (tmp_path / "run" / "best").exists()


def test_finetune_with_gradient_checkpointing_and_accumulation(tmp_path):
    """The whisper-medium settings (small batches, accumulation, checkpointing) must train and save a model."""
    rows, audio = _data()
    cfg = TrainConfig(base_model="openai/whisper-tiny", max_steps=2, eval_steps=1, save_steps=1, batch_size=1,
                      grad_accum=2, gradient_checkpointing=True, eval_batch_size=2, num_workers=0)
    out = finetune(cfg, rows.iloc[:3], rows.iloc[3:], audio, str(tmp_path / "run"))
    model, processor = load_model(out)
    assert isinstance(transcribe(model, processor, [audio[0]], batch_size=1)[0], str)
