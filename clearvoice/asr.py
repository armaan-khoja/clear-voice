"""Loading Whisper and transcribing clips (English, transcribe task, no timestamps)."""
import numpy as np
import pandas as pd
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor


def pick_device(device: str | None = None) -> str:
    """CUDA when available, otherwise CPU (Apple's MPS is avoided: Whisper generation is unreliable there)."""
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


def load_model(name_or_path: str, device: str | None = None):
    """Return (model, processor) ready for inference. Uses float16 on CUDA, float32 on CPU."""
    device = pick_device(device)
    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = WhisperProcessor.from_pretrained(name_or_path)
    model = WhisperForConditionalGeneration.from_pretrained(name_or_path, dtype=dtype).to(device).eval()
    model.generation_config.forced_decoder_ids = None
    return model, processor


@torch.inference_mode()
def transcribe(model, processor, audios: list, batch_size: int = 16) -> list:
    """Transcribe 16 kHz mono float32 clips. Returns one string per clip, in order."""
    out = []
    for i in range(0, len(audios), batch_size):
        batch = [np.asarray(a, dtype=np.float32) for a in audios[i : i + batch_size]]
        feats = processor.feature_extractor(batch, sampling_rate=16000, return_tensors="pt").input_features
        feats = feats.to(model.device, dtype=model.dtype)
        ids = model.generate(input_features=feats, language="en", task="transcribe", max_new_tokens=128)
        out += [s.strip() for s in processor.batch_decode(ids, skip_special_tokens=True)]
    return out


def transcribe_rows(model, processor, rows: pd.DataFrame, audio: dict, batch_size: int = 16) -> pd.DataFrame:
    """Add `hyp` (model output) and `ref` (the row's `text`) to each row, keeping row order."""
    hyps = transcribe(model, processor, [audio[int(r)] for r in rows["row_id"]], batch_size=batch_size)
    return rows.assign(ref=rows["text"].values, hyp=hyps)
