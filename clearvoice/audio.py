"""Audio helpers: everything the models see is float32, mono, 16 kHz."""
import io
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SR = 16000


def to_16k_mono(x: np.ndarray, sr: int) -> np.ndarray:
    """Convert any clip (int or float, mono or multi-channel, any rate) to 16 kHz mono float32."""
    x = np.asarray(x)
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float32) / np.iinfo(x.dtype).max
    x = x.astype(np.float32)
    if x.ndim == 2:
        # Channels are the short axis: (samples, channels) from Gradio, (channels, samples) elsewhere.
        x = x.mean(axis=1 if x.shape[1] <= x.shape[0] else 0)
    if sr != TARGET_SR:
        g = gcd(int(sr), TARGET_SR)
        x = resample_poly(x, TARGET_SR // g, int(sr) // g).astype(np.float32)
    return np.clip(x, -1.0, 1.0)


def decode_wav(b: bytes) -> np.ndarray:
    """Decode WAV (or FLAC) bytes to 16 kHz mono float32."""
    x, sr = sf.read(io.BytesIO(b), dtype="float32")
    return to_16k_mono(x, sr)
