import numpy as np

from clearvoice.audio import decode_wav, to_16k_mono
from tests.conftest import wav_bytes


def test_to_16k_mono_converts_stereo_int16_48k():
    stereo = (np.random.default_rng(0).uniform(-1, 1, (48000, 2)) * 32767).astype(np.int16)
    out = to_16k_mono(stereo, 48000)
    assert out.dtype == np.float32
    assert out.ndim == 1
    assert abs(len(out) - 16000) <= 2
    assert np.abs(out).max() <= 1.0


def test_to_16k_mono_keeps_16k_float():
    x = np.zeros(1600, dtype=np.float32)
    assert len(to_16k_mono(x, 16000)) == 1600


def test_decode_wav_resamples_8k_to_16k():
    out = decode_wav(wav_bytes(1.0, sr=8000))
    assert out.dtype == np.float32
    assert abs(len(out) - 16000) <= 2
