"""Draw a clip's spectrogram as a grey-scale PNG (dark = loud, as in a phonetics lab), for the video.

Run: .venv/bin/python scripts/video_spectrogram.py in.wav out.png
"""
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from scipy import signal  # noqa: E402


def spectrogram_png(wav: str, out: str, width: int = 1600, height: int = 360, top_hz: int = 6000) -> None:
    x, sr = sf.read(wav, dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = signal.lfilter([1, -0.97], [1], x)  # pre-emphasis, as speech tools do, so formants show
    f, t, z = signal.stft(x, fs=sr, nperseg=int(0.005 * sr) * 2 + 1, noverlap=int(0.005 * sr) * 2 - int(0.002 * sr), window="gaussian" if False else "hann")
    mag = 20 * np.log10(np.abs(z) + 1e-9)
    mag = mag[f <= top_hz]
    lo = mag.max() - 55  # 55 dB dynamic range
    img = np.clip((mag - lo) / (mag.max() - lo), 0, 1)
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img, origin="lower", aspect="auto", cmap="Greys", interpolation="bilinear")
    ax.axis("off")
    fig.savefig(out, dpi=100, transparent=True)
    plt.close(fig)


if __name__ == "__main__":
    spectrogram_png(sys.argv[1], sys.argv[2])
