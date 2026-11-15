"""Synthesise the demo video's music bed and sound effects (no downloaded assets).

Run: .venv/bin/python scripts/video_audio.py [total_seconds]
Writes video/public/sfx/{music,whoosh,riser,chime}.wav.
"""
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

SR = 44100
OUT = Path(__file__).resolve().parents[1] / "video" / "public" / "sfx"
rng = np.random.default_rng(7)


def hz(midi: float) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12)


def t_axis(sec: float) -> np.ndarray:
    return np.arange(int(sec * SR)) / SR


def soft_saw(f: float, t: np.ndarray, harmonics: int = 10, bright: float = 1.6) -> np.ndarray:
    """Band-limited saw with a gentle roll-off: warm, never buzzy."""
    out = np.zeros_like(t)
    for k in range(1, harmonics + 1):
        if f * k > SR / 2.2:
            break
        out += np.sin(2 * np.pi * f * k * t + rng.uniform(0, 2 * np.pi)) / k ** bright
    return out


def adsr(n: int, a: float, d: float, s: float, r: float) -> np.ndarray:
    a_n, d_n, r_n = int(a * SR), int(d * SR), int(r * SR)
    sus = max(0, n - a_n - d_n - r_n)
    env = np.concatenate([np.linspace(0, 1, a_n, endpoint=False), np.linspace(1, s, d_n, endpoint=False),
                          np.full(sus, s), np.linspace(s, 0, r_n)])
    return np.pad(env, (0, max(0, n - len(env))))[:n]


def reverb(x: np.ndarray, seconds: float = 2.4, mix: float = 0.3) -> np.ndarray:
    n = int(seconds * SR)
    ir = rng.standard_normal(n) * np.exp(-np.linspace(0, 7, n))
    ir = signal.sosfilt(signal.butter(2, 5000, "low", fs=SR, output="sos"), ir)
    ir /= np.sqrt(np.sum(ir ** 2))
    wet = signal.fftconvolve(x, ir)[: len(x)]
    return (1 - mix) * x + mix * wet * 0.6


def place(buf: np.ndarray, x: np.ndarray, at: float) -> None:
    i = int(at * SR)
    j = min(len(buf), i + len(x))
    if i < len(buf):
        buf[i:j] += x[: j - i]


def music(total: float, hook_end: float = 13.0) -> np.ndarray:
    bpm = 84
    beat = 60 / bpm
    bar = 4 * beat
    # A minor -> F -> C -> G, two bars each: hopeful, not saccharine.
    chords = [[57, 60, 64, 69], [53, 57, 60, 65], [55, 60, 64, 67], [55, 59, 62, 67]]
    roots = [45, 41, 48, 43]
    length = total + 3
    pad, bass, arp, drums = (np.zeros(int(length * SR)) for _ in range(4))
    t0, i = 0.0, 0
    while t0 < length:
        dur = 2 * bar
        ch, root = chords[i % 4], roots[i % 4]
        t = t_axis(dur + 1.5)
        env = adsr(len(t), 1.4, 0.5, 0.8, 1.6)
        voice = sum(soft_saw(hz(m), t) + soft_saw(hz(m) * 1.004, t) for m in ch) / (2 * len(ch))
        place(pad, voice * env, t0)
        place(bass, np.sin(2 * np.pi * hz(root - 12) * t) * adsr(len(t), 0.3, 0.4, 0.7, 1.2), t0)
        if t0 >= hook_end - 0.01:  # the pulse arrives with the title
            pattern = [0, 1, 2, 3, 2, 1, 2, 3]
            for k in range(int(dur / (beat / 2))):
                m = ch[pattern[k % 8]] + 12
                tt = t_axis(0.5)
                note = soft_saw(hz(m), tt, harmonics=6, bright=1.2) * np.exp(-tt * 9)
                place(arp, note, t0 + k * beat / 2)
            for k in range(int(dur / beat)):
                tt = t_axis(0.35)
                kick = np.sin(2 * np.pi * (48 + 70 * np.exp(-tt * 30)) * tt) * np.exp(-tt * 11)
                place(drums, kick * 0.9, t0 + k * beat)
                hat = signal.sosfilt(signal.butter(2, 7000, "high", fs=SR, output="sos"), rng.standard_normal(int(0.06 * SR)))
                place(drums, hat * np.exp(-t_axis(0.06) * 60) * 0.12, t0 + k * beat + beat / 2)
        t0 += dur
        i += 1
    pad = signal.sosfilt(signal.butter(2, 2200, "low", fs=SR, output="sos"), pad)
    arp = signal.sosfilt(signal.butter(2, 3500, "low", fs=SR, output="sos"), arp)
    # Ping-pong delay on the arp gives width.
    d = int(beat * 0.75 * SR)
    left = arp + 0.35 * np.pad(arp, (d, 0))[: len(arp)]
    right = arp + 0.35 * np.pad(arp, (2 * d, 0))[: len(arp)]
    end = int(total * SR)
    outro = np.ones(len(pad))
    outro[max(0, end - 8 * SR):] = np.linspace(1, 0, len(outro[max(0, end - 8 * SR):]))  # beat leaves for the ending
    mono = reverb(0.55 * pad, 3.0, 0.35) + 0.35 * bass
    L = mono + reverb(0.22 * left, 2.0, 0.3) * outro + 0.35 * drums * outro
    R = mono + reverb(0.22 * right, 2.0, 0.3) * outro + 0.35 * drums * outro
    st = np.stack([L, R], axis=1)[:end]
    fade = np.ones(len(st))
    fade[: 2 * SR] = np.linspace(0, 1, 2 * SR)
    fade[-3 * SR:] = np.linspace(1, 0, 3 * SR)
    st *= fade[:, None]
    return st / np.max(np.abs(st)) * 0.89


def whoosh(sec: float = 0.9) -> np.ndarray:
    n = int(sec * SR)
    noise = rng.standard_normal(n)
    out = np.zeros(n)
    block = 512
    for s in range(0, n, block):
        p = s / n
        centre = 300 * (12 ** np.sin(np.pi * p))  # sweeps up then down
        sos = signal.butter(2, [centre * 0.7, min(centre * 1.6, SR / 2.1)], "band", fs=SR, output="sos")
        out[s:s + block] = signal.sosfilt(sos, noise[s:s + block])
    env = np.sin(np.pi * np.linspace(0, 1, n)) ** 2
    x = reverb(out * env, 1.2, 0.25)
    pan = np.linspace(0.2, 0.8, n)
    st = np.stack([x * (1 - pan), x * pan], axis=1)
    return st / np.max(np.abs(st)) * 0.8


def riser(sec: float = 1.8) -> np.ndarray:
    t = t_axis(sec)
    f = 120 * (8 ** (t / sec))
    tone = np.sin(2 * np.pi * np.cumsum(f) / SR)
    noise = signal.sosfilt(signal.butter(2, 3000, "high", fs=SR, output="sos"), rng.standard_normal(len(t)))
    env = (t / sec) ** 2.2
    x = reverb((0.6 * tone + 0.25 * noise) * env, 1.5, 0.3)
    return np.stack([x, x], axis=1) / np.max(np.abs(x)) * 0.8


def chime(sec: float = 1.6) -> np.ndarray:
    t = t_axis(sec)
    x = sum(a * np.sin(2 * np.pi * hz(m) * r * t) * np.exp(-t * dcy)
            for m, a in [(88, 1.0), (95, 0.5)] for r, dcy in [(1, 3.2), (2.76, 6), (5.4, 9)])
    x *= np.minimum(1, t * 400)
    x = reverb(x, 2.0, 0.35)
    return np.stack([x, x], axis=1) / np.max(np.abs(x)) * 0.7


def pen(sec: float = 0.16) -> np.ndarray:
    """A felt-tip stroke: band-limited scratch with a fast attack and grainy texture."""
    n = int(sec * SR)
    noise = signal.sosfilt(signal.butter(2, [1800, 6500], "band", fs=SR, output="sos"), rng.standard_normal(n))
    grain = 0.6 + 0.4 * np.abs(signal.sosfilt(signal.butter(1, 60, "low", fs=SR, output="sos"), rng.standard_normal(n))) * 6
    env = np.minimum(1, np.linspace(0, 1, n) * 25) * np.exp(-np.linspace(0, 4, n))
    x = noise * env * np.clip(grain, 0, 1.5)
    return np.stack([x, x], axis=1) / np.max(np.abs(x)) * 0.6


def paper(sec: float = 0.55) -> np.ndarray:
    """A sheet sliding over another: soft low-passed rustle with a few crackles."""
    n = int(sec * SR)
    base = signal.sosfilt(signal.butter(2, [300, 3200], "band", fs=SR, output="sos"), rng.standard_normal(n))
    crack = np.zeros(n)
    for _ in range(14):
        i = rng.integers(0, n - 400)
        crack[i:i + 400] += rng.standard_normal(400) * np.exp(-np.linspace(0, 8, 400)) * 0.6
    crack = signal.sosfilt(signal.butter(2, 2500, "high", fs=SR, output="sos"), crack)
    env = np.sin(np.pi * np.linspace(0, 1, n)) ** 1.5
    x = (base + crack) * env
    pan = np.linspace(0.7, 0.35, n)
    st = np.stack([x * (1 - pan), x * pan], axis=1)
    return st / np.max(np.abs(st)) * 0.55


def main(total: float) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, x in {"music": music(total), "whoosh": whoosh(), "riser": riser(), "chime": chime(),
                    "pen": pen(), "paper": paper()}.items():
        sf.write(OUT / f"{name}.wav", x.astype(np.float32), SR, subtype="PCM_16")
        print(f"wrote {name}.wav ({len(x) / SR:.1f} s)")


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else 120.0)
