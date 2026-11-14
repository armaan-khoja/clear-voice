"""Word and character error rates, per group, with speaker-level bootstrap intervals."""
import math

import jiwer
import numpy as np
import pandas as pd

from .text import normalize


def score(refs: list, hyps: list) -> dict:
    """Corpus-level WER and CER after normalising both sides. Empty references are dropped and counted."""
    return _score_normed([normalize(r) for r in refs], [normalize(h) for h in hyps])


def _score_normed(refs: list, hyps: list) -> dict:
    pairs = list(zip(refs, hyps))
    kept = [(r, h) for r, h in pairs if r]
    out = {"n": len(kept), "n_dropped_empty_ref": len(pairs) - len(kept)}
    if not kept:
        return {**out, "wer": math.nan, "cer": math.nan}
    r, h = [p[0] for p in kept], [p[1] for p in kept]
    words = jiwer.process_words(r, h)
    chars = jiwer.process_characters(r, h)
    return {**out, "wer": words.wer, "cer": chars.cer}


def table(df: pd.DataFrame, by: list) -> pd.DataFrame:
    """WER, CER, clip and speaker counts for each group of `by` columns. `df` needs ref, hyp and speaker."""
    rows = []
    for key, g in df.groupby(by, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        s = score(list(g["ref"]), list(g["hyp"]))
        rows.append({**dict(zip(by, key)), "wer": s["wer"], "cer": s["cer"],
                     "n_clips": s["n"], "n_speakers": g["speaker"].nunique()})
    return pd.DataFrame(rows)


def speaker_bootstrap(df: pd.DataFrame, metric: str = "wer", n_boot: int = 1000, seed: int = 42) -> tuple:
    """95% interval for a corpus metric, resampling whole speakers with replacement."""
    rng = np.random.default_rng(seed)
    df = df.assign(ref=[normalize(r) for r in df["ref"]], hyp=[normalize(h) for h in df["hyp"]])
    by_spk = {s: g for s, g in df.groupby("speaker")}
    speakers = list(by_spk)
    stats = []
    for _ in range(n_boot):
        pick = pd.concat([by_spk[s] for s in rng.choice(speakers, size=len(speakers), replace=True)])
        v = _score_normed(list(pick["ref"]), list(pick["hyp"]))[metric]
        if not math.isnan(v):
            stats.append(v)
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))
