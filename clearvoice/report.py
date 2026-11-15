"""Tables for the write-up: headline, leakage, per speaker, worst clips and demo clips.

Prediction frames have columns: row_id, speaker, severity, is_dysarthric, prompt_type,
part (test_seen / test_newsent / test_strict), fold, system, ref, hyp.
"""
import numpy as np
import pandas as pd

from .metrics import score, speaker_bootstrap
from .text import normalize

CONDITIONS = {
    "test_seen": "New speaker, seen sentence",
    "test_newsent": "Seen speaker, new sentence",
    "test_strict": "New speaker, new sentence",
}


def _dys(df: pd.DataFrame, part: str = "test_strict") -> pd.DataFrame:
    return df[(df["part"] == part) & df["is_dysarthric"].astype(bool)]


def _metric_with_ci(df: pd.DataFrame, metric: str) -> tuple:
    if df.empty:
        return np.nan, np.nan, np.nan
    point = score(list(df["ref"]), list(df["hyp"]))[metric]
    lo, hi = speaker_bootstrap(df, metric, n_boot=1000) if df["speaker"].nunique() > 1 else (np.nan, np.nan)
    return point, lo, hi


def _median_clip(df: pd.DataFrame, metric: str) -> float:
    """Median per-clip error: robust to the rare clip where the model invents a long string of words."""
    if df.empty:
        return np.nan
    vals = [score([r], [h])[metric] for r, h in zip(df["ref"], df["hyp"])]
    vals = [v for v in vals if not np.isnan(v)]
    return float(np.median(vals)) if vals else np.nan


def headline(preds: dict) -> pd.DataFrame:
    """Strict split, dysarthric speakers: sentence WER and single-word CER with 95% speaker-bootstrap intervals."""
    rows = []
    for system, df in preds.items():
        d = _dys(df)
        s, s_lo, s_hi = _metric_with_ci(d[d["prompt_type"] == "sentence"], "wer")
        w, w_lo, w_hi = _metric_with_ci(d[d["prompt_type"] == "word"], "cer")
        rows.append({"system": system, "sentence_wer": s, "sentence_wer_lo": s_lo, "sentence_wer_hi": s_hi,
                     "sentence_wer_median": _median_clip(d[d["prompt_type"] == "sentence"], "wer"),
                     "word_cer": w, "word_cer_lo": w_lo, "word_cer_hi": w_hi,
                     "word_cer_median": _median_clip(d[d["prompt_type"] == "word"], "cer"),
                     "n_clips": len(d), "n_speakers": d["speaker"].nunique()})
    return pd.DataFrame(rows)


def leakage(preds_ft: pd.DataFrame) -> pd.DataFrame:
    """How the same fine-tuned model scores as the test gets harder (dysarthric speakers only)."""
    rows = []
    for part, label in CONDITIONS.items():
        d = _dys(preds_ft, part)
        sent, word = d[d["prompt_type"] == "sentence"], d[d["prompt_type"] == "word"]
        rows.append({"condition": label, "part": part,
                     "sentence_wer": score(list(sent["ref"]), list(sent["hyp"]))["wer"],
                     "word_cer": score(list(word["ref"]), list(word["hyp"]))["cer"],
                     "n_clips": len(d), "n_speakers": d["speaker"].nunique()})
    return pd.DataFrame(rows)


def per_speaker(preds: dict) -> pd.DataFrame:
    """Strict split, every speaker (controls included): sentence WER and word CER per system."""
    rows = []
    for system, df in preds.items():
        strict = df[df["part"] == "test_strict"]
        for spk, g in strict.groupby("speaker"):
            sent, word = g[g["prompt_type"] == "sentence"], g[g["prompt_type"] == "word"]
            rows.append({"system": system, "speaker": spk, "severity": g["severity"].iloc[0],
                         "sentence_wer": score(list(sent["ref"]), list(sent["hyp"]))["wer"],
                         "word_cer": score(list(word["ref"]), list(word["hyp"]))["cer"], "n_clips": len(g)})
    return pd.DataFrame(rows)


def _clip_cer(ref: str, hyp: str) -> float:
    return score([ref], [hyp])["cer"]


def worst_clips(preds_ft: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """The n strict-split dysarthric clips with the highest character error rate."""
    d = _dys(preds_ft).copy()
    d["cer"] = [_clip_cer(r, h) for r, h in zip(d["ref"], d["hyp"])]
    return d.sort_values("cer", ascending=False).head(n).reset_index(drop=True)


def pick_demo_clips(preds_ft: pd.DataFrame, per_severity: int = 5, seed: int = 42) -> pd.DataFrame:
    """A random (not cherry-picked) strict-split sample per severity, with up to 2 sentences each."""
    d = _dys(preds_ft)
    picks = []
    for sev, g in d.groupby("severity"):
        sent, word = g[g["prompt_type"] == "sentence"], g[g["prompt_type"] == "word"]
        n_sent = min(2, len(sent), per_severity)
        picks.append(sent.sample(n_sent, random_state=seed))
        picks.append(word.sample(min(per_severity - n_sent, len(word)), random_state=seed))
    return pd.concat(picks).reset_index(drop=True)


def compare_validation(val: dict) -> pd.DataFrame:
    """Score each model on its fold's validation sentences and mark the lower word error rate per fold as chosen.

    `val` maps a model name (e.g. 'holdA', 'holdA_medium') to its validation predictions. Choosing between
    models uses validation only, never the test splits.
    """
    rows = []
    for model, df in val.items():
        s = score(list(df["ref"]), list(df["hyp"]))
        rows.append({"model": model, "fold": df["fold"].iloc[0], "system": df["system"].iloc[0],
                     "wer": s["wer"], "cer": s["cer"], "n": s["n"]})
    t = pd.DataFrame(rows)
    t["chosen"] = t["wer"] == t.groupby("fold")["wer"].transform("min")
    return t.sort_values(["fold", "wer"]).reset_index(drop=True)


def choose_architecture(table: pd.DataFrame) -> str:
    """The base model ('small', 'medium', ...) with the lowest mean validation WER over both folds.

    `table` is compare_validation's output; model names look like 'holdA' (small) or 'holdA_medium'. Only
    architectures with a model for every fold are eligible, so Clear Voice is one architecture, not a mix.
    """
    t = table.copy()
    t["arch"] = t["model"].map(lambda m: m.split("_", 1)[1] if "_" in m else "small")
    folds = t["fold"].nunique()
    complete = t.groupby("arch")["fold"].nunique() == folds
    means = t[t["arch"].isin(complete[complete].index)].groupby("arch")["wer"].mean()
    return means.idxmin()
