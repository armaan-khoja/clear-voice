"""Build the demo video's data (video/src/data.json), clip audio and spectrograms from the real results.

Run: .venv/bin/python scripts/video_data.py --results <local copy of MyDrive/clearvoice/results> [--team "Names"] [--demo-url URL]

- Demo clips come from notebook 5's random draw (demo_clips.csv): one sentence per severity, so nothing is cherry-picked.
- The opening clip is chosen by a fixed rule, stated in the code: among unseen severe or moderate sentences of 2 to 5
  seconds, the one at the 75th percentile of improvement over stock Whisper (a good example, not the best one).
- Audio comes from the TORGO parquet files in the local Hugging Face cache and is written only under video/public/,
  which git ignores (TORGO is non-commercial research data).
"""
import argparse
import json
import sys
from pathlib import Path

import jiwer
import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from clearvoice.data import download_parquets, load_audio  # noqa: E402
from clearvoice.pipeline import read_preds  # noqa: E402
from clearvoice.text import normalize  # noqa: E402
from video_spectrogram import spectrogram_png  # noqa: E402

SYSTEMS = ["whisper-small", "whisper-large-v3", "Clear Voice"]
PUBLIC = ROOT / "video" / "public"


def tokens(ref: str, hyp: str) -> tuple[list, float]:
    """Each word the system wrote, tagged ok / sub / ins against the reference, plus the clip's word error rate."""
    r, h = normalize(ref), normalize(hyp)
    if not r:
        return [], float("nan")
    if not h:
        return [], 1.0
    out = jiwer.process_words(r, h)
    words = h.split()
    tags = ["ok"] * len(words)
    for chunk in out.alignments[0]:
        if chunk.type in ("substitute", "insert"):
            for j in range(chunk.hyp_start_idx, chunk.hyp_end_idx):
                tags[j] = "sub" if chunk.type == "substitute" else "ins"
    return [{"w": w, "tag": t} for w, t in zip(words, tags)], float(out.wer)


def bar(name: str, value: float, lo=None, hi=None) -> dict:
    b = {"name": name, "value": round(float(value), 4)}
    if lo is not None and hi is not None and not (np.isnan(lo) or np.isnan(hi)):
        b.update(lo=round(float(lo), 4), hi=round(float(hi), 4))
    return b


def clip_entry(row: pd.Series, hyps: dict, meta: pd.DataFrame, audio: dict) -> dict:
    rid = int(row["row_id"])
    wav = PUBLIC / "clips" / f"{rid}.wav"
    png = PUBLIC / "spec" / f"{rid}.png"
    wav.parent.mkdir(parents=True, exist_ok=True)
    png.parent.mkdir(parents=True, exist_ok=True)
    sf.write(wav, audio[rid], 16000)
    spectrogram_png(str(wav), str(png))
    m = meta.set_index("row_id").loc[rid]
    ref = row["ref"]  # the predictions carry the true transcript; the saved metadata does not
    systems = {}
    for s in SYSTEMS:
        if s in hyps and isinstance(hyps[s], str):
            toks, wer = tokens(ref, hyps[s])
            systems[s] = {"text": normalize(hyps[s]), "tokens": toks, "wer": round(wer, 4)}
    return {"rowId": rid, "file": f"clips/{rid}.wav", "spec": f"spec/{rid}.png", "speaker": m["speaker"],
            "severity": m["severity"], "duration": round(len(audio[rid]) / 16000, 2), "ref": normalize(ref),
            "systems": systems}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--meta", default=str(ROOT / ".superpowers" / "meta.pkl"))
    ap.add_argument("--team", default="Team names here")
    ap.add_argument("--demo-url", default="")
    ap.add_argument("--preview", action="store_true", help="mark the video as placeholder data (for test runs)")
    a = ap.parse_args()
    R = Path(a.results)
    meta = pd.read_pickle(a.meta)
    dur = meta.set_index("row_id")["duration"]

    h = pd.read_csv(R / "headline.csv").set_index("system")
    lk = pd.read_csv(R / "leakage.csv")
    zs_small = read_preds(R / "preds_zeroshot_whisper-small.csv")
    ft = pd.concat([read_preds(p) for p in sorted(R.glob("preds_ft_hold?.csv"))], ignore_index=True)
    zs = {s: read_preds(R / f"preds_zeroshot_{s}.csv") for s in ["whisper-small", "whisper-large-v3"]}
    zs = {s: df[df["fold"].isin(ft["fold"].unique())] for s, df in zs.items()}

    typical = zs_small[(zs_small["part"] == "test_strict") & ~zs_small["is_dysarthric"].astype(bool) & (zs_small["prompt_type"] == "sentence")]
    typical = typical[typical["fold"].isin(ft["fold"].unique())]
    stock_typical = jiwer.wer([normalize(x) for x in typical["ref"]], [normalize(x) or "-" for x in typical["hyp"]]) if len(typical) else None

    personal = []
    if (R / "personalise.csv").exists():
        ps = pd.read_csv(R / "personalise.csv")
        for spk, g in ps.groupby("speaker", sort=False):
            g = g.sort_values("minutes")
            personal.append({"speaker": spk, "severity": g["severity"].iloc[0],
                             "points": [{"minutes": int(m), "wer": round(float(w), 4)} for m, w in zip(g["minutes"], g["wer"])]})
        personal.sort(key=lambda s: {"severe": 0, "moderate": 1, "mild": 2}.get(s["severity"], 3))

    general = pd.read_csv(R / "general_speech.csv") if (R / "general_speech.csv").exists() else pd.DataFrame()
    libri = [bar(s, general[general["system"] == s]["librispeech_wer"].mean()) for s in ["whisper-small", "Clear Voice"]
             if len(general) and (general["system"] == s).any()]

    # Demo: notebook 5's random draw, first sentence of at most 4.5 s per severity.
    demo_rows = pd.read_csv(R / "demo_clips.csv", keep_default_na=False, na_values=[])
    demo_rows = demo_rows[(demo_rows["prompt_type"] == "sentence")]
    demo_rows = demo_rows[demo_rows["row_id"].map(dur) <= 4.5]
    picks = [g.iloc[0] for sev in ["mild", "moderate", "severe"] for _, g in [(sev, demo_rows[demo_rows["severity"] == sev])] if len(g)]

    # Opening: fixed rule, see the module docstring.
    strict = ft[(ft["part"] == "test_strict") & ft["is_dysarthric"].astype(bool) & (ft["prompt_type"] == "sentence")
                & ft["severity"].isin(["severe", "moderate"])].copy()
    strict = strict[strict["row_id"].map(dur).between(2, 5) & ~strict["row_id"].isin([int(p["row_id"]) for p in picks])]
    stock_hyp = zs["whisper-small"].set_index("row_id")["hyp"]
    strict["gain"] = [tokens(r, stock_hyp.get(rid, ""))[1] - tokens(r, hyp)[1] for rid, r, hyp in zip(strict["row_id"], strict["ref"], strict["hyp"])]
    good = strict[strict["gain"] > 0].sort_values("gain")
    opening_row = good.iloc[int(0.75 * (len(good) - 1))] if len(good) else strict.iloc[0]

    rows = [opening_row] + picks
    ids = [int(r["row_id"]) for r in rows]
    audio = load_audio(download_parquets(), meta, ids)

    def hyps_for(rid: int, cv: str) -> dict:
        out = {"Clear Voice": cv}
        for s, df in zs.items():
            m = df[df["row_id"] == rid]
            if len(m):
                out[s] = m["hyp"].iloc[0]
        return out

    opening = clip_entry(opening_row, hyps_for(int(opening_row["row_id"]), opening_row["hyp"]), meta, audio)
    demo = [clip_entry(p, hyps_for(int(p["row_id"]), p["Clear Voice"]), meta, audio) for p in picks]

    data = {
        "placeholder": a.preview,
        "team": a.team,
        "repoUrl": "github.com/armaan-khoja/clear-voice",
        "demoUrl": a.demo_url,
        "nRecordings": int(len(meta)),
        "nSpeakers": int(meta["speaker"].nunique()),
        "nDysarthric": int(meta[meta["is_dysarthric"].astype(bool)]["speaker"].nunique()),
        "trainMinutes": 30,
        "headline": [bar(s, h.loc[s, "sentence_wer"], h.loc[s, "sentence_wer_lo"], h.loc[s, "sentence_wer_hi"]) for s in SYSTEMS if s in h.index],
        "headlineWords": [bar(s, h.loc[s, "word_cer"]) for s in SYSTEMS if s in h.index],
        "leakage": [bar(c, w) for c, w in zip(lk["condition"], lk["sentence_wer"])],
        "stockTypical": None if stock_typical is None else round(float(stock_typical), 4),
        "librispeech": libri,
        "personalisation": personal,
        "opening": opening,
        "demo": demo,
        "hasVoiceover": (PUBLIC / "voiceover.wav").exists(),
    }
    (ROOT / "video" / "src" / "data.json").write_text(json.dumps(data, indent=2) + "\n")
    print(f"wrote data.json: opening {opening['speaker']} ({opening['severity']}), demo {[d['speaker'] for d in demo]}")


if __name__ == "__main__":
    main()
