"""Build the demo video's data (video/src/data.json), clip audio and spectrograms from the real results.

Run: .venv/bin/python scripts/video_data.py --results <folder with the results tables> [--team "Names"] [--demo-url URL]

- The final tables (scripts/final_report.py) are used when present: final_headline.csv, final_leakage.csv,
  final_general_speech.csv and final_demo_clips.csv. Older names (headline.csv, ...) are the fallback.
- The stock baseline is the model Clear Voice was fine-tuned from, chosen on validation (model_choice.csv).
- Demo clips come from the random draw (final_demo_clips.csv): the first sentence of at most 8 seconds per severity,
  so nothing is cherry-picked.
- The opening clip is chosen by a fixed rule. With the chosen model's per-clip predictions: among unseen severe or
  moderate sentences of 2 to 5 seconds, the one at the 75th percentile of improvement over the stock model (a good
  example, not the best one). Without them: among the random draw's moderate or severe clips, the one with the largest
  improvement over the stock model, preferring a clip Clear Voice transcribed exactly.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from clearvoice.pipeline import read_preds  # noqa: E402
from clearvoice.text import normalize  # noqa: E402

PUBLIC = ROOT / "video" / "public"
PARAMS_M = {"whisper-small": 244, "whisper-medium": 769, "whisper-large-v3-turbo": 809, "whisper-large-v3": 1550}
TRAIN_MINUTES = {"whisper-small": 20, "whisper-medium": 72}  # one fold on one L4 (overnight_log.txt)


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


def first_existing(results: Path, *names: str) -> Path | None:
    return next((results / n for n in names if (results / n).exists()), None)


def stock_model(results: Path) -> str:
    """The stock model Clear Voice was fine-tuned from: the architecture chosen on validation."""
    from clearvoice import report

    p = results / "model_choice.csv"
    return f"whisper-{report.choose_architecture(pd.read_csv(p))}" if p.exists() else "whisper-small"


def large_ratio(stock: str) -> int:
    """How many times bigger whisper-large-v3 is than the stock model, rounded."""
    return round(PARAMS_M["whisper-large-v3"] / PARAMS_M[stock])


def opening_from_draw(draw: pd.DataFrame, stock: str, exclude=()) -> pd.Series:
    """Among the random draw's moderate or severe clips, the largest improvement over the stock model,
    preferring a clip Clear Voice transcribed exactly; ties keep the draw's order."""
    d = draw[draw["severity"].isin(["moderate", "moderate-severe", "severe"]) & ~draw["row_id"].isin(list(exclude))].copy()
    ours = [tokens(r, h)[1] for r, h in zip(d["ref"], d["Clear Voice"])]
    d["gain"] = [tokens(r, s)[1] - o for r, s, o in zip(d["ref"], d[stock], ours)]
    d["exact"] = [o == 0 for o in ours]
    return d.sort_values(["gain", "exact"], ascending=False, kind="stable").iloc[0]


def clip_entry(row: pd.Series, hyps: dict, meta: pd.DataFrame, audio: dict, systems: list) -> dict:
    import soundfile as sf
    from video_spectrogram import spectrogram_png

    rid = int(row["row_id"])
    wav = PUBLIC / "clips" / f"{rid}.wav"
    png = PUBLIC / "spec" / f"{rid}.png"
    wav.parent.mkdir(parents=True, exist_ok=True)
    png.parent.mkdir(parents=True, exist_ok=True)
    sf.write(wav, audio[rid], 16000)
    spectrogram_png(str(wav), str(png))
    m = meta.set_index("row_id").loc[rid]
    ref = row["ref"]  # the predictions carry the true transcript
    out = {}
    for s in systems:
        if s in hyps and isinstance(hyps[s], str):
            toks, wer = tokens(ref, hyps[s])
            out[s] = {"text": normalize(hyps[s]), "tokens": toks, "wer": round(wer, 4)}
    return {"rowId": rid, "file": f"clips/{rid}.wav", "spec": f"spec/{rid}.png", "speaker": m["speaker"],
            "severity": m["severity"], "duration": round(len(audio[rid]) / 16000, 2), "ref": normalize(ref),
            "systems": out}


def main() -> None:
    from clearvoice.data import download_parquets, load_audio

    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--meta", default=str(ROOT / ".superpowers" / "meta.pkl"))
    ap.add_argument("--team", default="Team names here")
    ap.add_argument("--demo-url", default="")
    ap.add_argument("--stock-typical", type=float, default=None,
                    help="stock model's WER on typical speakers' sentences, if its predictions are not in --results")
    ap.add_argument("--preview", action="store_true", help="mark the video as placeholder data (for test runs)")
    a = ap.parse_args()
    R = Path(a.results)
    meta = pd.read_pickle(a.meta)
    dur = meta.set_index("row_id")["duration"]

    stock = stock_model(R)
    systems = [stock, "whisper-large-v3", "Clear Voice"]
    suffix = "" if stock == "whisper-small" else "_" + stock[len("whisper-"):]
    h = pd.read_csv(first_existing(R, "final_headline.csv", "headline.csv")).set_index("system")
    lk = pd.read_csv(first_existing(R, "final_leakage.csv", "leakage.csv"))

    ft_paths = sorted(R.glob(f"preds_ft_hold?{suffix}.csv"))
    ft = pd.concat([read_preds(p) for p in ft_paths], ignore_index=True) if len(ft_paths) == 2 else None
    zs = {s: read_preds(R / f"preds_zeroshot_{s}.csv") for s in systems if (R / f"preds_zeroshot_{s}.csv").exists()}

    stock_typical = a.stock_typical
    if stock in zs:
        z = zs[stock]
        typical = z[(z["part"] == "test_strict") & ~z["is_dysarthric"].astype(bool) & (z["prompt_type"] == "sentence")]
        if len(typical):
            stock_typical = jiwer.wer([normalize(x) for x in typical["ref"]], [normalize(x) or "-" for x in typical["hyp"]])

    personal = []
    pp = first_existing(R, "personalise_tuned_v2.csv", "personalise_tuned.csv", "personalise.csv")
    if pp is not None:
        ps = pd.read_csv(pp)
        for spk, g in ps.groupby("speaker", sort=False):
            g = g.sort_values("minutes")
            personal.append({"speaker": spk, "severity": g["severity"].iloc[0],
                             "points": [{"minutes": int(m), "wer": round(float(w), 4)} for m, w in zip(g["minutes"], g["wer"])]})
        personal.sort(key=lambda s: {"severe": 0, "moderate": 1, "mild": 2}.get(s["severity"], 3))

    gp = first_existing(R, "final_general_speech.csv", "general_speech.csv")
    general = pd.read_csv(gp) if gp is not None else pd.DataFrame(columns=["system", "librispeech_wer"])
    libri = [bar(s, general[general["system"] == s]["librispeech_wer"].mean()) for s in [stock, "Clear Voice"]
             if (general["system"] == s).any()]

    # Demo: the random draw, first sentence of at most 8 s per severity.
    draw = pd.read_csv(first_existing(R, "final_demo_clips.csv", "demo_clips.csv"), keep_default_na=False, na_values=[])
    sentences = draw[(draw["prompt_type"] == "sentence") & (draw["row_id"].map(dur) <= 8.0)]

    # Opening: fixed rule, see the module docstring.
    if ft is not None and stock in zs:
        strict = ft[(ft["part"] == "test_strict") & ft["is_dysarthric"].astype(bool) & (ft["prompt_type"] == "sentence")
                    & ft["severity"].isin(["severe", "moderate"])].copy()
        picked = [int(g.iloc[0]["row_id"]) for sev in ["mild", "moderate", "severe"]
                  for g in [sentences[sentences["severity"] == sev]] if len(g)]
        strict = strict[strict["row_id"].map(dur).between(2, 5) & ~strict["row_id"].isin(picked)]
        stock_hyp = zs[stock].set_index("row_id")["hyp"]
        strict["gain"] = [tokens(r, stock_hyp.get(rid, ""))[1] - tokens(r, hyp)[1]
                          for rid, r, hyp in zip(strict["row_id"], strict["ref"], strict["hyp"])]
        good = strict[strict["gain"] > 0].sort_values("gain", kind="stable")
        opening_row = good.iloc[int(0.75 * (len(good) - 1))] if len(good) else strict.iloc[0]
        opening_row = opening_row.rename({"hyp": "Clear Voice"})
        rule = "75th percentile of improvement among unseen severe or moderate sentences"
    else:
        opening_row = opening_from_draw(draw, stock)
        rule = "largest improvement among the random draw's moderate or severe clips"

    picks = [g.iloc[0] for sev in ["mild", "moderate", "severe"]
             for g in [sentences[(sentences["severity"] == sev) & (sentences["row_id"] != opening_row["row_id"])]] if len(g)]

    def hyps_for(row: pd.Series) -> dict:
        out = {s: row[s] for s in systems if s in row.index and isinstance(row[s], str)}
        for s, df in zs.items():
            if s not in out:
                m = df[df["row_id"] == int(row["row_id"])]
                if len(m):
                    out[s] = m["hyp"].iloc[0]
        return out

    rows = [opening_row] + picks
    audio = load_audio(download_parquets(), meta, [int(r["row_id"]) for r in rows])
    opening = clip_entry(opening_row, hyps_for(opening_row), meta, audio, systems)
    demo = [clip_entry(p, hyps_for(p), meta, audio, systems) for p in picks]

    data = {
        "placeholder": a.preview,
        "stockName": stock,
        "largeRatio": large_ratio(stock),
        "team": a.team,
        "repoUrl": "github.com/armaan-khoja/clear-voice",
        "demoUrl": a.demo_url,
        "nRecordings": int(len(meta)),
        "nSpeakers": int(meta["speaker"].nunique()),
        "nDysarthric": int(meta[meta["is_dysarthric"].astype(bool)]["speaker"].nunique()),
        "trainMinutes": TRAIN_MINUTES.get(stock, 60),
        "headline": [bar(s, h.loc[s, "sentence_wer"], h.loc[s, "sentence_wer_lo"], h.loc[s, "sentence_wer_hi"]) for s in systems if s in h.index],
        "headlineWords": [bar(s, h.loc[s, "word_cer"]) for s in systems if s in h.index],
        "leakage": [bar(c, w) for c, w in zip(lk["condition"], lk["sentence_wer"])],
        "stockTypical": None if stock_typical is None else round(float(stock_typical), 4),
        "librispeech": libri,
        "personalisation": personal,
        "opening": opening,
        "demo": demo,
        "hasVoiceover": (PUBLIC / "voiceover.wav").exists(),
    }
    (ROOT / "video" / "src" / "data.json").write_text(json.dumps(data, indent=2) + "\n")
    print(f"stock baseline {stock}; opening {opening['speaker']} ({opening['severity']}, {rule}); "
          f"demo {[d['speaker'] for d in demo]}")


if __name__ == "__main__":
    main()
