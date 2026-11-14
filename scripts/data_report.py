"""Download TORGO, load its metadata and write results/data_report.md plus results/meta_summary.csv."""
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from clearvoice.data import check_mic_pairs, download_parquets, load_audio, load_metadata  # noqa: E402

paths = download_parquets()
meta = load_metadata(paths)
meta.drop(columns=["text"]).to_pickle(ROOT / ".superpowers" / "meta.pkl")  # local cache, not committed

lines = ["# TORGO data report", "", f"Clips: {len(meta)}", ""]
per = meta.groupby("speaker").agg(clips=("row_id", "size"), minutes=("duration", lambda d: d.sum() / 60),
                                  severity=("severity", "first"), status=("speech_status", "first"),
                                  gender=("gender", "first"))
lines += ["## Per speaker (minutes count both microphone copies)", "", per.round(1).to_markdown(), ""]
lines += ["## Microphones", "", str(dict(Counter(meta["mic"]))), ""]
one_copy = meta.drop_duplicates("utt_key")
lines += [f"Unique utterances (speaker, session, utt): {len(one_copy)}", ""]
lines += [f"Unique normalised prompts: {meta['text_norm'].nunique()} "
          f"(words: {meta.loc[meta.prompt_type=='word','text_norm'].nunique()}, "
          f"sentences: {meta.loc[meta.prompt_type=='sentence','text_norm'].nunique()})", ""]
lines += [f"Empty normalised transcripts: {(meta['text_norm']=='').sum()}", ""]
bad = check_mic_pairs(meta)
lines += [f"Utterances whose mic copies disagree on transcript: {bad['utt_key'].nunique()}", ""]
if len(bad):
    lines += ["Examples:", "", bad.head(10)[["utt_key", "mic", "text_norm"]].to_markdown(index=False), ""]
dup = meta.duplicated(["speaker", "session", "mic", "utt"]).sum()
lines += [f"Exact duplicate filenames: {dup}", ""]

rng = np.random.default_rng(0)
sample = rng.choice(meta["row_id"], size=50, replace=False)
audio = load_audio(paths, meta, sample)
ratios = [len(audio[r]) / (meta.loc[r, "duration"] * 16000) for r in sample]
lines += [f"Decoded 50 random clips: length / (duration x 16 kHz) ranges {min(ratios):.3f} to {max(ratios):.3f}", ""]
lines += ["## Status values", "", str(dict(Counter(meta["speech_status"]))), "",
          "## Severity values", "", str(dict(Counter(meta["severity"]))), ""]
(ROOT / "results" / "data_report.md").write_text("\n".join(lines))
print("\n".join(lines))
