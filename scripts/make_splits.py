"""Build every split from TORGO metadata, write splits/*.csv, and append a summary to results/data_report.md."""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from clearvoice.data import download_parquets, load_metadata  # noqa: E402
from clearvoice.splits import (  # noqa: E402
    FOLDS, NEAR_DUPLICATE, assign_prompt_splits, fold_splits, personal_subsets, prompt_clusters, random_splits,
    write_splits,
)

# Personalisation: one severe and one moderate speaker, both in fold A (stretch: M03 mild, M04 severe).
PERSONAL_SPEAKERS = ["M01", "F03", "M03", "M04"]

cache = ROOT / ".superpowers" / "meta.pkl"
meta = pd.read_pickle(cache) if cache.exists() else load_metadata(download_parquets())
ps = assign_prompt_splits(meta)
out = ROOT / "splits"
for old in out.glob("*.csv"):  # start clean so no stale split (e.g. a now-capped personal size) survives
    old.unlink()
lines = ["", "## Splits (seed 42)", ""]
counts = pd.Series(ps).value_counts()
lines += [f"Prompts: train {counts['train']}, val {counts['val']}, test {counts['test']}", ""]
merged = [c for t, g in meta.groupby("prompt_type") for c in prompt_clusters(sorted(g["text_norm"].unique())) if len(c) > 1]
lines += [f"Near-duplicate prompt groups kept in one split (similarity > {NEAR_DUPLICATE}): {len(merged)}, "
          f"e.g. {merged[:3]}", ""]
from difflib import SequenceMatcher  # noqa: E402
train_p = [p for p, v in ps.items() if v == "train"]
leaks = [p for p, v in ps.items() if v != "train"
         and any(SequenceMatcher(None, p, t).ratio() > NEAR_DUPLICATE for t in train_p)]
lines += [f"Validation or test prompts more than {NEAR_DUPLICATE} similar to a training prompt: {len(leaks)}", ""]

for held in FOLDS:
    s = fold_splits(meta, ps, held)
    write_splits(s, out, f"hold{held}")
    rows = []
    for part, df in s.items():
        rows.append({"part": part, "clips": len(df), "speakers": df["speaker"].nunique(),
                     "dysarthric clips": int(df["is_dysarthric"].sum()), "minutes": round(df["duration"].sum() / 60, 1)})
    lines += [f"### Model that never hears fold {held} ({', '.join(FOLDS[held]['dys'] + FOLDS[held]['ctl'])})", "",
              pd.DataFrame(rows).to_markdown(index=False), ""]
    strict = s["test_strict"].groupby("speaker").agg(clips=("row_id", "size"),
                                                     words=("prompt_type", lambda p: (p == "word").sum()),
                                                     sentences=("prompt_type", lambda p: (p == "sentence").sum()))
    lines += [f"Strict test clips per held-out speaker (fold {held}):", "", strict.to_markdown(), ""]

write_splits(random_splits(meta), out, "random")

lines += ["### Personalisation pools (one copy per utterance, training prompts only)", ""]
prow = []
for spk in PERSONAL_SPEAKERS:
    subs = personal_subsets(meta, ps, spk, minutes=(2, 5, 10, 20))
    for k, v in subs.items():
        if isinstance(v, pd.DataFrame):
            v[["row_id", "path", "speaker", "utt_key", "mic", "prompt_type", "duration"]].to_csv(
                out / f"personal_{spk}_{k}min.csv", index=False)
    prow.append({"speaker": spk, **{f"{k} min": ("capped" if isinstance(v, str) else f"{len(v)} clips")
                                    for k, v in subs.items()}})
lines += [pd.DataFrame(prow).to_markdown(index=False), ""]

with open(ROOT / "results" / "data_report.md", "a") as f:
    f.write("\n".join(lines))
print("\n".join(lines))
