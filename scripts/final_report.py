"""Build the final results from the model chosen on validation.

Notebook 5 built the tables from whisper-small. After the upgrade experiments, Clear Voice is whichever base model
(small, medium, large-v3-turbo) has the lowest mean validation WER across both folds (report.choose_architecture).
This script rebuilds the headline, leakage and demo-clip tables for that model, and lists every candidate on the
strict test so nothing is hidden. Writes results/final_*.csv and results/final_summary.md.

In Colab: %run scripts/final_report.py   (or locally: python scripts/final_report.py <results dir> <meta.pkl>)
"""
import os
import sys

import pandas as pd


def main(work: str, meta_path: str | None = None) -> None:
    import importlib

    from clearvoice import report

    importlib.reload(report)  # sessions that imported report before choose_architecture existed
    from clearvoice.pipeline import align_folds, read_preds

    R = f"{work}/results"
    choice = pd.read_csv(f"{R}/model_choice.csv")
    arch = report.choose_architecture(choice)
    suffix = "" if arch == "small" else f"_{arch}"
    label = {"small": "whisper-small", "medium": "whisper-medium"}.get(arch, f"whisper-{arch}")

    def ft(sfx: str) -> pd.DataFrame:
        frames = [read_preds(f"{R}/preds_ft_hold{f}{sfx}.csv") for f in ["A", "B"] if os.path.exists(f"{R}/preds_ft_hold{f}{sfx}.csv")]
        df = pd.concat(frames, ignore_index=True)
        df["system"] = "Clear Voice"
        return df

    chosen = ft(suffix)
    zs = align_folds({s: read_preds(f"{R}/preds_zeroshot_{s}.csv") for s in ["whisper-small", "whisper-large-v3"]}, chosen)
    head = report.headline({**zs, "Clear Voice": chosen})
    lk = report.leakage(chosen)

    candidates = {}
    for a in sorted(choice["model"].map(lambda m: m.split("_", 1)[1] if "_" in m else "small").unique()):
        sfx = "" if a == "small" else f"_{a}"
        if all(os.path.exists(f"{R}/preds_ft_hold{f}{sfx}.csv") for f in ["A", "B"]):
            candidates[f"Clear Voice ({a})"] = ft(sfx)
    cand = report.headline({**zs, **candidates})
    val_means = choice.assign(arch=choice["model"].map(lambda m: m.split("_", 1)[1] if "_" in m else "small")).groupby("arch")["wer"].mean()
    cand["validation_wer"] = cand["system"].map(lambda s: val_means.get(s[len("Clear Voice ("):-1]) if s.startswith("Clear Voice (") else None)

    demo = report.pick_demo_clips(chosen, per_severity=5).rename(columns={"hyp": "Clear Voice"})
    for s, df in zs.items():
        demo = demo.merge(df[["row_id", "hyp"]].rename(columns={"hyp": s}), on="row_id", how="left")
    meta = pd.read_pickle(meta_path or f"{work}/meta.pkl")
    demo = demo.merge(meta[["row_id", "file_idx", "row_in_file"]], on="row_id", how="left")

    lib = pd.read_csv(f"{R}/librispeech_zeroshot.csv")
    for f in ["A", "B"]:
        p = f"{R}/librispeech_ft_hold{f}{suffix}.csv"
        if os.path.exists(p):
            lib = pd.concat([lib, pd.read_csv(p).assign(system="Clear Voice")], ignore_index=True)

    head.to_csv(f"{R}/final_headline.csv", index=False)
    lk.to_csv(f"{R}/final_leakage.csv", index=False)
    cand.to_csv(f"{R}/final_candidates.csv", index=False)
    demo.to_csv(f"{R}/final_demo_clips.csv", index=False)
    lib.to_csv(f"{R}/final_general_speech.csv", index=False)
    with open(f"{R}/final_summary.md", "w") as fh:
        fh.write(f"# Final results\n\nClear Voice = fine-tuned {label}, chosen by mean validation WER over both folds.\n\n")
        fh.write("## Validation (chooses the model)\n\n" + choice.to_markdown(index=False) + "\n\n")
        fh.write("## Headline\n\n" + head.round(3).to_markdown(index=False) + "\n\n")
        fh.write("## Every candidate on the strict test\n\n" + cand.round(3).to_markdown(index=False) + "\n\n")
        fh.write("## Leakage\n\n" + lk.round(3).to_markdown(index=False) + "\n\n")
        fh.write("## Everyday English\n\n" + lib.round(3).to_markdown(index=False) + "\n")
    print(f"Clear Voice = {label}; wrote final_* tables", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 2 and not sys.argv[1].startswith("-"):
        main(sys.argv[1].rsplit("/results", 1)[0], sys.argv[2])
    else:
        from IPython import get_ipython

        main(get_ipython().user_ns["WORK"])
