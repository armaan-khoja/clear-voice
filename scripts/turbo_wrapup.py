"""Finish the large-v3-turbo run that trained without Google Drive (Sunday 15 November 2026).

The Mac's battery died overnight and the new Colab runtime could not be connected to Drive with nobody there to
approve it, so turbo trained on Colab's own disk (CLEARVOICE_WORK=/content/work). That disk has none of the earlier
results. This script:

1. compares turbo's validation scores with fine-tuned whisper-small and whisper-medium, whose scores are copied
   below from MyDrive/clearvoice/results/model_choice.csv, and picks the architecture by mean validation WER
   (report.choose_architecture), never by test scores;
2. scores turbo on the strict test (headline, leakage, everyday English);
3. if turbo is chosen, reruns stock whisper-small and whisper-large-v3 (notebook 2's baselines), rebuilds the final
   tables with scripts/final_report.py and picks the video's opening clip by the documented rule;
4. writes everything into a new notebook cell (set_next_input), where it can be copied out.

In Colab, after `%run scripts/overnight.py --round4`:
    %run scripts/turbo_wrapup.py
"""
import json
import os

import pandas as pd

# MyDrive/clearvoice/results/model_choice.csv as of 05:29 ET, 15 November 2026.
KNOWN_CHOICE = [
    {"model": "holdA_medium", "fold": "A", "system": "Clear Voice (medium)", "wer": 0.168839, "cer": 0.0978396, "n": 650},
    {"model": "holdA", "fold": "A", "system": "Clear Voice", "wer": 0.18017, "cer": 0.103269, "n": 650},
    {"model": "holdB_medium", "fold": "B", "system": "Clear Voice (medium)", "wer": 0.132776, "cer": 0.0800966, "n": 384},
    {"model": "holdB", "fold": "B", "system": "Clear Voice", "wer": 0.164345, "cer": 0.0951496, "n": 384},
]
TURBO = "large-v3-turbo"


def merge_choice(known: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """One validation table from both sources, with `chosen` recomputed per fold."""
    cols = ["model", "fold", "system", "wer", "cer", "n"]
    t = pd.concat([known[cols], new[cols]], ignore_index=True)
    t["chosen"] = t["wer"] == t.groupby("fold")["wer"].transform("min")
    return t.sort_values(["fold", "wer"]).reset_index(drop=True)


def main(work: str) -> None:
    import importlib
    import sys

    from clearvoice import report
    from clearvoice.pipeline import read_preds

    importlib.reload(report)
    R = f"{work}/results"
    out = []

    def section(title: str, text: str) -> None:
        out.append(f"## {title}\n{text.strip()}\n")

    val = {f"hold{f}_{TURBO}": pd.read_csv(f"{R}/val_ft_hold{f}_{TURBO}.csv", keep_default_na=False, na_values=[])
           for f in "AB" if os.path.exists(f"{R}/val_ft_hold{f}_{TURBO}.csv")}
    new = report.compare_validation(val) if val else pd.DataFrame(columns=["model", "fold", "system", "wer", "cer", "n"])
    choice = merge_choice(pd.DataFrame(KNOWN_CHOICE), new)
    choice.to_csv(f"{R}/model_choice.csv", index=False)
    arch = report.choose_architecture(choice)
    section("validation", choice.to_csv(index=False) + f"chosen architecture: {arch}")

    preds = [f"{R}/preds_ft_hold{f}_{TURBO}.csv" for f in "AB"]
    if all(os.path.exists(p) for p in preds):
        ft = pd.concat([read_preds(p) for p in preds], ignore_index=True)
        ft["system"] = "Clear Voice"
        systems = {"whisper-large-v3-turbo": read_preds(f"{R}/preds_zeroshot_whisper-large-v3-turbo.csv"), "Clear Voice": ft}
        section("turbo strict test", report.headline(systems).round(4).to_csv(index=False))
        section("turbo leakage", report.leakage(ft).round(4).to_csv(index=False))
        libri = [pd.read_csv(f"{R}/librispeech_ft_hold{f}_{TURBO}.csv") for f in "AB" if os.path.exists(f"{R}/librispeech_ft_hold{f}_{TURBO}.csv")]
        if libri:
            section("turbo everyday English", pd.concat(libri).to_csv(index=False))

    if arch == TURBO:
        sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))
        import final_report
        import zero_shot_extra

        if not all(os.path.exists(f"{R}/preds_zeroshot_{s}.csv") for s in ["whisper-small", "whisper-large-v3"]):
            extra = f"{R}/librispeech_zeroshot_extra.csv"
            keep = pd.read_csv(extra) if os.path.exists(extra) else None
            zero_shot_extra.main(work, bases=["openai/whisper-small", "openai/whisper-large-v3"])
            os.replace(extra, f"{R}/librispeech_zeroshot.csv")  # the same file notebook 2 writes
            if keep is not None:
                keep.to_csv(extra, index=False)
        final_report.main(work)
        section("final summary", open(f"{R}/final_summary.md").read())
        section("final demo clips", pd.read_csv(f"{R}/final_demo_clips.csv").to_csv(index=False))

        from video_data import opening_by_rule

        meta = pd.read_pickle(f"{work}/meta.pkl")
        dur = meta.set_index("row_id")["duration"]
        ft = pd.concat([read_preds(p) for p in preds], ignore_index=True)
        draw = pd.read_csv(f"{R}/final_demo_clips.csv", keep_default_na=False, na_values=[])
        sentences = draw[(draw["prompt_type"] == "sentence") & (draw["row_id"].map(dur) <= 8.0)]
        picked = [int(g.iloc[0]["row_id"]) for sev in ["mild", "moderate", "severe"]
                  for g in [sentences[sentences["severity"] == sev]] if len(g)]
        zs = {s: read_preds(f"{R}/preds_zeroshot_{s}.csv").set_index("row_id")["hyp"]
              for s in ["whisper-large-v3-turbo", "whisper-large-v3", "whisper-small"]}
        row = opening_by_rule(ft, zs["whisper-large-v3-turbo"], dur, exclude=picked)
        rid = int(row["row_id"])
        opening = {"row_id": rid, "speaker": row["speaker"], "severity": row["severity"], "ref": row["ref"],
                   "Clear Voice": row["Clear Voice"], **{s: zs[s].get(rid) for s in zs}}
        section("opening clip", json.dumps(opening))

    from IPython import get_ipython

    get_ipython().set_next_input("# TURBO WRAP-UP\n" + "\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    from IPython import get_ipython

    main(get_ipython().user_ns["WORK"])
