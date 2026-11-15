"""Tune the personalisation recipe on held-out recordings of each speaker, then rerun the curve on the test.

The first personalisation run (notebook 4) used a cautious recipe (learning rate 1e-5, 5 epochs) and barely moved.
This script picks a recipe using only each speaker's own training-prompt clips that are NOT in their 10-minute set
(clearvoice.splits.personal_validation), so the choice never sees the test sentences. It then reruns the 0, 2, 5 and
10 minute curve on the strict test with the chosen recipe.

In the Colab session that ran notebook 3 (Drive already connected):
    !git -C /content/clear-voice pull -q
    %run scripts/personal_tune.py

Writes results/personal_tuning.csv (validation score per recipe) and results/personalise_tuned.csv (the new curve).
"""
import shutil
import tempfile

import pandas as pd

RECIPES = {
    "original (lr 1e-5, 5 epochs)": dict(learning_rate=1e-5, num_train_epochs=5),
    "lr 5e-5, 10 epochs": dict(learning_rate=5e-5, num_train_epochs=10),
    "lr 1e-4, 10 epochs": dict(learning_rate=1e-4, num_train_epochs=10),
    "lr 5e-5, 20 epochs": dict(learning_rate=5e-5, num_train_epochs=20),
}
SPEAKERS = {"M01": "A", "F03": "A"}  # both held out by the fold-A model
MINUTES = [2, 5, 10]


def choose(scores: list) -> str:
    """The recipe with the lowest mean validation WER across speakers; ties go to the earlier (simpler) recipe."""
    df = pd.DataFrame(scores)
    means = df.groupby("recipe", sort=False)["wer"].mean()
    return means.idxmin()


def main(work: str, splits: str = "splits", limit: int | None = None, base_model: str | None = None) -> None:
    """`limit` and `base_model` exist for a quick local check (a few clips, whisper-tiny); Colab uses neither."""
    import importlib

    import torch

    from clearvoice import splits as splits_mod

    importlib.reload(splits_mod)  # this session imported the module before personal_validation existed
    from clearvoice.asr import load_model
    from clearvoice.data import load_audio
    from clearvoice.metrics import score
    from clearvoice.pipeline import load_torgo, predict, rows_for
    from clearvoice.train import TrainConfig, finetune

    parquets, meta = load_torgo(work)
    ps = splits_mod.assign_prompt_splits(meta)

    def train_and_score(base, adapt, rows, audio, recipe, part, fold):
        cfg = TrainConfig(base_model=base, max_steps=None, eval=False, batch_size=8, **RECIPES[recipe])
        tmp = tempfile.mkdtemp()  # personal models stay off Google Drive
        try:
            out = finetune(cfg, adapt, None, audio, tmp) if adapt is not None else base
            model, processor = load_model(out)
            p = predict(model, processor, parquets, meta, rows, "Clear Voice", fold, part, audio=audio)
            del model
            torch.cuda.empty_cache()
            return score(list(p.ref), list(p.hyp))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    scores = []
    for spk, fold in SPEAKERS.items():
        base = base_model or f"{work}/models/hold{fold}/best"
        adapt = rows_for(meta, splits, f"personal_{spk}", "10min").head(limit or None)
        val = splits_mod.personal_validation(meta, ps, spk, after_minutes=10)
        val = meta.set_index("row_id").loc[val["row_id"]].reset_index().head(limit or None)
        audio = load_audio(parquets, meta, list(adapt.row_id) + list(val.row_id))
        for recipe in RECIPES:
            s = train_and_score(base, adapt, val, audio, recipe, "personal_val", fold)
            scores.append({"recipe": recipe, "speaker": spk, "wer": s["wer"], "cer": s["cer"], "n_clips": s["n"]})
            print(f"{spk} {recipe}: validation WER {s['wer']:.3f} on {s['n']} clips", flush=True)
    pd.DataFrame(scores).to_csv(f"{work}/results/personal_tuning.csv", index=False)
    best = choose(scores)
    print("chosen recipe (by validation):", best, flush=True)

    curve = []
    for spk, fold in SPEAKERS.items():
        base = base_model or f"{work}/models/hold{fold}/best"
        test = rows_for(meta, splits, f"hold{fold}", "test_strict")
        test = test[test.speaker == spk].head(limit or None)
        test_audio = load_audio(parquets, meta, test.row_id)
        s0 = train_and_score(base, None, test, test_audio, best, "test_strict", fold)
        curve.append({"speaker": spk, "severity": test.severity.iloc[0], "minutes": 0, "wer": s0["wer"], "cer": s0["cer"], "n_clips": s0["n"], "recipe": best})
        for k in MINUTES:
            adapt = rows_for(meta, splits, f"personal_{spk}", f"{k}min").head(limit or None)
            audio = {**test_audio, **load_audio(parquets, meta, adapt.row_id)}
            s = train_and_score(base, adapt, test, audio, best, "test_strict", fold)
            curve.append({"speaker": spk, "severity": test.severity.iloc[0], "minutes": k, "wer": s["wer"], "cer": s["cer"], "n_clips": s["n"], "recipe": best})
            print(f"{spk} {k} min: test WER {s['wer']:.3f}", flush=True)
    pd.DataFrame(curve).to_csv(f"{work}/results/personalise_tuned.csv", index=False)
    print("personal tuning finished", flush=True)


if __name__ == "__main__":
    from IPython import get_ipython

    main(get_ipython().user_ns["WORK"])
