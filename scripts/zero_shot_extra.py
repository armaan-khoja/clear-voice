"""Stock (not fine-tuned) whisper-medium and whisper-large-v3-turbo on the strict test.

Clear Voice is fine-tuned from whichever base model wins on validation (scripts/final_report.py), so the final tables
compare it with the exact model it started from, as well as stock whisper-small and whisper-large-v3 (notebook 2).

In the Colab session that ran notebook 3 (Drive already connected):
    %run scripts/zero_shot_extra.py

Writes results/preds_zeroshot_whisper-<base>.csv for each base and results/librispeech_zeroshot_extra.csv.
"""
import pandas as pd

BASES = ["openai/whisper-medium", "openai/whisper-large-v3-turbo"]


def main(work: str, splits: str = "splits", bases: list = BASES, limit: int | None = None,
         n_librispeech: int = 200) -> None:
    """`limit` and `n_librispeech` exist for a quick local check; Colab uses the defaults."""
    import torch

    from clearvoice.asr import load_model
    from clearvoice.pipeline import librispeech_check, load_torgo, predict_fold

    parquets, meta = load_torgo(work)
    general = []
    for name in bases:
        short = name.split("/")[1]
        model, processor = load_model(name)
        preds = pd.concat([predict_fold(model, processor, parquets, meta, splits, fold, ["test_strict"], short,
                                        batch_size=32, limit=limit) for fold in ["A", "B"]], ignore_index=True)
        preds.to_csv(f"{work}/results/preds_zeroshot_{short}.csv", index=False)
        ls = librispeech_check(model, processor, n=n_librispeech, batch_size=32)
        general.append({"system": short, "librispeech_wer": ls["wer"], "n": ls["n"]})
        print(short, "done", flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    pd.DataFrame(general).to_csv(f"{work}/results/librispeech_zeroshot_extra.csv", index=False)


if __name__ == "__main__":
    from IPython import get_ipython

    main(get_ipython().user_ns["WORK"])
