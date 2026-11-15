"""Run the remaining notebooks one after another in the Colab session that already has Google Drive connected.

In a Colab cell, after notebook 3 has run once:
    !git -C /content/clear-voice pull -q
    %run scripts/overnight.py

Each step is skipped if its output files already exist, so rerunning after a disconnect resumes where it stopped.
Progress is appended to MyDrive/clearvoice/results/overnight_log.txt.
"""
import gc
import os
import time
from dataclasses import dataclass, field


@dataclass
class Step:
    label: str
    notebook: str
    outputs: list
    env: dict = field(default_factory=dict)


def _finetune(fold: str, base: str) -> dict:
    return {"CV_HELDOUT": fold, "CV_BASE": base, "CV_PUSH_TO_HUB": "0"}


def plan() -> list:
    """Core results first (they are what gets submitted), then the whisper-medium upgrade experiment."""
    nb = "notebooks/{}.ipynb".format
    return [
        Step("fold B", nb("03_finetune"), ["results/preds_ft_holdB.csv", "results/val_ft_holdB.csv"],
             _finetune("B", "small")),
        Step("baselines", nb("02_zero_shot_baselines"),
             ["results/preds_zeroshot_whisper-small.csv", "results/preds_zeroshot_whisper-large-v3.csv"]),
        Step("personalisation", nb("04_personalise"), ["results/personalise.csv"]),
        Step("results", nb("05_results_and_failures"), ["results/summary.md"]),
        # Fold A ran before notebook 3 saved validation predictions; this reuses its model (no retraining).
        Step("fold A validation", nb("03_finetune"), ["results/val_ft_holdA.csv"], _finetune("A", "small")),
        Step("medium, fold A", nb("03_finetune"), ["results/val_ft_holdA_medium.csv"], _finetune("A", "medium")),
        Step("medium, fold B", nb("03_finetune"), ["results/val_ft_holdB_medium.csv"], _finetune("B", "medium")),
    ]


def _free_gpu(ip) -> None:
    """%run leaves a notebook's variables behind; drop the big ones so the next step has the GPU's memory."""
    for k in ["model", "processor", "trainer", "audio", "test_audio", "preds"]:
        ip.user_ns.pop(k, None)
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except ImportError:
        pass


def main(work: str) -> None:
    import pandas as pd
    from IPython import get_ipython

    ip = get_ipython()
    log_path = f"{work}/results/overnight_log.txt"

    def log(msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
        print(line, flush=True)
        with open(log_path, "a") as f:
            f.write(line + "\n")

    def finished(step: Step) -> bool:
        return all(os.path.exists(f"{work}/{o}") for o in step.outputs)

    for step in plan():
        if finished(step):
            log(f"skip  {step.label} (already done)")
            continue
        for k in [k for k in os.environ if k.startswith("CV_")]:
            del os.environ[k]
        os.environ.update(step.env)
        start = time.time()
        log(f"start {step.label}")
        try:
            ip.run_line_magic("run", step.notebook)
        except BaseException as e:  # keep going: later steps may not depend on this one
            log(f"error {step.label}: {type(e).__name__}: {e}")
        _free_gpu(ip)
        log(f"{'done ' if finished(step) else 'FAILED'} {step.label} ({(time.time() - start) / 60:.0f} min)")

    import importlib

    from clearvoice import report

    importlib.reload(report)  # this session imported the module before compare_validation existed
    val = {m: pd.read_csv(f"{work}/results/val_ft_{m}.csv", keep_default_na=False, na_values=[])
           for m in ["holdA", "holdB", "holdA_medium", "holdB_medium"]
           if os.path.exists(f"{work}/results/val_ft_{m}.csv")}
    if val:
        table = report.compare_validation(val)
        table.to_csv(f"{work}/results/model_choice.csv", index=False)
        log("validation comparison:\n" + table.to_string(index=False))
    log("overnight run finished")


if __name__ == "__main__":
    from IPython import get_ipython

    main(get_ipython().user_ns["WORK"])
