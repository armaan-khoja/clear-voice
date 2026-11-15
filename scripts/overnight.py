"""Run the remaining notebooks one after another in the Colab session that already has Google Drive connected.

In a Colab cell, after notebook 3 has run once:
    !git -C /content/clear-voice pull -q
    %run scripts/overnight.py              (round 1: fold B, baselines, personalisation, results, medium)
    %run scripts/overnight.py --round2     (round 2: tuned personalisation, large-v3-turbo)
    %run scripts/overnight.py --round3     (round 3: personalisation tuned on new sentences)
    %run scripts/overnight.py --round4     (round 4: large-v3-turbo again, then the final tables rebuilt)
    %run scripts/overnight.py --round5     (round 5: stock medium and turbo as baselines, final tables rebuilt)

Each step is skipped if its output files already exist, so rerunning after a disconnect resumes where it stopped.
Progress is appended to MyDrive/clearvoice/results/overnight_log.txt.
"""
import gc
import os
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Step:
    label: str
    notebook: str
    outputs: list
    env: dict = field(default_factory=dict)
    rerun: bool = False  # run even if the outputs exist (tables that must be rebuilt from newer results)


def should_skip(step: Step, work: str) -> bool:
    return not step.rerun and all(os.path.exists(f"{work}/{o}") for o in step.outputs)


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


def plan_round2() -> list:
    """After the first round: tune personalisation, then try whisper-large-v3-turbo as the base model."""
    nb = "notebooks/{}.ipynb".format
    return [
        Step("personal tuning", "scripts/personal_tune.py", ["results/personalise_tuned.csv"]),
        Step("turbo, fold A", nb("03_finetune"), ["results/val_ft_holdA_large-v3-turbo.csv"], _finetune("A", "large-v3-turbo")),
        Step("turbo, fold B", nb("03_finetune"), ["results/val_ft_holdB_large-v3-turbo.csv"], _finetune("B", "large-v3-turbo")),
    ]


def plan_round3() -> list:
    """Personalisation tuned on validation prompts (new sentences), then the final tables for the chosen model."""
    return [
        Step("personal tuning on new sentences", "scripts/personal_tune.py", ["results/personalise_tuned_v2.csv"]),
        Step("final tables", "scripts/final_report.py", ["results/final_summary.md"]),
    ]


def plan_round4() -> list:
    """large-v3-turbo again (round 2 loaded it in float16 and crashed at the first evaluation), then the final
    tables rebuilt so the model is chosen on validation among small, medium and turbo."""
    nb = "notebooks/{}.ipynb".format
    return [
        Step("turbo, fold A", nb("03_finetune"), ["results/val_ft_holdA_large-v3-turbo.csv"], _finetune("A", "large-v3-turbo")),
        Step("turbo, fold B", nb("03_finetune"), ["results/val_ft_holdB_large-v3-turbo.csv"], _finetune("B", "large-v3-turbo")),
        Step("final tables", "scripts/final_report.py", ["results/final_summary.md"], rerun=True),
    ]


def plan_round5() -> list:
    """Stock whisper-medium and large-v3-turbo on the strict test, so Clear Voice is compared with the model it
    started from, then the final tables rebuilt."""
    return [
        Step("stock medium and turbo", "scripts/zero_shot_extra.py",
             ["results/preds_zeroshot_whisper-medium.csv", "results/preds_zeroshot_whisper-large-v3-turbo.csv"]),
        Step("final tables", "scripts/final_report.py", ["results/final_summary.md"], rerun=True),
    ]


def validation_files(work: str) -> list:
    """Model names that have validation predictions, e.g. ['holdA', 'holdA_medium']."""
    import glob

    return [Path(p).name[len("val_ft_"):-len(".csv")] for p in glob.glob(f"{work}/results/val_ft_*.csv")]


def write_model_choice(work: str):
    """Compare every model with validation predictions and write results/model_choice.csv (None if there are none)."""
    import importlib

    import pandas as pd

    from clearvoice import report

    importlib.reload(report)  # this session imported the module before compare_validation existed
    val = {m: pd.read_csv(f"{work}/results/val_ft_{m}.csv", keep_default_na=False, na_values=[])
           for m in sorted(validation_files(work))}
    if not val:
        return None
    table = report.compare_validation(val)
    table.to_csv(f"{work}/results/model_choice.csv", index=False)
    return table


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


def main(work: str, steps: list | None = None) -> None:
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

    for step in steps or plan():
        if should_skip(step, work):
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
        write_model_choice(work)  # a later step (the final tables) reads it

    table = write_model_choice(work)
    if table is not None:
        log("validation comparison:\n" + table.to_string(index=False))
    log("overnight run finished")


if __name__ == "__main__":
    import sys

    from IPython import get_ipython

    rounds = {"--round2": plan_round2, "--round3": plan_round3, "--round4": plan_round4, "--round5": plan_round5}
    chosen = [rounds[a] for a in sys.argv if a in rounds]
    main(get_ipython().user_ns["WORK"], chosen[0]() if chosen else None)
