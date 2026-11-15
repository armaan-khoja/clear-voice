import ast
import importlib
import re
from pathlib import Path

import nbformat
import pytest

NB_DIR = Path(__file__).resolve().parents[1] / "notebooks"
NOTEBOOKS = ["01_data_and_splits", "02_zero_shot_baselines", "03_finetune", "04_personalise",
             "05_results_and_failures"]


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_notebook_parses_and_every_package_name_it_uses_exists(name):
    nb = nbformat.read(NB_DIR / f"{name}.ipynb", as_version=4)
    code = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    python = "\n".join(l for l in code.splitlines() if not l.lstrip().startswith(("!", "%")))
    ast.parse(python)  # valid Python once Colab magics are removed
    # Modules imported whole (e.g. `from clearvoice import plots`): every `plots.x(` call must exist.
    for modules in re.findall(r"from clearvoice import ([\w, ]+)", python):
        for module in [m.strip() for m in modules.split(",")]:
            mod = importlib.import_module(f"clearvoice.{module}")
            for attr in re.findall(rf"\b{module}\.([A-Za-z_]+)\(", python):
                assert hasattr(mod, attr), f"{name}: {module}.{attr}"
    for module, names in re.findall(r"from clearvoice\.(\w+) import ([\w, ]+)", python):
        mod = importlib.import_module(f"clearvoice.{module}")
        for n in [n.strip() for n in names.split(",")]:
            assert hasattr(mod, n), f"{name}: clearvoice.{module}.{n}"


def _setup_code():
    nb = nbformat.read(NB_DIR / "01_data_and_splits.ipynb", as_version=4)
    return [c.source for c in nb.cells if c.cell_type == "code"][0]


def test_smoke_runs_write_to_their_own_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ns = {}
    exec(_setup_code().replace("SMOKE = False", "SMOKE = True"), ns)
    assert ns["WORK"].endswith("/smoke")


def test_real_runs_write_to_the_main_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ns = {}
    exec(_setup_code(), ns)
    assert not ns["WORK"].endswith("/smoke")


def test_colab_can_run_without_google_drive(tmp_path, monkeypatch):
    """CLEARVOICE_WORK points a Colab run at a local folder, e.g. when nobody is there to approve Drive access."""
    import os
    import subprocess
    import sys
    import types

    def no_drive(*args, **kwargs):
        raise AssertionError("mounted Google Drive")

    def no_secret(key):
        raise KeyError(key)

    colab = types.ModuleType("google.colab")
    colab.drive, colab.userdata = types.SimpleNamespace(mount=no_drive), types.SimpleNamespace(get=no_secret)
    google = types.ModuleType("google")
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(os, "chdir", lambda path: None)
    monkeypatch.setenv("CLEARVOICE_WORK", str(tmp_path / "local"))
    ns = {}
    exec(_setup_code(), ns)
    assert ns["WORK"] == str(tmp_path / "local")
    assert (tmp_path / "local" / "results").is_dir()


def _finetune_code():
    nb = nbformat.read(NB_DIR / "03_finetune.ipynb", as_version=4)
    return [c.source for c in nb.cells if c.cell_type == "code"]


def _finetune_settings():
    return next(c for c in _finetune_code() if "HELDOUT =" in c)


def test_finetune_settings_default_to_the_main_model(monkeypatch):
    for k in ["CV_HELDOUT", "CV_BASE", "CV_PUSH_TO_HUB"]:
        monkeypatch.delenv(k, raising=False)
    ns = {}
    exec(_finetune_settings(), ns)
    assert (ns["HELDOUT"], ns["name"], ns["SYSTEM"], ns["TRAIN_OVERRIDES"], ns["PUSH_TO_HUB"]) == \
        ("A", "holdA", "Clear Voice", {}, True)


def test_finetune_settings_can_come_from_the_environment(monkeypatch):
    from clearvoice.train import TrainConfig

    monkeypatch.setenv("CV_HELDOUT", "B")
    monkeypatch.setenv("CV_BASE", "medium")
    monkeypatch.setenv("CV_PUSH_TO_HUB", "0")
    ns = {}
    exec(_finetune_settings(), ns)
    assert ns["name"] == "holdB_medium" and ns["SYSTEM"] == "Clear Voice (medium)" and ns["PUSH_TO_HUB"] is False
    assert TrainConfig(**ns["TRAIN_OVERRIDES"]).base_model == "openai/whisper-medium"


def test_finetune_reads_splits_by_fold_and_saves_validation_predictions():
    code = "\n".join(_finetune_code())
    assert 'rows_for(meta, SPLITS, f"hold{HELDOUT}", "train")' in code
    assert "preds_ft_{name}.csv" in code and "val_ft_{name}.csv" in code
