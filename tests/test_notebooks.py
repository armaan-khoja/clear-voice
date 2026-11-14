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
