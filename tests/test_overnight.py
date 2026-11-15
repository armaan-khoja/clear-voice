from scripts.overnight import plan


def test_core_results_run_before_the_upgrade_experiment():
    labels = [s.label for s in plan()]
    order = ["fold B", "baselines", "personalisation", "results", "medium, fold A", "medium, fold B"]
    assert [labels.index(x) for x in order] == sorted(labels.index(x) for x in order)


def test_finetune_steps_set_their_fold_and_never_upload():
    steps = [s for s in plan() if s.notebook.endswith("03_finetune.ipynb")]
    assert {(s.env["CV_HELDOUT"], s.env["CV_BASE"]) for s in steps} >= {("B", "small"), ("A", "medium"), ("B", "medium")}
    assert all(s.env["CV_PUSH_TO_HUB"] == "0" for s in steps)


def test_every_step_names_the_files_that_prove_it_finished():
    assert all(s.outputs and all(o.startswith("results/") for o in s.outputs) for s in plan())
