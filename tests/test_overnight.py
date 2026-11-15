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


def test_second_round_tunes_personalisation_then_tries_turbo_on_both_folds():
    from scripts.overnight import plan_round2

    steps = plan_round2()
    assert steps[0].notebook == "scripts/personal_tune.py"
    turbo = [(s.env["CV_HELDOUT"], s.env["CV_BASE"]) for s in steps if s.notebook.endswith("03_finetune.ipynb")]
    assert turbo == [("A", "large-v3-turbo"), ("B", "large-v3-turbo")]
    assert all(s.env.get("CV_PUSH_TO_HUB", "0") == "0" for s in steps)


def test_validation_files_are_found_for_any_model_name(tmp_path):
    from scripts.overnight import validation_files

    (tmp_path / "results").mkdir()
    for m in ["holdA", "holdB_medium", "holdA_large-v3-turbo"]:
        (tmp_path / "results" / f"val_ft_{m}.csv").write_text("x\n")
    assert sorted(validation_files(str(tmp_path))) == ["holdA", "holdA_large-v3-turbo", "holdB_medium"]
