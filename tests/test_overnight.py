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


def test_fourth_round_retries_turbo_then_rebuilds_the_final_tables():
    from scripts.overnight import plan_round4

    steps = plan_round4()
    turbo = [(s.env["CV_HELDOUT"], s.env["CV_BASE"]) for s in steps if s.notebook.endswith("03_finetune.ipynb")]
    assert turbo == [("A", "large-v3-turbo"), ("B", "large-v3-turbo")]
    assert steps[-1].notebook == "scripts/final_report.py"


def test_a_step_marked_rerun_is_not_skipped_when_its_outputs_already_exist(tmp_path):
    from scripts.overnight import Step, should_skip

    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "final_summary.md").write_text("old\n")
    once = Step("tables", "scripts/final_report.py", ["results/final_summary.md"])
    again = Step("tables", "scripts/final_report.py", ["results/final_summary.md"], rerun=True)
    assert should_skip(once, str(tmp_path)) and not should_skip(again, str(tmp_path))


def test_model_choice_is_written_from_every_validation_file(tmp_path):
    # Round 4 rebuilds the final tables from model_choice.csv, so it must include models trained in the same run.
    import pandas as pd

    from scripts.overnight import write_model_choice

    (tmp_path / "results").mkdir()
    for m, fold, hyp in [("holdA", "A", "the hat"), ("holdA_large-v3-turbo", "A", "the cat")]:
        pd.DataFrame({"ref": ["the cat"], "hyp": [hyp], "fold": fold, "system": "Clear Voice"}).to_csv(
            tmp_path / "results" / f"val_ft_{m}.csv", index=False)
    write_model_choice(str(tmp_path))
    choice = pd.read_csv(tmp_path / "results" / "model_choice.csv")
    assert sorted(choice.model) == ["holdA", "holdA_large-v3-turbo"]
