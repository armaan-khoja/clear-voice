def test_the_stock_model_clear_voice_started_from_is_a_baseline_when_its_predictions_exist(tmp_path):
    from scripts.final_report import stock_systems

    (tmp_path / "preds_zeroshot_whisper-medium.csv").write_text("x\n")
    assert stock_systems(str(tmp_path), "medium") == ["whisper-small", "whisper-medium", "whisper-large-v3"]
    assert stock_systems(str(tmp_path), "large-v3-turbo") == ["whisper-small", "whisper-large-v3"]  # not run yet
    assert stock_systems(str(tmp_path), "small") == ["whisper-small", "whisper-large-v3"]


def test_fifth_round_runs_the_stock_bases_then_rebuilds_the_final_tables():
    from scripts.overnight import plan_round5

    steps = plan_round5()
    assert steps[0].notebook == "scripts/zero_shot_extra.py"
    assert steps[-1].notebook == "scripts/final_report.py" and steps[-1].rerun
