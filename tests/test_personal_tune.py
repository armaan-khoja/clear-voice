from scripts.personal_tune import RECIPES, choose


def test_choose_picks_the_recipe_with_the_lowest_mean_validation_wer():
    scores = [
        {"recipe": "a", "speaker": "M01", "wer": 0.50}, {"recipe": "a", "speaker": "F03", "wer": 0.30},
        {"recipe": "b", "speaker": "M01", "wer": 0.40}, {"recipe": "b", "speaker": "F03", "wer": 0.32},
    ]
    assert choose(scores) == "b"


def test_recipes_include_the_original_one_so_it_can_win():
    assert any(r["learning_rate"] == 1e-5 and r["num_train_epochs"] == 5 for r in RECIPES.values())
