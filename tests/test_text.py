from clearvoice.text import normalize, prompt_type


def test_normalize_ignores_case_and_punctuation():
    assert normalize("Seven, Seven!") == normalize("seven seven")


def test_normalize_empty_stays_empty():
    assert normalize("   ") == ""


def test_prompt_type_single_word():
    assert prompt_type("yes") == "word"


def test_prompt_type_sentence():
    assert prompt_type("the quick brown fox") == "sentence"


def test_normalize_treats_missing_values_as_empty():
    assert normalize(float("nan")) == "" and normalize(None) == ""
