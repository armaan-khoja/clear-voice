import pandas as pd
import pytest

from clearvoice.splits import (
    assign_prompt_splits, fold_splits, personal_subsets, pick_eval_copies, read_split, write_splits,
)

SPEAKERS = {"F01": True, "M02": True, "FC01": False, "FC02": False}  # F01, FC01 in fold A; M02, FC02 in fold B
WORDS = [f"word{i}" for i in range(20)]
SENTS = [f"this is sentence {i}" for i in range(10)]


def make_meta():
    rows = []
    for spk, dys in SPEAKERS.items():
        for j, prompt in enumerate(WORDS + SENTS):
            for mic in ("headMic", "arrayMic"):
                rows.append({
                    "speaker": spk, "session": "1", "mic": mic, "utt": f"{j:04d}",
                    "utt_key": f"{spk}_1_{j:04d}", "path": f"{spk}_1_{mic}_{j:04d}.wav",
                    "text_norm": prompt, "prompt_type": "word" if " " not in prompt else "sentence",
                    "duration": 30.0, "is_dysarthric": dys,
                })
    meta = pd.DataFrame(rows)
    meta.insert(0, "row_id", range(len(meta)))
    return meta


def test_prompt_splits_are_disjoint_and_cover_every_prompt():
    ps = assign_prompt_splits(make_meta())
    assert set(ps) == set(WORDS + SENTS)
    assert set(ps.values()) == {"train", "val", "test"}


def test_prompt_splits_are_about_70_10_20_per_type():
    ps = assign_prompt_splits(make_meta())
    words = [ps[w] for w in WORDS]
    sents = [ps[s] for s in SENTS]
    assert (words.count("train"), words.count("val"), words.count("test")) == (14, 2, 4)
    assert (sents.count("train"), sents.count("val"), sents.count("test")) == (7, 1, 2)


def test_prompt_splits_are_deterministic():
    assert assign_prompt_splits(make_meta(), seed=42) == assign_prompt_splits(make_meta(), seed=42)


def test_fold_splits_keep_heldout_speakers_and_prompts_out_of_training():
    meta = make_meta()
    ps = assign_prompt_splits(meta)
    s = fold_splits(meta, ps, heldout="A")
    assert set(s["train"]["speaker"]) == {"M02", "FC02"}
    assert set(s["val"]["speaker"]) == {"M02", "FC02"}
    assert set(s["test_strict"]["speaker"]) == {"F01", "FC01"}
    assert set(s["test_seen"]["speaker"]) == {"F01", "FC01"}
    assert set(s["test_newsent"]["speaker"]) == {"M02", "FC02"}
    train_prompts = set(s["train"]["text_norm"])
    for part in ("val", "test_strict", "test_newsent"):
        assert not train_prompts & set(s[part]["text_norm"]), part
    assert set(s["test_seen"]["text_norm"]) <= train_prompts


def test_training_uses_both_microphones_but_tests_use_one_copy():
    meta = make_meta()
    s = fold_splits(meta, assign_prompt_splits(meta), heldout="B")
    assert set(s["train"]["mic"]) == {"headMic", "arrayMic"}
    for part in ("val", "test_seen", "test_strict", "test_newsent"):
        assert s[part]["utt_key"].is_unique, part


def test_pick_eval_copies_prefers_head_mic():
    meta = make_meta()
    meta = meta[~((meta["utt_key"] == "F01_1_0000") & (meta["mic"] == "headMic"))]
    picked = pick_eval_copies(meta)
    assert picked["utt_key"].is_unique
    assert picked.set_index("utt_key").loc["F01_1_0001", "mic"] == "headMic"
    assert picked.set_index("utt_key").loc["F01_1_0000", "mic"] == "arrayMic"


def test_personal_subsets_are_nested_and_reach_each_size():
    meta = make_meta()
    ps = assign_prompt_splits(meta)
    subs = personal_subsets(meta, ps, "F01", minutes=(1, 2, 3))
    ids = {k: set(v["row_id"]) for k, v in subs.items()}
    assert ids[1] <= ids[2] <= ids[3]
    for k, v in subs.items():
        assert v["duration"].sum() >= k * 60
        assert v["utt_key"].is_unique


def test_personal_subsets_mark_sizes_beyond_the_pool_as_capped():
    meta = make_meta()
    ps = assign_prompt_splits(meta)
    subs = personal_subsets(meta, ps, "F01", minutes=(2, 500))
    assert isinstance(subs[2], pd.DataFrame)
    assert subs[500] == "capped"


def test_personal_pool_never_contains_test_or_val_prompts():
    meta = make_meta()
    ps = assign_prompt_splits(meta)
    subs = personal_subsets(meta, ps, "F01", minutes=(5,))
    assert {ps[t] for t in subs[5]["text_norm"]} == {"train"}


def test_write_and_read_split_round_trip_without_transcripts(tmp_path):
    meta = make_meta()
    s = fold_splits(meta, assign_prompt_splits(meta), heldout="A")
    write_splits(s, tmp_path, "holdA")
    back = read_split(tmp_path, "holdA", "test_strict")
    assert list(back["row_id"]) == list(s["test_strict"]["row_id"])
    assert "text_norm" not in back.columns and "text" not in back.columns
