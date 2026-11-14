"""Leakage-safe splits: unseen sentences, unseen speakers, one copy per utterance when scoring."""
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
NEAR_DUPLICATE = 0.8  # prompts more similar than this are treated as one sentence

# Each fold: two severe dysarthric speakers, one mid, one mild, plus controls.
FOLDS = {
    "A": {"dys": ["F01", "M01", "F03", "M03"], "ctl": ["FC01", "MC01", "MC02"]},
    "B": {"dys": ["M02", "M04", "M05", "F04"], "ctl": ["FC02", "FC03", "MC03", "MC04"]},
}
MAX_SECONDS = 30.0  # Whisper hears at most 30 s; TORGO has 2 clips of ~3 minutes holding one word.
SPLIT_COLUMNS = ["row_id", "path", "speaker", "utt_key", "mic", "prompt_type", "duration"]


def prompt_clusters(prompts: list) -> list:
    """Group prompts that are near-duplicates (similarity above NEAR_DUPLICATE), e.g. 'as i choose' / 'as i chose'.

    TORGO has transcript variants of the same sentence; splitting them apart would leak test sentences into training.
    """
    parent = {p: p for p in prompts}

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    sm = SequenceMatcher(autojunk=False)
    for i, a in enumerate(prompts):
        sm.set_seq2(a)
        for b in prompts[i + 1:]:
            sm.set_seq1(b)
            if sm.real_quick_ratio() > NEAR_DUPLICATE and sm.quick_ratio() > NEAR_DUPLICATE \
                    and sm.ratio() > NEAR_DUPLICATE:
                parent[root(b)] = root(a)
    groups = {}
    for p in prompts:
        groups.setdefault(root(p), []).append(p)
    return [sorted(g) for g in sorted(groups.values(), key=lambda g: min(g))]


def assign_prompt_splits(meta: pd.DataFrame, seed: int = SEED) -> dict:
    """Map each unique normalised prompt to 'train' (70%), 'val' (10%) or 'test' (20%), per prompt type.

    Near-duplicate prompts are kept together in one split.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for _, group in meta.groupby("prompt_type"):
        clusters = prompt_clusters(sorted(group["text_norm"].unique()))
        rng.shuffle(clusters)
        n = sum(len(c) for c in clusters)
        n_test, n_val = round(0.2 * n), round(0.1 * n)
        counts = {"test": 0, "val": 0}
        for cluster in clusters:
            split = "test" if counts["test"] < n_test else "val" if counts["val"] < n_val else "train"
            if split in counts:
                counts[split] += len(cluster)
            for p in cluster:
                out[p] = split
    return out


def pick_eval_copies(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per utterance, preferring the head microphone."""
    order = df.assign(_pref=(df["mic"] != "headMic").astype(int)).sort_values(["utt_key", "_pref", "row_id"])
    return order.drop_duplicates("utt_key").drop(columns="_pref").sort_values("row_id").reset_index(drop=True)


def usable(meta: pd.DataFrame) -> pd.DataFrame:
    """Drop clips longer than Whisper's 30-second window."""
    return meta[meta["duration"] <= MAX_SECONDS]


def _speakers(fold: str) -> list:
    return FOLDS[fold]["dys"] + FOLDS[fold]["ctl"]


def fold_splits(meta: pd.DataFrame, prompt_split: dict, heldout: str) -> dict:
    """Splits for the model that never hears the speakers of fold `heldout`."""
    other = "B" if heldout == "A" else "A"
    meta = usable(meta)
    ps = meta["text_norm"].map(prompt_split)
    held, train_spk = meta["speaker"].isin(_speakers(heldout)), meta["speaker"].isin(_speakers(other))
    return {
        "train": meta[train_spk & (ps == "train")].reset_index(drop=True),
        "val": pick_eval_copies(meta[train_spk & (ps == "val")]),
        "test_seen": pick_eval_copies(meta[held & (ps == "train")]),
        "test_strict": pick_eval_copies(meta[held & (ps == "test")]),
        "test_newsent": pick_eval_copies(meta[train_spk & (ps == "test")]),
    }


def random_splits(meta: pd.DataFrame, seed: int = SEED) -> dict:
    """Deliberately leaky 80/10/10 split over rows, ignoring speakers and microphone copies."""
    meta = usable(meta).reset_index(drop=True)
    idx = np.random.default_rng(seed).permutation(len(meta))
    n_test, n_val = round(0.1 * len(meta)), round(0.1 * len(meta))
    take = lambda ix: meta.iloc[np.sort(ix)].reset_index(drop=True)  # noqa: E731
    return {"train": take(idx[n_test + n_val:]), "val": take(idx[n_test:n_test + n_val]), "test": take(idx[:n_test])}


def personal_subsets(meta: pd.DataFrame, prompt_split: dict, speaker: str,
                     minutes=(2, 5, 10), seed: int = SEED) -> dict:
    """Nested sets of one speaker's own training-prompt clips reaching k minutes, or 'capped' if too few."""
    meta = usable(meta)
    mine = meta[(meta["speaker"] == speaker) & (meta["text_norm"].map(prompt_split) == "train")]
    pool = pick_eval_copies(mine)
    pool = pool.iloc[np.random.default_rng(seed).permutation(len(pool))].reset_index(drop=True)
    cum = pool["duration"].cumsum().to_numpy()
    out = {}
    for k in minutes:
        if len(cum) == 0 or cum[-1] < k * 60:
            out[k] = "capped"
        else:
            out[k] = pool.iloc[: int(np.searchsorted(cum, k * 60)) + 1].copy()
    return out


def write_splits(splits: dict, out_dir, name: str) -> None:
    """Write each part as `{name}_{part}.csv` with row indices and metadata only (no transcripts)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for part, df in splits.items():
        if isinstance(df, pd.DataFrame):
            df[SPLIT_COLUMNS].to_csv(out_dir / f"{name}_{part}.csv", index=False)


def read_split(out_dir, name: str, part: str) -> pd.DataFrame:
    return pd.read_csv(Path(out_dir) / f"{name}_{part}.csv")
