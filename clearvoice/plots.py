"""Figures for the write-up and demo. Colour follows the system, the same in every chart."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # fixed categorical order
SYSTEM_COLORS = {"Clear Voice": SERIES[0], "whisper-small": SERIES[1], "whisper-large-v3": SERIES[2]}
SEVERITY_ORDER = ["severe", "moderate-severe", "moderate", "mild", "typical"]


def _style(ax, title: str, ylabel: str):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold")
    ax.set_ylabel(ylabel, color=INK_2)
    ax.tick_params(colors=INK_2, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))


def _bar_axis(ax, vals):
    """Bars start at zero; leave headroom for value labels."""
    top = np.nanmax(vals) if np.isfinite(vals).any() else 0
    ax.set_ylim(0, max(0.05, top * 1.15))


def _label_bars(ax, x, vals):
    for xi, v in zip(x, vals):
        if np.isfinite(v):
            ax.annotate(f"{v:.0%}", (xi, v), xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", color=INK, fontsize=9)


def _color(system: str) -> str:
    return SYSTEM_COLORS.get(system, SERIES[3])


def _save(fig, path, top: float = 1.0):
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout(rect=(0, 0, 1, top))
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def bar_systems(df: pd.DataFrame, path) -> None:
    """Headline: strict-split error per system, sentences (WER) and single words (CER), with 95% intervals."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, (col, title) in zip(axes, [("sentence_wer", "Sentences: word error rate"),
                                       ("word_cer", "Single words: character error rate")]):
        x = np.arange(len(df))
        vals = df[col].to_numpy(dtype=float)
        lo, hi = df.get(f"{col}_lo"), df.get(f"{col}_hi")
        ax.bar(x, vals, width=0.6, color=[_color(s) for s in df["system"]], edgecolor=SURFACE, linewidth=2)
        if lo is not None and hi is not None and not np.isnan(lo.to_numpy(dtype=float)).all():
            err = np.vstack([vals - lo.to_numpy(dtype=float), hi.to_numpy(dtype=float) - vals])
            ax.errorbar(x, vals, yerr=np.nan_to_num(err), fmt="none", ecolor=INK_2, elinewidth=1, capsize=3)
            vals = np.fmax(vals, hi.to_numpy(dtype=float))
        _label_bars(ax, x, df[col].to_numpy(dtype=float))
        ax.set_xticks(x, df["system"], color=INK_2)
        _style(ax, title, "Error rate (lower is better)")
        _bar_axis(ax, vals)
    fig.suptitle("Dysarthric speakers the model never heard, saying sentences it never heard (95% intervals)",
                 x=0.01, ha="left", color=INK_2, fontsize=9)
    _save(fig, path, top=0.93)


def leakage(df: pd.DataFrame, path) -> None:
    """The same fine-tuned model scored on easier and harder tests."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, (col, title) in zip(axes, [("sentence_wer", "Sentences: word error rate"),
                                       ("word_cer", "Single words: character error rate")]):
        x = np.arange(len(df))
        vals = df[col].to_numpy(dtype=float)
        ax.bar(x, vals, width=0.6, color=SYSTEM_COLORS["Clear Voice"], edgecolor=SURFACE, linewidth=2)
        _label_bars(ax, x, vals)
        ax.set_xticks(x, [c.replace(", ", ",\n") for c in df["condition"]], color=INK_2, fontsize=8)
        _style(ax, title, "Error rate (lower is better)")
        _bar_axis(ax, vals)
    _save(fig, path)


def per_speaker(df: pd.DataFrame, path, metric: str = "word_cer") -> None:
    """Strict-split error for each speaker, grouped by system, speakers ordered by severity."""
    order = sorted(df["speaker"].unique(),
                   key=lambda s: (SEVERITY_ORDER.index(df.loc[df.speaker == s, "severity"].iloc[0]), s))
    systems = list(dict.fromkeys(df["system"]))
    fig, ax = plt.subplots(figsize=(max(7, 0.9 * len(order) + 2), 3.8))
    width = 0.8 / len(systems)
    x = np.arange(len(order))
    for i, system in enumerate(systems):
        vals = df[df.system == system].set_index("speaker").reindex(order)[metric].to_numpy(dtype=float)
        ax.bar(x + (i - (len(systems) - 1) / 2) * width, vals, width=width, color=_color(system),
               edgecolor=SURFACE, linewidth=2, label=system)
    sev = df.drop_duplicates("speaker").set_index("speaker")["severity"]
    ax.set_xticks(x, [f"{s}\n{sev[s]}" for s in order], color=INK_2, fontsize=8)
    label = "Character error rate" if metric == "word_cer" else "Word error rate"
    _style(ax, f"{label} per speaker (unseen speakers, unseen sentences)", "Error rate (lower is better)")
    _bar_axis(ax, df[metric].to_numpy(dtype=float))
    ax.legend(frameon=False, labelcolor=INK_2, fontsize=8, loc="upper right")
    _save(fig, path)


def personalisation(df: pd.DataFrame, path, metric: str = "cer") -> None:
    """Error on a speaker's unseen sentences as minutes of their own voice are added."""
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for i, (spk, g) in enumerate(df.groupby("speaker", sort=False)):
        g = g.sort_values("minutes")
        color = SERIES[i % len(SERIES)]
        ax.plot(g["minutes"], g[metric], color=color, linewidth=2, marker="o", markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2, label=f"{spk} ({g['severity'].iloc[0]})")
        last = g.iloc[-1]
        ax.annotate(f"{spk} {last[metric]:.0%}", (last["minutes"], last[metric]), xytext=(6, 0),
                    textcoords="offset points", va="center", color=INK, fontsize=9)
    ax.set_xlabel("Minutes of the speaker's own recordings added", color=INK_2)
    ax.set_xticks(sorted(df["minutes"].unique()))
    label = "Character error rate" if metric == "cer" else "Word error rate"
    _style(ax, f"{label} on new sentences as personal data grows", "Error rate (lower is better)")
    ax.legend(frameon=False, labelcolor=INK_2, fontsize=8)
    _save(fig, path)
