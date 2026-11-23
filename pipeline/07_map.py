"""Draw the first static data map, with MMLU-Redux labels overlaid.

Output: figures/data_map.png. One panel per label group on identical axes, so the
regions can be compared directly and highlighted points never hide each other.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

INK, MUTED, GRID, CONTEXT, HIGHLIGHT, SURFACE = "#0b0b0b", "#52514e", "#e7e6e1", "#c9c8c2", "#2a78d6", "#fcfcfb"
OTHER = ["bad_question_clarity", "multiple_correct_answers", "no_correct_answer", "expert", "bad_options_clarity"]


def main():
    q = pd.read_parquet("data/processed/questions.parquet").merge(
        pd.read_parquet("data/processed/scores.parquet"), on="qid")
    n_models = len(np.load("data/processed/probs.npz")["models"])
    panels = [
        ("Reviewed and found fine", q[q.redux_error == "ok"]),
        ("Other flagged errors", q[q.redux_error.isin(OTHER)]),
        ("Wrong answer key", q[q.redux_error == "wrong_groundtruth"]),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), sharex=True, sharey=True, facecolor=SURFACE)
    for ax, (title, sub) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        ax.scatter(q.confidence, q.variability, s=3, c=CONTEXT, alpha=0.35, linewidths=0, rasterized=True)
        size, alpha = (4, 0.45) if len(sub) > 1000 else (20, 0.9)
        ax.scatter(sub.confidence, sub.variability, s=size, c=HIGHLIGHT, alpha=alpha, linewidths=0.4 if size > 4 else 0,
                   edgecolors=SURFACE)
        ax.set_title(f"{title} (n = {len(sub):,})", loc="left", fontsize=11, color=INK)
        ax.set_xlabel("Confidence in the official answer", color=MUTED, fontsize=10)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, None)
        ax.grid(color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, labelsize=9, length=0)
        for side in ax.spines.values():
            side.set_visible(False)
    axes[0].set_ylabel("Variability across models", color=MUTED, fontsize=10)
    fig.suptitle(f"Wrong answer keys concentrate at low confidence, alongside many fine but hard questions "
                 f"({n_models} models)", x=0.01, ha="left", fontsize=13, color=INK, fontweight="bold")
    fig.text(0.01, 0.005, f"Blue: MMLU-Redux review labels. Grey: all {len(q):,} MMLU questions.",
             color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig("figures/data_map.png", dpi=170, facecolor=SURFACE)
    print("wrote figures/data_map.png")


if __name__ == "__main__":
    main()
