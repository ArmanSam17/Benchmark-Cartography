"""Figure for the usage scenarios (Section 6): the inspector's P(key)-vs-accuracy strip plot
for four questions that IRT scores alike but that the tool's evidence tells apart.

Run from the repo root:  python report/fig_scenarios.py
Output: report/fig_scenarios.pdf
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

q = pd.read_parquet("data/processed/questions.parquet")
s = pd.read_parquet("data/processed/scores.parquet")
ms = pd.read_csv("data/processed/model_scores.csv")
P = np.load("data/processed/probs.npz", allow_pickle=True)["probs"]
d = q.merge(s, on="qid")
acc, top = ms.macro_acc.values, ms.top_tier.values
picks = P.argmax(axis=2)

PANELS = [
    ("college_biology/77", "Hard, key right (reference)"),
    ("virology/93", "Wrong key (Redux)"),
    ("high_school_physics/83", "Trap: key right, consensus wrong"),
    ("miscellaneous/136", "Hard, key right: models split"),
]
GRAY, ACCENT = "#9aa0a6", "#c2571a"

fig, axes = plt.subplots(1, 4, figsize=(11, 2.9), sharey=True)
for ax, (qid, title) in zip(axes, PANELS):
    j = int(np.where(d.qid == qid)[0][0])
    r = d.iloc[j]
    pk = P[:, j, r.gold]
    ax.scatter(acc[~top], pk[~top], s=10, color=GRAY, alpha=0.7, lw=0, label="other 120 models")
    ax.scatter(acc[top], pk[top], s=14, color=ACCENT, lw=0, label="top 30 models")
    tv = np.bincount(picks[top, j], minlength=4) / top.sum()
    alt = int(r.consensus_alt)
    ax.set_title(f"{qid}\n{title}", fontsize=8.5)
    ax.text(0.03, 0.97,
            f"2PL: a = {r.irt_a:.2f}, d = {r.irt_d:.2f}\n"
            f"top tier on key: {tv[r.gold]:.0%}\n"
            f"top tier on {'ABCD'[alt]}: {tv[alt]:.0%}",
            transform=ax.transAxes, va="top", fontsize=7.5,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
    ax.set_xlabel("model macro accuracy", fontsize=8)
    ax.set_ylim(-0.03, 1.03)
    ax.tick_params(labelsize=7)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[0].set_ylabel("P(official key)", fontsize=8)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", ncol=2, fontsize=8, frameon=False)
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig("report/fig_scenarios.pdf")
print("saved report/fig_scenarios.pdf")
