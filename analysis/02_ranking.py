"""RQ3: does fixing or removing broken questions reorder the leaderboard?

For each action (remove flagged questions, or correct wrong keys) we re-score all 150 models,
re-rank them, and count how many PAIRS of models swap order. Then we ask whether that is
more than chance, by removing the same number of random questions of SIMILAR DIFFICULTY
1,000 times (the professor's feedback point 3).

Run from the repo root:  python analysis/02_ranking.py
Output: results/ranking_impact.csv, results/rank_changes.csv
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import kendalltau

REPS = 1000

# 1. Load ------------------------------------------------------------------------------
q = pd.read_parquet("data/processed/questions.parquet")
s = pd.read_parquet("data/processed/scores.parquet")
z = np.load("data/processed/probs.npz", allow_pickle=True)
probs, models = z["probs"], list(z["models"])
d = q.merge(s, on="qid")
assert (d.qid.values == q.qid.values).all()
picks = probs.argmax(axis=2)                      # models x questions
gold = d.gold.values
subj = d.subject.values
n_m, n_q = picks.shape


def leaderboard(correct, keep):
    """Macro accuracy (average over subjects), as leaderboards report it, on the kept questions."""
    df = pd.DataFrame(correct[:, keep].T, index=subj[keep])
    return df.groupby(level=0).mean().mean(axis=0).values


def pair_swaps(old, new):
    """How many pairs of models are in a different order after the change."""
    a = old[:, None] - old[None, :]
    b = new[:, None] - new[None, :]
    return int(((a * b) < 0).sum() // 2)


all_q = np.ones(n_q, bool)
correct = picks == gold[None, :]
base = leaderboard(correct, all_q)
print(f"{n_m} models -> {n_m * (n_m - 1) // 2:,} model pairs; accuracy {base.min():.3f} to {base.max():.3f}")

# 2. Difficulty bins for the fair random comparison -------------------------------------
bins = pd.qcut(d.correctness.values, 10, labels=False, duplicates="drop")
pools = [np.where(bins == b)[0] for b in range(bins.max() + 1)]
rng = np.random.default_rng(0)


def matched_random(remove_mask):
    """Remove random questions with the same difficulty mix as remove_mask, REPS times."""
    need = np.bincount(bins[remove_mask], minlength=len(pools))
    out = []
    for _ in range(REPS):
        keep = all_q.copy()
        for b, k in enumerate(need):
            if k:
                keep[rng.choice(pools[b], k, replace=False)] = False
        out.append(pair_swaps(base, leaderboard(correct, keep)))
    return np.array(out)


def plain_random(k):
    out = []
    for _ in range(REPS):
        keep = all_q.copy()
        keep[rng.choice(n_q, k, replace=False)] = False
        out.append(pair_swaps(base, leaderboard(correct, keep)))
    return np.array(out)


# 3. The actions ----------------------------------------------------------------------
redux_bad = (d.redux_error.notna() & (d.redux_error != "ok")).values
K = int(redux_bad.sum())
ours = np.zeros(n_q, bool)
ours[np.argsort(d.top_confidence.values)[:K]] = True          # our K most suspicious questions

rows = []
for name, rm in [("Remove Redux-flawed questions", redux_bad),
                 (f"Remove our top {K} flags", ours)]:
    new = leaderboard(correct, ~rm)
    sw = pair_swaps(base, new)
    null_m, null_p = matched_random(rm), plain_random(int(rm.sum()))
    rows.append({"action": name, "questions": int(rm.sum()), "pair swaps": sw,
                 "kendall tau": round(kendalltau(base, new)[0], 4),
                 "matched random: mean swaps": round(null_m.mean(), 1),
                 "p (matched)": round((1 + (null_m >= sw).sum()) / (REPS + 1), 3),
                 "plain random: mean swaps": round(null_p.mean(), 1)})
    print(f"done: {name}")

# Correcting (not removing) the wrong keys Redux fixed
fix = d[(d.redux_error == "wrong_groundtruth") & (d.redux_correct_idx >= 0)]
correct_fixed = correct.copy()
for j, g in zip(fix.index, fix.redux_correct_idx.astype(int)):
    correct_fixed[:, j] = picks[:, j] == g
fixed = leaderboard(correct_fixed, all_q)
rows.append({"action": f"Correct {len(fix)} Redux wrong keys", "questions": len(fix),
             "pair swaps": pair_swaps(base, fixed), "kendall tau": round(kendalltau(base, fixed)[0], 4)})

table = pd.DataFrame(rows)
print("\n" + table.to_string(index=False))

# 4. Who moved when the wrong keys are corrected? --------------------------------------
r_old = (-base).argsort().argsort() + 1
r_new = (-fixed).argsort().argsort() + 1
moves = pd.DataFrame({"model": models, "acc_before": base.round(4), "acc_after": fixed.round(4),
                      "rank_before": r_old, "rank_after": r_new})
moves["rank_change"] = moves.rank_before - moves.rank_after       # + = moved up
moves = moves.sort_values("rank_before")
moved = moves[moves.rank_change != 0]
print(f"\nmodels whose rank changed after correcting wrong keys: {len(moved)} of {n_m}")
print(f"biggest move: {moved.rank_change.abs().max()} places")
print(f"top-10 changed: {set(moves.head(10).model) != set(moves.sort_values('rank_after').head(10).model)}")
print(moved.reindex(moved.rank_change.abs().sort_values(ascending=False).index).head(5).to_string(index=False))

os.makedirs("results", exist_ok=True)
table.to_csv("results/ranking_impact.csv", index=False)
moves.to_csv("results/rank_changes.csv", index=False)
print("\nsaved results/ranking_impact.csv and results/rank_changes.csv")
