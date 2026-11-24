"""RQ2: how well does each score find the broken questions?

We line up all questions from "most suspicious" to "least suspicious" with each score,
then grade the line-up against MMLU-Redux's human labels with AUPRC (higher = the broken
questions are nearer the front). Random shuffling is the floor to beat.

Run from the repo root:  python analysis/01_detection.py
Output: results/detection_auprc.csv, results/consensus_check.csv
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

# 1. Load the cached data ---------------------------------------------------------------
q = pd.read_parquet("data/processed/questions.parquet")   # question text, key, Redux label
s = pd.read_parquet("data/processed/scores.parquet")      # our six scores + IRT, per question
probs = np.load("data/processed/probs.npz")["probs"]       # models x questions x options
d = q.merge(s, on="qid")

# 2. Majority-vote baseline: share of all models whose top pick is NOT the key -----------
picks = probs.argmax(axis=2)                               # each model's pick per question
d["majority_against"] = (picks != d.gold.values[None, :]).mean(axis=0)

# 3. Keep only the questions humans checked (MMLU-Redux) ---------------------------------
lab = d[d.redux_error.notna()].copy()
lab["any_error"] = lab.redux_error != "ok"
lab["wrong_key"] = lab.redux_error == "wrong_groundtruth"
print(f"labelled questions: {len(lab)} | any error: {lab.any_error.sum()} | wrong key: {lab.wrong_key.sum()}")

# 4. Each score, written so that HIGHER = MORE SUSPICIOUS ---------------------------------
rng = np.random.default_rng(0)
scores = {
    "Random": rng.random(len(lab)),
    "Majority vote vs. key": lab.majority_against,
    "Low accuracy": 1 - lab.correctness,
    "IRT low discrimination": -lab.irt_a,
    "Low confidence": 1 - lab.confidence,
    "Negative ability alignment": -lab.ability_alignment,
    "Low top-tier confidence": 1 - lab.top_confidence,
}

# 5. Grade every line-up with AUPRC --------------------------------------------------------
rows = []
for name, score in scores.items():
    rows.append({
        "score": name,
        "AUPRC any error": average_precision_score(lab.any_error, score),
        "AUPRC wrong key": average_precision_score(lab.wrong_key, score),
    })
table = pd.DataFrame(rows).sort_values("AUPRC wrong key").round(3)
base_any, base_wk = lab.any_error.mean(), lab.wrong_key.mean()
print(f"\nbase rates (= what random should score): any error {base_any:.3f}, wrong key {base_wk:.3f}\n")
print(table.to_string(index=False))

best = table.iloc[-1]
print(f"\nbest score: {best.score}, {best['AUPRC wrong key'] / base_wk:.1f}x the wrong-key base rate")

# 6. Consensus check: when the top models agree on another option, is it Redux's fix? ----
wk = lab[lab.wrong_key & (lab.redux_correct_idx >= 0)]
strong = wk[wk.consensus_share >= 0.5]
hits = (strong.consensus_alt == strong.redux_correct_idx).sum()
print(f"\nconsensus check: {len(wk)} wrong keys with a single-option fix; top tier agrees "
      f"(>=50%) on {len(strong)}; matches Redux's fix in {hits} ({hits / max(len(strong), 1):.0%})")

os.makedirs("results", exist_ok=True)
table.to_csv("results/detection_auprc.csv", index=False)
pd.DataFrame([{"wrong_keys_with_fix": len(wk), "top_tier_agrees": len(strong), "matches_redux": hits}]
             ).to_csv("results/consensus_check.csv", index=False)
print("\nsaved results/detection_auprc.csv and results/consensus_check.csv")
