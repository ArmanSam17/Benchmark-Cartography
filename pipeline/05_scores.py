"""Compute per-question cartography scores and fit a 2PL IRT baseline.

Input:  data/processed/probs.npz, questions.parquet
Output: data/processed/scores.parquet, data/processed/model_scores.csv

The 2PL is a maximum a posteriori fit of P(correct) = sigmoid(a_q * theta_m - d_q).
Discrimination a_q may go negative, since that is the signal of a suspect item.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

TOP_SHARE = 0.20


def entropy_of_votes(votes, n_options=4):
    """Entropy in bits of the option each model picks. Votes: models x questions."""
    counts = np.stack([(votes == k).sum(axis=0) for k in range(n_options)], axis=1)
    shares = counts / votes.shape[0]
    with np.errstate(divide="ignore", invalid="ignore"):
        logs = np.where(shares > 0, np.log2(shares), 0.0)
    return -(shares * logs).sum(axis=1), shares


def _fit_items(y, theta, a_sd, d_sd, start, maxiter=300):
    """Update all (a, d) given theta. Convex, and independent across items."""
    n_q = y.shape[1]

    def objective(x):
        a, d = x[:n_q], x[n_q:]
        z = theta[:, None] * a[None, :] - d[None, :]
        resid = expit(z) - y
        value = np.sum(np.logaddexp(0, z) - y * z) + (a ** 2).sum() / (2 * a_sd ** 2) + (d ** 2).sum() / (2 * d_sd ** 2)
        return value, np.concatenate([(resid * theta[:, None]).sum(axis=0) + a / a_sd ** 2,
                                      -resid.sum(axis=0) + d / d_sd ** 2])

    x = minimize(objective, start, jac=True, method="L-BFGS-B", options=dict(maxiter=maxiter)).x
    return x[:n_q], x[n_q:]


def _fit_abilities(y, a, d, start, maxiter=100):
    """Update theta given items. Convex, and independent across models."""

    def objective(theta):
        z = theta[:, None] * a[None, :] - d[None, :]
        return np.sum(np.logaddexp(0, z) - y * z), ((expit(z) - y) * a[None, :]).sum(axis=1)

    return minimize(objective, start, jac=True, method="L-BFGS-B", options=dict(maxiter=maxiter)).x


def fit_2pl(correct, a_sd=2.0, d_sd=4.0, rounds=5):
    """MAP 2PL by alternating updates. correct: models x questions of 0/1.

    Returns theta, a, b, d for z = a * theta - d. Three choices matter:
    - Slope-intercept form: z = a * (theta - b) gets stuck at a > 0 with extreme b, which
      hides the negative-discrimination items a baseline must catch.
    - theta is rescaled to mean 0, SD 1 each round: otherwise the prior on thousands of
      items pushes a toward zero and theta toward large values (we saw theta = 28).
    - The prior on a is centred on zero, so IRT is not handicapped as a baseline.
    """
    y = correct.astype(np.float64)
    n_q = y.shape[1]
    standardise = lambda t: (t - t.mean()) / max(t.std(), 1e-6)
    theta = standardise(y.mean(axis=1))
    x = np.zeros(2 * n_q)
    for _ in range(rounds):
        a, d = _fit_items(y, theta, a_sd, d_sd, x)
        x = np.concatenate([a, d])
        theta = standardise(_fit_abilities(y, a, d, theta))
    a, d = _fit_items(y, theta, a_sd, d_sd, x)
    # d is the stable difficulty measure; b = d / a is unbounded as a approaches zero.
    b = np.divide(d, a, out=np.full(n_q, np.nan), where=np.abs(a) > 1e-6)
    return theta, a, b, d


def main():
    data = np.load("data/processed/probs.npz")
    probs, models = data["probs"].astype(np.float64), data["models"]
    q = pd.read_parquet("data/processed/questions.parquet")
    n_m, n_q, _ = probs.shape
    gold = q.gold.values
    idx = np.arange(n_q)

    p_gold = probs[:, idx, gold]  # models x questions
    votes = probs.argmax(axis=2)
    correct = votes == gold[None, :]

    # Overall score: mean per-subject accuracy, as leaderboards report it.
    subject_acc = pd.DataFrame(correct.T, index=q.subject).groupby(level=0).mean()
    ability = subject_acc.mean(axis=0).values
    n_top = max(1, int(round(TOP_SHARE * n_m)))
    top = np.argsort(-ability)[:n_top]

    ent, _ = entropy_of_votes(votes)
    top_ent, top_shares = entropy_of_votes(votes[top])
    alt_shares = top_shares.copy()
    alt_shares[idx, gold] = -1
    consensus = alt_shares.argmax(axis=1)

    a_c = ability - ability.mean()
    p_c = p_gold - p_gold.mean(axis=0, keepdims=True)
    denom = np.sqrt((a_c ** 2).sum()) * np.sqrt((p_c ** 2).sum(axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        alignment = np.where(denom > 0, (a_c[:, None] * p_c).sum(axis=0) / denom, 0.0)

    theta, disc, diff, intercept = fit_2pl(correct)

    scores = pd.DataFrame({
        "qid": q.qid,
        "confidence": p_gold.mean(axis=0),
        "variability": p_gold.std(axis=0),
        "correctness": correct.mean(axis=0),
        "ability_alignment": alignment,
        "top_confidence": p_gold[top].mean(axis=0),
        "top_correctness": correct[top].mean(axis=0),
        "consensus_alt": consensus,
        "consensus_share": top_shares[idx, consensus],
        "vote_entropy": ent,
        "top_vote_entropy": top_ent,
        "irt_a": disc,
        "irt_b": diff,
        "irt_d": intercept,
    })
    scores.to_parquet("data/processed/scores.parquet", index=False)
    # Models named after MMLU may have trained on it; results get reported with and without them.
    pd.DataFrame({"model": models, "macro_acc": ability, "theta": theta,
                  "top_tier": np.isin(np.arange(n_m), top),
                  "name_mentions_mmlu": ["mmlu" in str(m).lower() for m in models]}).to_csv("data/processed/model_scores.csv", index=False)

    print(f"models {n_m}, questions {n_q}, top tier {n_top}")
    print(f"correlation of theta with macro accuracy: {np.corrcoef(theta, ability)[0, 1]:.3f}")
    print(scores.drop(columns=["qid", "consensus_alt"]).describe().round(3).loc[["mean", "std", "min", "max"]].T)


if __name__ == "__main__":
    main()
