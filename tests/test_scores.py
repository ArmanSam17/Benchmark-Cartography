"""Checks on the scoring code. Run: python tests/test_scores.py"""
import importlib.util
import os

import numpy as np
from scipy.special import expit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("scores", os.path.join(ROOT, "pipeline", "05_scores.py"))
scores = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scores)


def test_2pl_recovers_known_parameters():
    rng = np.random.default_rng(0)
    theta = rng.normal(size=150)
    a = rng.normal(1.2, 0.6, size=2000)
    a[:100] = rng.uniform(-1.5, -0.3, size=100)  # planted bad items
    b = rng.normal(0, 1.2, size=2000)
    y = rng.random((150, 2000)) < expit(a[None, :] * (theta[:, None] - b[None, :]))
    t, a_hat, b_hat, d_hat = scores.fit_2pl(y)
    print(f"corr theta {np.corrcoef(t, theta)[0, 1]:.3f}, a {np.corrcoef(a_hat, a)[0, 1]:.3f}, "
          f"planted items with a<0: {(a_hat[:100] < 0).mean():.2f}")
    assert np.corrcoef(t, theta)[0, 1] > 0.95
    assert np.corrcoef(a_hat, a)[0, 1] > 0.8
    d = a * b
    print(f"corr d: {np.corrcoef(d_hat, d)[0, 1]:.3f}; theta SD: {t.std():.2f}")
    assert np.corrcoef(d_hat, d)[0, 1] > 0.9
    assert abs(t.std() - 1) < 0.01  # the scale must stay pinned
    # Clearly negative items must come back negative; the prior must not hide them.
    strong = a[:100] < -0.7
    print(f"items with a < -0.7 recovered as negative: {(a_hat[:100][strong] < 0).mean():.2f}")
    assert (a_hat[:100][strong] < 0).mean() > 0.9


def test_vote_entropy():
    votes = np.array([[0, 0], [0, 1], [0, 2], [0, 3]])
    ent, shares = scores.entropy_of_votes(votes)
    assert np.isclose(ent[0], 0) and np.isclose(ent[1], 2)
    assert np.allclose(shares.sum(axis=1), 1)


if __name__ == "__main__":
    test_2pl_recovers_known_parameters()
    test_vote_entropy()
    print("all passed")
