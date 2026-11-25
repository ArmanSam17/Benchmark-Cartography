"""Numbers and figures for the final report that the two analysis scripts do not produce.

RQ1   separation of MMLU-Redux error types on the data map
RQ2   AUPRC per error type, precision@k, bootstrap CI, and the "beyond IRT" checks:
      correlations with 2PL, cross-validated models with and without our scores,
      and wrong key vs. ambiguous separation
RQ3   matched and unmatched random baselines (same design as the tool: matched on
      correctness deciles, drawn from Redux-clean questions outside the selection)
Extra candidate wrong keys outside MMLU-Redux

Run from the repo root:  python report/report_numbers.py
Output: results/report_numbers.json, results/candidates_outside_redux.csv, report/fig_*.pdf
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score as ap, roc_auc_score as auc
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

os.makedirs("results", exist_ok=True)
os.makedirs("report", exist_ok=True)
out = {}

q = pd.read_parquet("data/processed/questions.parquet")
s = pd.read_parquet("data/processed/scores.parquet")
z = np.load("data/processed/probs.npz", allow_pickle=True)
probs = z["probs"]
d = q.merge(s, on="qid")
assert (d.qid.values == q.qid.values).all()
picks = probs.argmax(axis=2)
gold = d.gold.values
n_m, n_q = picks.shape

lab = d[d.redux_error.notna()].copy()
wk = (lab.redux_error == "wrong_groundtruth").values
anyerr = (lab.redux_error != "ok").values
AMBIG = ["bad_question_clarity", "multiple_correct_answers", "bad_options_clarity"]

# ---------------------------------------------------------------- RQ1: the map
rq1 = {}
ok = lab[lab.redux_error == "ok"]
for t, g in lab.groupby("redux_error"):
    row = {"n": len(g),
           "median_top_confidence": round(g.top_confidence.median(), 3),
           "median_confidence": round(g.confidence.median(), 3),
           "median_variability": round(g.variability.median(), 3),
           "share_top_conf_below_0.2": round((g.top_confidence < 0.2).mean(), 3)}
    if t != "ok":
        y = np.r_[np.ones(len(g)), np.zeros(len(ok))]
        row["auroc_vs_ok_low_top_conf"] = round(auc(y, -np.r_[g.top_confidence, ok.top_confidence]), 3)
    rq1[t] = row
out["rq1"] = rq1

# ---------------------------------------------------------------- RQ2: detection
SC = {
    "Low top-tier confidence": 1 - lab.top_confidence,
    "Low confidence": 1 - lab.confidence,
    "Negative ability alignment": -lab.ability_alignment,
    "Low accuracy (majority vote)": 1 - lab.correctness,
    "High 2PL difficulty": lab.irt_d,
    "Low 2PL discrimination": -lab.irt_a,
}
per_type = {}
for t in ["wrong_groundtruth", "bad_question_clarity", "multiple_correct_answers",
          "no_correct_answer", "bad_options_clarity", "expert"]:
    m = (lab.redux_error == t).values
    per_type[t] = {"n": int(m.sum()), "base_rate": round(m.mean(), 4),
                   **{k: round(ap(m, v), 3) for k, v in SC.items()}}
per_type["any_error"] = {"n": int(anyerr.sum()), "base_rate": round(anyerr.mean(), 4),
                         **{k: round(ap(anyerr, v), 3) for k, v in SC.items()}}
out["rq2_auprc_per_type"] = per_type

order = np.argsort(lab.top_confidence.values)
out["rq2_precision_at_k_wrong_key"] = {k: round(wk[order[:k]].mean(), 3) for k in (25, 50, 100, 200)}
out["rq2_recall_at_k_wrong_key"] = {k: round(wk[order[:k]].sum() / wk.sum(), 3) for k in (100, 200, 500)}

rng = np.random.default_rng(20261124)
score = (1 - lab.top_confidence).values
boots = []
for _ in range(1000):
    i = rng.integers(0, len(lab), len(lab))
    if wk[i].any():
        boots.append(ap(wk[i], score[i]))
out["rq2_top_conf_wrong_key_ci95"] = [round(np.percentile(boots, 2.5), 3), round(np.percentile(boots, 97.5), 3)]

# beyond IRT: correlations
out["corr_spearman"] = {
    "top_confidence~irt_d": round(spearmanr(d.top_confidence, d.irt_d)[0], 3),
    "confidence~irt_d": round(spearmanr(d.confidence, d.irt_d)[0], 3),
    "ability_alignment~irt_a": round(spearmanr(d.ability_alignment, d.irt_a, nan_policy="omit")[0], 3),
    "top_confidence~irt_a": round(spearmanr(d.top_confidence, d.irt_a, nan_policy="omit")[0], 3),
}

# beyond IRT: 5-fold CV grouped by subject, out-of-fold AUPRC
X = lab
F_IRT = ["irt_a", "irt_d"]  # 2PL: z = a*theta - d, so d is difficulty (b = d/a is unstable)
F_OURS = ["top_confidence", "confidence", "variability", "ability_alignment",
          "consensus_share", "top_vote_entropy"]
cv = GroupKFold(n_splits=5)


def oof(feats, y):
    p = np.zeros(len(X))
    for tr, te in cv.split(X, y, groups=X.subject):
        mdl = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        mdl.fit(X.iloc[tr][feats], y[tr])
        p[te] = mdl.predict_proba(X.iloc[te][feats])[:, 1]
    return p


cvres = {}
for name, feats in [("2PL only", F_IRT), ("ours only", F_OURS), ("2PL + ours", F_IRT + F_OURS)]:
    cvres[name] = {"wrong key": round(ap(wk, oof(feats, wk)), 3),
                   "any error": round(ap(anyerr, oof(feats, anyerr)), 3)}
out["rq2_cv_logistic_auprc"] = cvres

# wrong key vs ambiguous among low-confidence questions
low = lab[(lab.top_confidence < 0.3) & lab.redux_error.isin(["wrong_groundtruth"] + AMBIG)]
yk = (low.redux_error == "wrong_groundtruth").values
out["wrong_vs_ambiguous"] = {
    "n_wrong": int(yk.sum()), "n_ambiguous": int((~yk).sum()),
    "auroc_consensus_share": round(auc(yk, low.consensus_share), 3),
    "auroc_low_top_entropy": round(auc(yk, -low.top_vote_entropy), 3),
    "auroc_low_2pl_discrimination": round(auc(yk, -low.irt_a), 3),
    "auroc_high_2pl_difficulty": round(auc(yk, low.irt_d), 3),
    "median_consensus_share_wrong": round(low[yk].consensus_share.median(), 3),
    "median_consensus_share_ambiguous": round(low[~yk].consensus_share.median(), 3),
}

# consensus alternative vs Redux fix
fx = lab[(lab.redux_error == "wrong_groundtruth") & (lab.redux_correct_idx >= 0)]
out["consensus_matches_fix"] = {
    "n_with_fix": len(fx),
    "all": round((fx.consensus_alt == fx.redux_correct_idx).mean(), 3),
    "share>=0.5_n": int((fx.consensus_share >= 0.5).sum()),
    "share>=0.5": round((fx[fx.consensus_share >= 0.5].consensus_alt
                         == fx[fx.consensus_share >= 0.5].redux_correct_idx).mean(), 3),
}

# ---------------------------------------------------------------- RQ3: baselines (tool design)
subj_codes, subj = np.unique(d.subject.values, return_inverse=True)
S = np.zeros((n_q, len(subj_codes)))
S[np.arange(n_q), subj] = 1
correct = (picks == gold[None, :]).astype(float)


def board(corr, keep):
    k = keep.astype(float)
    return ((corr * k) @ S / (k @ S)).mean(axis=1)


def swaps(a, b):
    x = a[:, None] - a[None, :]
    y = b[:, None] - b[None, :]
    return int(((x * y) < 0).sum() // 2)


def ranks(v):
    return (-v).argsort().argsort() + 1


base = board(correct, np.ones(n_q, bool))
diff_bin = np.minimum((d.correctness.values * 10).astype(int), 9)
clean_all = (d.redux_error.isna() | (d.redux_error == "ok")).values
clean_redux = (d.redux_error == "ok").values
DRAWS = 1000


def null(sel, rekey=None, matched=True):
    pool_ok = clean_all & ~sel
    need = np.bincount(diff_bin[sel], minlength=10)
    pools = [np.where(pool_ok & (diff_bin == b))[0] for b in range(10)]
    res = []
    for _ in range(DRAWS):
        if matched:
            pick = np.concatenate([rng.choice(pools[b], min(k, len(pools[b])), replace=False)
                                   for b, k in enumerate(need) if k])
        else:
            pick = rng.choice(np.where(~sel)[0], sel.sum(), replace=False)
        if rekey is None:
            keep = np.ones(n_q, bool); keep[pick] = False
            res.append(swaps(base, board(correct, keep)))
        else:
            c = correct.copy()
            c[:, pick] = (picks[:, pick] == d.consensus_alt.values[pick][None, :])
            res.append(swaps(base, board(c, np.ones(n_q, bool))))
    return np.array(res)


def pval(nulls, obs):
    return round((1 + (nulls >= obs).sum()) / (len(nulls) + 1), 4)


wk_all = (d.redux_error == "wrong_groundtruth").values
anyerr_all = (d.redux_error.notna() & (d.redux_error != "ok")).values
rq3 = {}
saved = {}
for name, sel, mode in [("remove 106 wrong keys", wk_all, "remove"),
                        ("remove 370 Redux-flagged", anyerr_all, "remove"),
                        ("re-key 106 wrong keys to consensus", wk_all, "consensus")]:
    if mode == "remove":
        keep = ~sel
        new = board(correct, keep)
        nm, nu = null(sel), null(sel, matched=False)
    else:
        c = correct.copy()
        idx = np.where(sel)[0]
        c[:, idx] = (picks[:, idx] == d.consensus_alt.values[idx][None, :])
        new = board(c, np.ones(n_q, bool))
        nm, nu = null(sel, rekey=True), null(sel, rekey=True, matched=False)
    sw = swaps(base, new)
    npairs = n_m * (n_m - 1) // 2
    r0, r1 = ranks(base), ranks(new)
    top0, top1 = set(np.where(r0 <= 10)[0]), set(np.where(r1 <= 10)[0])
    rq3[name] = {"questions": int(sel.sum()), "pair_swaps": sw, "kendall_tau": round(1 - 2 * sw / npairs, 4),
                 "models_moved": int((r0 != r1).sum()), "max_move": int(np.abs(r0 - r1).max()),
                 "new_in_top10": len(top1 - top0),
                 "matched_median": float(np.median(nm)), "matched_p": pval(nm, sw),
                 "unmatched_median": float(np.median(nu)), "unmatched_p": pval(nu, sw)}
    saved[name] = (sw, nm, nu)

# Redux fix re-key (only questions where the fix differs from the key)
fix_all = d[(d.redux_error == "wrong_groundtruth") & (d.redux_correct_idx >= 0) & (d.redux_correct_idx != d.gold)]
c = correct.copy()
idx = fix_all.index.values
c[:, idx] = (picks[:, idx] == fix_all.redux_correct_idx.values[None, :])
new = board(c, np.ones(n_q, bool))
sw = swaps(base, new)
r0, r1 = ranks(base), ranks(new)
rq3["re-key to Redux fix"] = {"questions": len(idx), "pair_swaps": sw,
                              "kendall_tau": round(1 - 2 * sw / (n_m * (n_m - 1) // 2), 4),
                              "models_moved": int((r0 != r1).sum()), "max_move": int(np.abs(r0 - r1).max())}
out["rq3"] = rq3
out["rq3_draws"] = DRAWS
out["accuracy_range"] = [round(base.min(), 3), round(base.max(), 3)]
gaps = np.diff(np.sort(base))
out["median_gap_adjacent_models"] = round(float(np.median(gaps)), 5)

# ---------------------------------------------------------------- candidates outside Redux
unc = d[d.redux_error.isna()]
cand = unc[(unc.top_confidence < 0.2) & (unc.consensus_share >= 0.8)].sort_values("top_confidence")
out["candidates_outside_redux"] = {"unchecked": len(unc), "flagged_strict": len(cand),
                                   "flagged_top_conf<0.2": int((unc.top_confidence < 0.2).sum())}
cand[["qid", "subject", "question", "gold", "consensus_alt", "top_confidence", "consensus_share"]] \
    .to_csv("results/candidates_outside_redux.csv", index=False)

with open("results/report_numbers.json", "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))

# ---------------------------------------------------------------- figures
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#8f8e88", "axes.labelcolor": "#52514e",
                     "xtick.color": "#52514e", "ytick.color": "#52514e", "font.family": "sans-serif"})
BLUE, ORANGE, GRAY, LIGHT = "#2a78d6", "#eb6834", "#8f8e88", "#c9c8c2"

# Fig 1: data map, Redux-checked questions only
fig, ax = plt.subplots(figsize=(3.4, 2.7))
clean = lab[lab.redux_error == "ok"]
other = lab[(lab.redux_error != "ok") & (lab.redux_error != "wrong_groundtruth")]
wrong = lab[lab.redux_error == "wrong_groundtruth"]
ax.scatter(clean.confidence, clean.variability, s=2, c=LIGHT, lw=0, label=f"fine ({len(clean):,})", rasterized=True)
ax.scatter(other.confidence, other.variability, s=5, c=BLUE, lw=0, label=f"other error ({len(other)})", rasterized=True)
ax.scatter(wrong.confidence, wrong.variability, s=9, c=ORANGE, lw=0.3, edgecolors="white", label=f"wrong key ({len(wrong)})")
ax.set_xlabel("Confidence: mean P(key) over 150 models")
ax.set_ylabel("Variability: std of P(key)")
fig.legend(*ax.get_legend_handles_labels(), frameon=False, loc="upper center", ncol=3, markerscale=2,
           fontsize=7, handletextpad=0.1, columnspacing=0.8, title="MMLU-Redux label", title_fontsize=7)
fig.tight_layout(rect=(0, 0, 1, 0.86)); fig.savefig("report/fig_map.pdf", dpi=300); plt.close(fig)

# Fig 2: AUPRC for wrong key and any error
names = sorted(SC, key=lambda n: per_type["wrong_groundtruth"][n])  # best at top
fig, ax = plt.subplots(figsize=(3.4, 2.5))
yv = np.arange(len(names))
ax.barh(yv + 0.2, [per_type["wrong_groundtruth"][n] for n in names], 0.38, color=ORANGE, label="Wrong key (n=106)")
ax.barh(yv - 0.2, [per_type["any_error"][n] for n in names], 0.38, color=BLUE, label="Any error (n=370)")
ax.axvline(per_type["wrong_groundtruth"]["base_rate"], color=ORANGE, ls=":", lw=1)
ax.axvline(per_type["any_error"]["base_rate"], color=BLUE, ls=":", lw=1)
ax.set_yticks(yv, names); ax.set_xlabel("AUPRC (dotted lines: random)")
fig.legend(*ax.get_legend_handles_labels(), frameon=False, fontsize=7, loc="upper center", ncol=2)
fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig("report/fig_auprc.pdf"); plt.close(fig)

# Fig 3: RQ3 null distributions
fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.1), sharey=False)
for ax, name in zip(axes, ["remove 106 wrong keys", "re-key 106 wrong keys to consensus"]):
    sw, nm, nu = saved[name]
    hi = max(nm.max(), nu.max(), sw) + 2
    bins = np.arange(0, hi + 1, max(1, hi // 30))
    ax.hist(nu, bins=bins, color=LIGHT, label="random, unmatched")
    ax.hist(nm, bins=bins, color=GRAY, alpha=0.85, label="random, difficulty-matched")
    ax.axvline(sw, color=ORANGE, lw=2, label="Redux wrong keys (observed)")
    ax.text(sw, ax.get_ylim()[1] * 0.95, f" {sw}", color=ORANGE, va="top", fontsize=8)
    ax.set_title(name.capitalize(), fontsize=8, loc="left")
    ax.set_xlabel("Model pairs that swap rank"); ax.set_yticks([])
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, frameon=False, fontsize=7, loc="upper center", ncol=3)
fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig("report/fig_rq3.pdf"); plt.close(fig)
print("figures written")
