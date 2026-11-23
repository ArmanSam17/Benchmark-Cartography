"""Combine model outputs into one question table and probability array, joined to MMLU-Redux.

Output:
  data/processed/questions.parquet   one row per question, with Redux labels
  data/processed/probs.npz           probs[model, question, option] and model names
  data/processed/join_report.json    how the join went, against the 98% target

The two sources share no ID, so the join uses subject plus normalised text in three
tiers: question and options, then question only, then fuzzy match within the subject.
"""
import difflib
import glob
import json
import os
import re

import numpy as np
import pandas as pd
import pyarrow.ipc as ipc
from scipy.special import softmax

DETAILS = "data/raw/details"
REDUX = "data/raw/redux"


def norm(text):
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def option_index(value):
    """Redux corrections mix digits, letters and free text; return 0-3 only for a single option."""
    v = str(value).strip().upper()
    if v in "ABCD" and len(v) == 1:
        return "ABCD".index(v)
    return int(float(v)) if re.fullmatch(r"[0-3](\.0)?", v) else -1


def split_query(query):
    """Split 'stem\\nA. a\\nB. b\\nC. c\\nD. d\\nAnswer:' into the stem and four options."""
    body = re.sub(r"\s*Answer:\s*$", "", query)
    match = re.match(r"^(.*)\nA\. (.*)\nB\. (.*)\nC\. (.*)\nD\. (.*)$", body, flags=re.S)
    if match is None:
        return body, [None] * 4
    return match.group(1), [match.group(i) for i in range(2, 6)]


def load_redux():
    frames = []
    for path in sorted(glob.glob(f"{REDUX}/*/data-*.arrow")):
        with open(path, "rb") as f:
            t = ipc.open_stream(f).read_all().to_pandas()
        t["subject"] = path.split("/")[-2]
        frames.append(t)
    r = pd.concat(frames, ignore_index=True)
    r["q_norm"] = r.question.map(norm)
    r["full_norm"] = r.q_norm + " || " + r.choices.map(lambda c: " | ".join(norm(x) for x in c))
    return r


def join_redux(q, r):
    q = q.copy()
    q["q_norm"] = q.question.map(norm)
    q["full_norm"] = q.q_norm + " || " + q.options.map(lambda c: " | ".join(norm(x) for x in c))
    q["redux_idx"] = -1
    q["join_tier"] = ""
    used = set()

    def assign(qi, ri, tier):
        q.at[qi, "redux_idx"] = ri
        q.at[qi, "join_tier"] = tier
        used.add(ri)

    full = {}
    for ri, row in r.iterrows():
        full.setdefault((row.subject, row.full_norm), []).append(ri)
    for qi, row in q.iterrows():
        hits = [x for x in full.get((row.subject, row.full_norm), []) if x not in used]
        if hits:
            assign(qi, hits[0], "exact_with_options")

    stem = {}
    for ri, row in r.iterrows():
        if ri not in used:
            stem.setdefault((row.subject, row.q_norm), []).append(ri)
    for qi, row in q[q.redux_idx < 0].iterrows():
        hits = [x for x in stem.get((row.subject, row.q_norm), []) if x not in used]
        if hits:
            assign(qi, hits[0], "exact_question")

    for ri, row in r[~r.index.isin(used)].iterrows():
        pool = q[(q.subject == row.subject) & (q.redux_idx < 0)]
        lookup = dict(zip(pool.q_norm, pool.index))
        close = difflib.get_close_matches(row.q_norm, list(lookup), n=1, cutoff=0.9)
        if close:
            assign(lookup[close[0]], ri, "fuzzy")
    return q


def main():
    models = pd.read_csv("data/processed/models.csv")
    log = pd.read_csv("data/processed/downloads.csv")
    ok = set(log[log.status == "ok"].model)
    names = [m for m in models.model if m in ok]

    keys = pd.read_parquet("data/processed/gold_reference.parquet").rename(columns={"gold": "key"})
    reference, stack, key_conflicts = None, [], 0
    for m in names:
        d = pd.read_parquet(f"{DETAILS}/{m.replace('/', '__')}.parquet").sort_values(["subject", "row"])
        d = d.reset_index(drop=True)
        if reference is None:
            # MMLU repeats some questions within a subject, so align by position and verify the text.
            reference = d[["subject", "row", "example"]].merge(
                keys, on=["subject", "row"], how="left", validate="1:1", suffixes=("", "_ref"))
            if reference.key.isna().any() or (reference.example != reference.example_ref).any():
                n = int((reference.example != reference.example_ref).sum())
                raise ValueError(f"gold reference does not line up with the questions ({n} rows differ)")
            reference = reference.drop(columns="example_ref")
            reference = reference.rename(columns={"key": "gold", "example": "query"})
            reference["gold"] = reference.gold.astype(int)
        elif len(d) != len(reference) or (d["example"].values != reference["query"].values).any():
            raise ValueError(f"{m}: questions differ from the first model")
        logit = np.vstack(d.predictions.values).astype(np.float64)
        # The recovered key must reproduce each model's own recorded right/wrong results.
        key_conflicts += int(((logit.argmax(axis=1) == reference.gold.values) != (d.acc.values > 0.5)).sum())
        stack.append(logit)
    logits = np.stack(stack)  # models x questions x options
    probs = softmax(logits, axis=2).astype(np.float32)

    q = reference.rename(columns={"row": "subject_row"})
    parts = q["query"].map(split_query)
    q["question"] = parts.map(lambda p: p[0])
    q["options"] = parts.map(lambda p: p[1])
    unparsed = int(q.options.map(lambda o: o[0] is None).sum())
    q["options"] = q.options.map(lambda o: ["" if x is None else x for x in o])
    q["qid"] = q.subject + "/" + q.subject_row.astype(str)

    r = load_redux()
    q = join_redux(q, r)
    cols = {"error_type": "redux_error", "correct_answer": "redux_correct", "potential_reason": "redux_reason",
            "answer": "redux_gold"}
    matched = q.redux_idx >= 0
    for src, dst in cols.items():
        q[dst] = None
        q.loc[matched, dst] = r.loc[q.loc[matched, "redux_idx"], src].astype(str).values
    q["redux_correct_idx"] = q.redux_correct.map(lambda v: option_index(v) if v is not None else -1)
    gold_agree = (q.loc[matched, "redux_gold"].astype(float).astype(int) == q.loc[matched, "gold"]).mean()

    report = {
        "n_models": len(names),
        "n_questions": len(q),
        "n_redux": len(r),
        "redux_matched": int(matched.sum()),
        "redux_match_rate": round(float(matched.sum() / len(r)), 4),
        "by_tier": q.join_tier[matched].value_counts().to_dict(),
        "official_key_agrees_with_redux": round(float(gold_agree), 4),
        "unparsed_queries": unparsed,
        "key_vs_recorded_acc_conflicts": key_conflicts,
        "key_check_cells": int(len(q) * len(names)),
    }
    os.makedirs("data/processed", exist_ok=True)
    keep = ["qid", "subject", "subject_row", "question", "options", "gold", "join_tier",
            "redux_error", "redux_correct", "redux_correct_idx", "redux_reason"]
    q[keep].to_parquet("data/processed/questions.parquet", index=False)
    np.savez_compressed("data/processed/probs.npz", probs=probs, models=np.array(names))
    with open("data/processed/join_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
