"""Export the cached tables into compact files the D3 tool loads in the browser.

Outputs in tool/data/:
  questions.json  columnar per-question fields (text, options, keys, Redux labels, scores)
  models.json     per-model name, family, type, macro accuracy, top-tier flag
  matrix.bin      uint8 [2, n_models, n_questions]: plane 0 is each model's chosen option
                  (argmax of its probabilities), plane 1 is its probability on the official
                  key, quantized to 0..255.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("tool/data")
ERROR_CODES = ["unchecked", "ok", "wrong_groundtruth", "multiple_correct_answers", "no_correct_answer",
               "bad_question_clarity", "bad_options_clarity", "expert"]


def fix_mojibake(text):
    if not isinstance(text, str):
        return ""
    try:
        return text.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return text


def type_label(raw):
    raw = str(raw)
    for key in ["pretrained", "fine-tuned", "chat", "merges", "instruction", "RL-tuned"]:
        if key in raw:
            return {"chat": "chat", "merges": "merge", "instruction": "instruction-tuned"}.get(key, key)
    return "other"


def main():
    data = np.load("data/processed/probs.npz")
    probs, model_names = data["probs"], data["models"]
    q = pd.read_parquet("data/processed/questions.parquet").merge(
        pd.read_parquet("data/processed/scores.parquet"), on="qid", validate="1:1")
    ms = pd.read_csv("data/processed/model_scores.csv")
    meta = pd.read_csv("data/processed/models.csv").drop_duplicates("model")
    assert list(ms.model) == list(model_names), "model_scores.csv must follow probs.npz order"
    ms = ms.merge(meta[["model", "family", "model_type", "params_m"]], on="model", how="left", validate="1:1")

    n_m, n_q, _ = probs.shape
    gold = q.gold.values
    votes = probs.argmax(axis=2).astype(np.uint8)
    p_gold = probs[:, np.arange(n_q), gold]
    p_gold_q = np.clip(np.rint(p_gold * 255), 0, 255).astype(np.uint8)

    OUT.mkdir(parents=True, exist_ok=True)
    np.stack([votes, p_gold_q]).tofile(OUT / "matrix.bin")

    subjects = sorted(q.subject.unique())
    subject_idx = {s: i for i, s in enumerate(subjects)}
    error = q.redux_error.fillna("unchecked")
    unknown = set(error) - set(ERROR_CODES)
    assert not unknown, f"unexpected Redux labels {unknown}"

    def col(name, digits=4, values=None):
        values = q[name].values if values is None else values
        # JSON has no NaN; undefined scores (2PL b when a is 0) become null.
        return [None if not np.isfinite(v) else round(float(v), digits) for v in values]

    # 2PL difficulty b = -d / a explodes when a is near zero; the tool shows it clipped.
    irt_b = np.clip(q.irt_b.values, -6, 6)
    questions = {
        "n": int(n_q),
        "subjects": subjects,
        "error_codes": ERROR_CODES,
        "qid": q.qid.tolist(),
        "subject": [subject_idx[s] for s in q.subject],
        "question": q.question.tolist(),
        "options": [list(map(str, o)) for o in q.options],
        "gold": gold.astype(int).tolist(),
        "error": [ERROR_CODES.index(e) for e in error],
        "redux_correct": q.redux_correct_idx.astype(int).tolist(),
        "redux_reason": [fix_mojibake(r) for r in q.redux_reason],
        "confidence": col("confidence"),
        "variability": col("variability"),
        "correctness": col("correctness"),
        "ability_alignment": col("ability_alignment"),
        "top_confidence": col("top_confidence"),
        "top_correctness": col("top_correctness"),
        "consensus_alt": q.consensus_alt.astype(int).tolist(),
        "consensus_share": col("consensus_share"),
        "vote_entropy": col("vote_entropy"),
        "top_vote_entropy": col("top_vote_entropy"),
        "irt_a": col("irt_a"),
        "irt_b": col("irt_b", values=irt_b),
    }
    with open(OUT / "questions.json", "w") as f:
        json.dump(questions, f, separators=(",", ":"), ensure_ascii=False, allow_nan=False)

    models = {
        "n": int(n_m),
        "name": ms.model.tolist(),
        "family": ms.family.fillna("unknown").tolist(),
        "type": [type_label(t) for t in ms.model_type],
        "params_b": [None if pd.isna(p) else round(float(p) / 1000, 1) for p in ms.params_m],
        "macro_acc": [round(float(v), 5) for v in ms.macro_acc],
        "theta": [round(float(v), 4) for v in ms.theta],
        "top_tier": ms.top_tier.astype(bool).tolist(),
        "mentions_mmlu": ms.name_mentions_mmlu.astype(bool).tolist(),
    }
    with open(OUT / "models.json", "w") as f:
        json.dump(models, f, separators=(",", ":"))

    # Sanity check: the browser recomputes macro accuracy from votes, so it must match.
    correct = votes == gold[None, :]
    macro = pd.DataFrame(correct.T, index=q.subject).groupby(level=0).mean().mean(axis=0).values
    assert np.allclose(macro, ms.macro_acc.values, atol=1e-6), "macro accuracy mismatch"

    for p in sorted(OUT.iterdir()):
        print(f"{p.name:16s} {p.stat().st_size / 1e6:6.2f} MB")


if __name__ == "__main__":
    main()
