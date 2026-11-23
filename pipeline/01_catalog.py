"""Scan leaderboard result files and list every run that has complete 5-shot MMLU.

Input:  data/raw/results and data/raw/requests (shallow clones of the HF repos)
Output: data/processed/catalog.parquet, one row per complete run, with a
        `prompt_set` hash that identifies which exact prompts the run used.
"""
import glob
import hashlib
import json
import os
import re
import sys

import pandas as pd

RESULTS = "data/raw/results"
REQUESTS = "data/raw/requests"
OUT = "data/processed/catalog.parquet"
KEY = re.compile(r"^harness\|hendrycksTest-(.+)\|5$")
N_SUBJECTS = 57


def load_requests():
    """Map model name to metadata from its most complete request file."""
    meta = {}
    for path in glob.glob(f"{REQUESTS}/**/*.json", recursive=True):
        try:
            with open(path) as f:
                r = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        name = r.get("model")
        if not name:
            continue
        keep = meta.get(name)
        if keep is None or r.get("status") == "FINISHED":
            meta[name] = r
    return meta


def parse_run(path):
    with open(path) as f:
        d = json.load(f)
    tasks = d.get("summary_tasks") or {}
    results = d.get("results") or {}
    subjects = {}
    for key, s in tasks.items():
        m = KEY.match(key)
        if m and key in results and "acc" in results[key]:
            subjects[m.group(1)] = (s.get("hashes") or {}, results[key]["acc"], s)
    if len(subjects) != N_SUBJECTS:
        return None
    names = sorted(subjects)
    fp = lambda field: hashlib.sha256(
        "|".join(str(subjects[n][0].get(field)) for n in names).encode()
    ).hexdigest()[:12]
    cfg = d.get("config_general") or {}
    return {
        "lighteval_sha": cfg.get("lighteval_sha"),
        "model_sha": cfg.get("model_sha"),
        "model_dtype": cfg.get("model_dtype"),
        "model_size": cfg.get("model_size"),
        "prompt_set": fp("hash_full_prompts"),
        "example_set": fp("hash_examples"),
        "mmlu_macro_acc": sum(subjects[n][1] for n in names) / N_SUBJECTS,
        "n_truncated": sum(subjects[n][2].get("truncated", 0) or 0 for n in names),
        "n_non_truncated": sum(subjects[n][2].get("non_truncated", 0) or 0 for n in names),
    }


def main():
    if not os.path.isdir(RESULTS):
        sys.exit(f"Missing {RESULTS}; clone the results repo first (see README).")
    meta = load_requests()
    rows, unreadable = [], 0
    for path in glob.glob(f"{RESULTS}/**/results_*.json", recursive=True):
        try:
            run = parse_run(path)
        except (json.JSONDecodeError, OSError):
            unreadable += 1
            continue
        if run is None:
            continue
        rel = os.path.relpath(path, RESULTS)
        model = os.path.dirname(rel)
        stamp = re.search(r"results_(.+)\.json$", rel).group(1)
        r = meta.get(model, {})
        run.update(
            model=model,
            run=stamp,
            params_m=r.get("params"),
            model_type=r.get("model_type"),
            weight_type=r.get("weight_type"),
            base_model=r.get("base_model"),
            license=r.get("license"),
            likes=r.get("likes"),
            has_request=model in meta,
        )
        rows.append(run)
    df = pd.DataFrame(rows).sort_values(["model", "run"])
    # Some request files hold a list in these free-text fields.
    for col in ["model_type", "weight_type", "base_model", "license"]:
        df[col] = df[col].map(lambda v: v if v is None or isinstance(v, str) else str(v))
    df["params_m"] = pd.to_numeric(df["params_m"], errors="coerce")
    df["likes"] = pd.to_numeric(df["likes"], errors="coerce")
    df.to_parquet(OUT, index=False)
    print(f"complete runs: {len(df)}; models: {df.model.nunique()}; unreadable files: {unreadable}")


if __name__ == "__main__":
    main()
