"""Record exactly what the cached data is, or verify a copy against that record.

Usage:  python pipeline/06_manifest.py            write data/processed/manifest.json
        python pipeline/06_manifest.py --verify   check the cached files against it

Checksums cover content rather than file bytes, because compressed files can differ
between runs and machines while holding identical data.
"""
import hashlib
import importlib.metadata as meta
import json
import subprocess
import sys

import numpy as np
import pandas as pd

MANIFEST = "data/processed/manifest.json"
GOLD = {"model": "mistralai/Mistral-7B-v0.1", "run": "2023-09-27T15-30-59.039834",
        "revision": "7b4f5b61de1f8f11f143aa59f90349d79b8d16e5"}
HARNESS = "494ee12240e716e804ae9ea834f84a2c864c07ca"


def sha_frame(df):
    text = df.astype(str).to_csv(index=False).encode()
    return hashlib.sha256(text).hexdigest()


def content_checksums():
    data = np.load("data/processed/probs.npz")
    questions = pd.read_parquet("data/processed/questions.parquet")
    scores = pd.read_parquet("data/processed/scores.parquet")
    return {
        "probs_values": hashlib.sha256(data["probs"].tobytes()).hexdigest(),
        "probs_shape": list(data["probs"].shape),
        "model_order": hashlib.sha256("\n".join(data["models"].tolist()).encode()).hexdigest(),
        "questions_table": sha_frame(questions),
        "scores_table": sha_frame(scores.round(5)),
    }


def clone_revision(path):
    return subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def verify():
    with open(MANIFEST) as f:
        recorded = json.load(f)["checksums"]
    current = content_checksums()
    bad = [k for k in recorded if recorded[k] != current[k]]
    if bad:
        sys.exit(f"CHECKSUM MISMATCH: {', '.join(bad)}")
    print(f"verified: {len(recorded)} checksums match the manifest")


def write():
    log = pd.read_csv("data/processed/downloads.csv")
    models = pd.read_csv("data/processed/models.csv")
    data = np.load("data/processed/probs.npz")
    used = pd.DataFrame({"model": data["models"]}).merge(models, on="model").merge(
        log[log.status == "ok"][["model", "revision"]], on="model")
    with open("data/processed/join_report.json") as f:
        join = json.load(f)
    manifest = {
        "harness": {"name": "lighteval", "sha": HARNESS, "task": "harness|hendrycksTest|5 (5-shot MMLU)"},
        "sources": {
            "results": {"repo": "open-llm-leaderboard-old/results", "revision": clone_revision("data/raw/results")},
            "requests": {"repo": "open-llm-leaderboard-old/requests", "revision": clone_revision("data/raw/requests")},
            "mmlu_redux": {"repo": "edinburgh-dawg/mmlu-redux-2.0", "revision": clone_revision("data/raw/redux")},
            "answer_key": GOLD,
        },
        "join": join,
        "failed_downloads": log[log.status != "ok"][["model", "status"]].to_dict("records"),
        "models": used[["model", "run", "revision", "lighteval_sha", "family", "origin"]].to_dict("records"),
        "packages": {p: meta.version(p) for p in ["numpy", "pandas", "pyarrow", "scipy", "huggingface_hub"]},
        "python": sys.version.split()[0],
        "checksums": content_checksums(),
    }
    with open(MANIFEST, "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"wrote {MANIFEST}: {len(used)} models")
    for k, v in manifest["checksums"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    verify() if "--verify" in sys.argv else write()
