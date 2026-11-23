"""Download 5-shot MMLU details for the selected models, keeping only needed columns.

Usage:  python pipeline/03_download.py [pilot|all]
Output: data/raw/details/<org>__<name>.parquet, one file per complete model,
        and data/processed/downloads.csv, which records the dataset revision of each.

Safe to rerun: finished models are skipped. Hugging Face rate-limits anonymous
requests, so this runs one file at a time and waits when it receives a 429.
Setting HF_TOKEN gives a higher limit.
"""
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import EntryNotFoundError, HfHubHTTPError, RepositoryNotFoundError

OUT_DIR = "data/raw/details"
LOG = "data/processed/downloads.csv"
N_SUBJECTS = 57
GOLD_FILE = "data/processed/gold_reference.parquet"
# Runs by our harness version leave `gold` empty, so the key comes from this older run.
# 04_build_table.py checks it against every model's recorded right/wrong results.
GOLD_MODEL, GOLD_RUN = "mistralai/Mistral-7B-v0.1", "2023-09-27T15-30-59.039834"
PAUSE = 0.15
TMP_CACHE = "data/raw/.hf_tmp"  # raw files are ~150 MB per model, so delete after compacting


def with_backoff(fn, tries=8):
    for attempt in range(tries):
        try:
            return fn()
        except (EntryNotFoundError, RepositoryNotFoundError):
            raise
        except Exception as e:  # rate limits surface as several exception types
            text = str(e)
            if "404" in text or attempt == tries - 1:
                raise
            wait = 310 if "429" in text else 2 ** attempt
            print(f"    waiting {wait}s after: {text[:80]}", flush=True)
            time.sleep(wait)


def fetch_model(api, model, run):
    repo = "open-llm-leaderboard-old/details_" + model.replace("/", "__")
    info = with_backoff(lambda: api.dataset_info(repo))
    files = with_backoff(lambda: api.list_repo_files(repo, repo_type="dataset", revision=info.sha))
    files = [f for f in files if f.startswith(run + "/") and "hendrycksTest" in f and f.endswith(".parquet")]
    if len(files) != N_SUBJECTS:
        raise ValueError(f"expected {N_SUBJECTS} files, found {len(files)}")
    frames = []
    for name in sorted(files):
        path = with_backoff(lambda: hf_hub_download(repo, name, repo_type="dataset", revision=info.sha, cache_dir=TMP_CACHE))
        subject = name.split("hendrycksTest-")[1].split("|")[0]
        raw = pd.read_parquet(path)
        df = pd.DataFrame({"subject": subject, "row": range(len(raw)), "example": raw["example"],
                           "predictions": raw["predictions"]})
        # Both result formats exist: older runs have `acc` and `gold`, newer ones `metrics`.
        df["acc"] = raw["acc"] if "acc" in raw else raw["metrics"].map(lambda m: m["acc"])
        if "gold" in raw and raw["gold"].map(lambda g: isinstance(g, (int, np.integer))).all():
            df["gold"] = raw["gold"]
        frames.append(df)
        time.sleep(PAUSE)
    return pd.concat(frames, ignore_index=True), info.sha


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "pilot"
    models = pd.read_csv("data/processed/models.csv")
    want = 20 if mode == "pilot" else 150
    os.makedirs(OUT_DIR, exist_ok=True)
    log = pd.read_csv(LOG).to_dict("records") if os.path.exists(LOG) else []
    done = {r["model"] for r in log if r["status"] == "ok"}
    failed = {r["model"] for r in log if r["status"] != "ok"}
    api = HfApi()
    if not os.path.exists(GOLD_FILE):
        ref, sha = fetch_model(api, GOLD_MODEL, GOLD_RUN)
        ref[["subject", "row", "example", "gold"]].to_parquet(GOLD_FILE, index=False)
        print(f"gold reference: {len(ref)} questions from {GOLD_MODEL} at {sha}", flush=True)
    for row in models.itertuples():
        if len(done) >= want:
            break
        if row.model in done or row.model in failed:
            continue
        start = time.time()
        try:
            df, sha = fetch_model(api, row.model, row.run)
            df.to_parquet(f"{OUT_DIR}/{row.model.replace('/', '__')}.parquet", index=False)
            done.add(row.model)
            log.append(dict(model=row.model, run=row.run, revision=sha, status="ok", rows=len(df)))
            print(f"[{len(done)}/{want}] {row.model} {len(df)} rows {time.time() - start:.0f}s", flush=True)
        except (ValueError, EntryNotFoundError, RepositoryNotFoundError, HfHubHTTPError) as e:
            failed.add(row.model)
            log.append(dict(model=row.model, run=row.run, revision="", status=f"failed: {str(e)[:100]}", rows=0))
            print(f"FAILED {row.model}: {str(e)[:100]}", flush=True)
        shutil.rmtree(TMP_CACHE, ignore_errors=True)
        pd.DataFrame(log).to_csv(LOG, index=False)
    print(f"finished: {len(done)} models ok, {len(failed)} failed", flush=True)


if __name__ == "__main__":
    main()
