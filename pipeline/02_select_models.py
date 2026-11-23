"""Choose the models we analyse, stratified by family, size and MMLU score.

Input:  data/processed/catalog.parquet
Output: data/processed/models.csv, ordered so the first rows form the pilot.
        Models past the target stay listed as reserves, used if a download fails.

Filters: one harness version, no truncated prompts, score clearly above chance,
one run per model, and no duplicate models (identical MMLU score within a family).
"""
import re

import numpy as np
import pandas as pd

SEED = 21
HARNESS = "494ee12240e716e804ae9ea834f84a2c864c07ca"  # most common lighteval SHA
MIN_ACC = 0.30  # chance is 0.25; lower scores are usually broken conversions
TARGET = 150
PILOT = 20
CAP_PER_FAMILY = 12  # limits the fine-tunes and merges of any one base model

FAMILIES = [
    "mixtral", "mistral", "zephyr", "llama", "alpaca", "vicuna", "wizard", "qwen", "yi-",
    "gemma", "phi", "falcon", "mpt", "bloom", "starling", "deepseek", "baichuan", "internlm",
    "solar", "gpt-?j", "gpt-?neo", "pythia", "stablelm", "stable", "rwkv", "xverse", "orca",
    "hermes", "dolphin", "openchat", "tulu", "olmo", "smaug", "dbrx",
]
PATTERN = re.compile("|".join(FAMILIES))


def family_of(row):
    text = f"{row.model} {row.base_model or ''}".lower()
    match = PATTERN.search(text)
    return match.group(0).rstrip("-").replace("-", "") if match else "other"


def origin_of(model_type):
    t = (model_type or "").lower()
    return "base" if "pretrained" in t else "derived"


def main():
    c = pd.read_parquet("data/processed/catalog.parquet")
    steps = [("complete runs", len(c))]
    c = c[c.lighteval_sha == HARNESS]
    steps.append(("single harness version", len(c)))
    c = c[c.n_truncated == 0]
    steps.append(("no truncated prompts", len(c)))
    c = c[c.mmlu_macro_acc >= MIN_ACC]
    steps.append((f"macro accuracy >= {MIN_ACC}", len(c)))
    c = c[c.has_request].sort_values("run").groupby("model", as_index=False).tail(1)
    steps.append(("latest run per model", len(c)))

    c = c.assign(
        family=c.apply(family_of, axis=1),
        origin=c.model_type.map(origin_of),
        size_band=pd.cut(c.params_m, [-1, 1, 10, 30, 1e6], labels=["<1B", "1-10B", "10-30B", ">30B"]),
    )
    c["score_q"] = pd.qcut(c.mmlu_macro_acc, 4, labels=False)
    c["dup_key"] = c.mmlu_macro_acc.round(6)
    c = c.drop_duplicates(["family", "dup_key"])
    steps.append(("no duplicates within family", len(c)))
    for name, n in steps:
        print(f"{name:32s}{n:6d}")

    rng = np.random.default_rng(SEED)
    c = c.assign(tie=rng.random(len(c)))

    # Within a family, take base models first, then spread across size and score.
    c = c.sort_values(["family", "origin", "tie"])
    c["stratum_rank"] = c.groupby(["family", "origin", "size_band", "score_q"], observed=True).cumcount()
    c = c.sort_values(["family", "stratum_rank", "origin", "tie"])
    c["family_rank"] = c.groupby("family").cumcount()

    # Round-robin over families, so no family dominates the sample or the pilot.
    ordered = c.sort_values(["family_rank", "tie"]).reset_index(drop=True)
    capped = ordered[ordered.family_rank < CAP_PER_FAMILY]
    reserve = ordered[ordered.family_rank >= CAP_PER_FAMILY]
    out = pd.concat([capped, reserve], ignore_index=True)
    out["role"] = ["pilot" if i < PILOT else "target" if i < TARGET else "reserve" for i in range(len(out))]
    cols = ["model", "run", "role", "family", "origin", "size_band", "params_m",
            "mmlu_macro_acc", "model_sha", "lighteval_sha", "model_type"]
    out[cols].to_csv("data/processed/models.csv", index=False)

    use = out[out.role != "reserve"]
    print(f"\nlisted {len(out)}; pilot + target {len(use)}; families {use.family.nunique()}")
    print(use.origin.value_counts().to_string())
    print(use.size_band.value_counts().to_string())
    print(use.mmlu_macro_acc.describe().round(3).to_string())
    print("\npilot:")
    print(out[out.role == "pilot"][["model", "family", "origin", "mmlu_macro_acc"]].to_string(index=False))


if __name__ == "__main__":
    main()
