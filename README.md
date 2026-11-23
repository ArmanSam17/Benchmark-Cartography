# Benchmark Cartography

Auditing MMLU's answer keys using the agreement of many public LLMs.
DS-GA 3001, Fall 2026, Team 21: Arman Samayam and Padmapriya Venkatesh.

## Stage 1: data and scores

The cached outputs are committed in `data/processed/`, so analysis and the tool can start
without any downloads. Check that your copy is intact:

```bash
python pipeline/06_manifest.py --verify
```

To rebuild everything from the sources:

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
export GIT_LFS_SKIP_SMUDGE=1
git clone --depth 1 https://huggingface.co/datasets/open-llm-leaderboard-old/results data/raw/results
git clone --depth 1 https://huggingface.co/datasets/open-llm-leaderboard-old/requests data/raw/requests
GIT_LFS_SKIP_SMUDGE=0 git clone --depth 1 https://huggingface.co/datasets/edinburgh-dawg/mmlu-redux-2.0 data/raw/redux
.venv/bin/python pipeline/01_catalog.py
.venv/bin/python pipeline/02_select_models.py
.venv/bin/python pipeline/03_download.py all
.venv/bin/python pipeline/04_build_table.py
.venv/bin/python pipeline/05_scores.py
.venv/bin/python pipeline/06_manifest.py --verify
.venv/bin/python pipeline/07_map.py
.venv/bin/python tests/test_scores.py
```

The download takes about 90 seconds per model without a token because of rate limits.
Setting `HF_TOKEN` raises the limit. The clones were made on case-insensitive macOS, where
git warns about a few model names that differ only by case; those models are unaffected
unless selected, and a failed one is replaced from the reserve list.

## Outputs in `data/processed/`

| File | Contents |
|---|---|
| `questions.parquet` | One row per question: text, options, official key, MMLU-Redux label and correction |
| `probs.npz` | `probs[model, question, option]`, plus the model order |
| `scores.parquet` | Cartography scores and 2PL parameters per question |
| `model_scores.csv` | Accuracy, IRT ability, top-tier flag and MMLU-in-name flag per model |
| `models.csv` | Selected models and reserves, with family, size and origin |
| `downloads.csv` | Dataset revision of each download, and any failures |
| `join_report.json` | Join rate by tier, and the answer-key check |
| `manifest.json` | Source revisions, harness version, model list and content checksums |

## Decisions worth knowing

- **Harness version:** one lighteval SHA (`494ee122`), the most common. Every complete MMLU
  run in the archive has identical prompt hashes across SHAs, so this is a conservative choice.
  It does exclude some well-known models, such as Mistral-7B, that ran on other SHAs.
- **Answer key:** runs on this SHA leave `gold` empty. The key comes from an older Mistral-7B
  run, and `04_build_table.py` checks it against every model's recorded right or wrong result.
- **Model filters:** no truncated prompts, macro accuracy of at least 0.30, latest run per
  model, no duplicates within a family, and at most 12 models per family in the target set.
- **"Base" label:** `origin` comes from the self-reported `model_type`, which is unreliable
  for some merges, so base-only robustness results need a manual check of that list.
- **MMLU-named models:** kept and flagged, so results can be reported with and without them.
- **Near-duplicate models:** family deduplication doesn't catch merges across families. Four
  pairs agree on over 99% of votes, and five 70B-class models form a cluster above 90% that
  inflates their IRT ability. All 150 stay in the cached data. Keeping one model per cluster
  raises the correlation of ability with accuracy from 0.89 to 0.94 and barely moves
  wrong-key AUPRC, but analysis should report this sensitivity check.
- **2PL fit:** `05_scores.py` explains the choice of form, scaling and prior. Each one fixes a
  failure we observed that would have made the IRT baseline look worse than it should.
- **`redux_correct_idx`:** -1 when a correction isn't a single option, such as "0 or 1".

## Stage 3: the tool

A D3 page with three linked views: the data map, the question inspector and the leaderboard view.
Its data in `tool/data/` is exported from `data/processed/`, so rerun the export after the scores change:

```bash
.venv/bin/python pipeline/08_export_tool.py
python3 -m http.server 8766 --directory tool   # then open http://localhost:8766
```

- **Data map:** any two scores as axes, colored by MMLU-Redux label. Drag to select questions
  (hold Shift to add), click a point to inspect it. The presets select the Redux wrong keys, all
  Redux errors, or questions below a top-tier confidence threshold.
- **Question inspector:** the options with the official key, the Redux correction and the
  consensus alternative; the vote shares of all models and of the top tier; each model's probability
  on the key plotted against its accuracy.
- **Model set:** the Models menu switches every view to a subset (pretrained base models, one per
  family, two random halves split within families, or without the two models named after MMLU).
  Scores are recomputed in the browser from the same formulas as `05_scores.py`; they match the
  pipeline to within 0.003 for the full set (the probabilities are stored as 8-bit values). The 2PL
  fit is not refitted per subset. A collapsible table under the map gives detection AUPRC for the
  current set.
- **Leaderboard view:** recomputes every model's macro accuracy after removing the selected questions,
  or re-keying them to the consensus alternative or the Redux fix. It reports rank swaps and
  Kendall τ, and compares them with 200 difficulty-matched random draws (same count per decile of
  share correct, from questions Redux did not flag; seeded, so results repeat).

### Hosting on GitHub Pages

`.github/workflows/pages.yml` publishes only the `tool/` folder whenever `tool/` changes on `main`.
The repo owner turns it on once: **Settings → Pages → Build and deployment → Source: GitHub Actions**.
The tool is then at `https://armansam17.github.io/Benchmark-Cartography/`, and the expert-study
version at the same address with `?study` added. `tool/data/` must be committed; it is generated by
`08_export_tool.py`, so rerun the export and commit it whenever the scores change.

## Stage 4: expert study

`study/expert_study_protocol.md` is the session script, the facilitator rules, the analysis plan and
the log template for the qualitative think-aloud sessions (RQ4). Run sessions in study mode,
`http://localhost:8766/?study`, which hides all MMLU-Redux labels and adds the fixed 8-question
triage set from `tool/data/study_set.json`. The answer key for that set is in `study/facilitator_key.md`;
keep it away from participants.
