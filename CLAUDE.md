# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**TimeSeriousDoctor** finds the recurring ways forecasts fail across thousands of time series and has a
forecasting expert's knowledge applied to each one. You give it actuals and predictions. It cuts the scaled
residuals into windows and keeps only the abnormal ones. Then it clusters those windows into failure
patterns (lagging, missed peaks, unadapted level shifts, over-smoothing, ...). **Gemma 4** looks at each
cluster's chart panel and its feature numbers and diagnoses it: name, evidence, likely causes, fixes, and a
metric to monitor. Expert knowledge comes in two ways. `taxonomy.yaml` holds the failure definitions,
signatures, causes and fixes that the time-series expert writes and refines; it is injected into every
prompt. Expert Confirm/Reject/Rename feedback (P2) is fed back as few-shot examples. The goal is to turn
hours of eyeballing plots into a ranked list of explained model problems.

Core rule: **code measures, Gemma explains.** Every number in the UI comes from code. Gemma only interprets
the numbers and images it is handed, and never computes them.

SPEC.md calls the project "Residual Atlas", and the Python package is `atlas`. Keep that name.

## Working in this repo (hackathon rules)

- `SPEC.md` is the binding spec. Build in its §9 priority order (P0 → P1 → P2). The **time budget is 4 h**, so build the MVP and skip edge cases.
- Two things are required for judging and must never be cut:
  - The **Gemma-alone comparison** (SPEC §7.11). Same model, taxonomy and series, but Gemma alone only sees a full-series chart. It proves the pipeline beats the naive "just ask Gemma" approach.
  - The **submission checklist + README template** (SPEC §15). The README sections are fixed by the judging template.
- Three people each run their own Claude Code session. Each person owns specific tasks and files: A = engine, B = Gemma + app, C = data + truth. Every stub module's docstring names its owner, task ID and spec section (e.g. `Owner: A. Task: P0.6. Spec: SPEC.md §7.3.`). Stay inside your owner's files.
- The shared interfaces are `atlas/schema.py`, `atlas/config.py`, `taxonomy.yaml` and the run-folder format. Change them only when the team agrees.
- If the spec is ambiguous, pick the simplest option and add a one-line note to `NOTES.md`. `NOTES.md` also records verified facts about calling Gemma (backend, model ids, system-instruction / JSON / `response_schema` support from `scripts/smoke_gemini.py`). Check it before touching `atlas/gemma.py`.
- After each task, print a 3-line summary: what was built, how to run it, and what's next.

## Commands

Use Python 3.11. Every dependency is in `requirements.txt`. MOMENT (`momentfm` + `torch`) is optional and lives in `requirements-moment.txt`.

```bash
py -3.11 -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # Windows (POSIX: python3.11, .venv/bin/)
.venv/Scripts/python -m pytest -q                                          # all tests (pyproject sets pythonpath=".")
.venv/Scripts/python -m pytest tests/test_schema.py::test_save_load_run_roundtrip -q   # single test
.venv/Scripts/python scripts/smoke_gemini.py --env-file .env               # real-API Gemma check (network)
```

Pipeline commands from the spec. The scripts are stubs until their tasks land:

```bash
python scripts/make_synthetic.py --n 400 --out data/synthetic
python scripts/make_mock_run.py                       # fake run so the UI can be built before the engine
python scripts/run_atlas.py --data data/synthetic --embedder features --run-name synth_features [--no-gemma] [--config config.default.yaml]
python scripts/run_eval.py --runs runs/synth_features
streamlit run app.py
```

There is no linter or build step.

## Architecture

```
dataset folder ─► residuals ─► windows ─► gate ─► features / MOMENT ─► cluster ─► PNG panels ─► Gemma ─► run folder ─► Streamlit
```

- **Input contract** (`schema.load_dataset`) is a folder with `data.parquet` and `series_meta.parquet`. `data.parquet` is long format (`series_id, ts, actual, predicted, is_forecast[, true_label]`). `series_meta.parquet` has `series_id, model_card[, group]`. The synthetic and M5 producers write it with `schema.save_dataset`.
- **The run folder `runs/<name>/` is the only interface between the engine and the UI.** `pipeline.run_atlas()` builds a `RunResult` and calls `save_run`. `app.py` only calls `load_run` / `list_runs`, and never runs the pipeline. That is why the UI can be built against `make_mock_run.py` output before the engine exists.
- **`windows.parquet` conventions:**
  - `window_id = "<series_id>#<start_idx>"`. `start_idx` and `end_idx` (exclusive) index the series' *forecast* rows.
  - `cluster` is `-1` for gated windows that didn't cluster (`CLUSTER_NOISE`, the "look manually" bucket, which never gets a Gemma call). It is `-2` for windows the gate dropped (`CLUSTER_NOT_GATED`).
  - `emb_row` maps a gated window to its row in `embeddings.npy`, and is `-1` when not gated.
  - Every feature column from `atlas.features.FEATURES` is appended.
- **Cluster ids** run `0..n-1`, ordered by size descending. `ClusterInfo.image_path` is relative to the run folder, and the pipeline writes PNGs straight into `<run>/images/`. `ClusterInfo.report` is a `GemmaClusterReport`, or `None` when Gemma failed. The UI then shows "explanation unavailable".
- **Residual convention:** `r = actual − predicted`, so a positive r means under-prediction. It is scaled by the per-series MASE denominator (mean |a_t − a_{t−season}| over history), so about 1.0 is a typical seasonal-naive error. Gate thresholds and Gemma prompts both depend on this scale.
- **Embedders** (`cfg.embedder`):
  - `features`: handcrafted, interpretable features. This is the default and the fallback.
  - `moment`: MOMENT-1 embeddings, then PCA(16).
  - `both`: both blocks, each scaled by 1/sqrt(dim).
  - `torch`/`momentfm` may be imported **only lazily inside `atlas/embed_moment.py`**. Everything else must run without them.
- **Gemma** is called through the Gemini API with the `google-genai` SDK. Default model is `gemma-4-26b-a4b-it`, with fallback `gemma-4-31b-it`.
  - Calls are throttled to `cfg.gemma.rpm` and cached on disk by sha256(model + system + user text + image bytes) under `cache/gemma/`. Re-runs must never re-call the API.
  - Raw responses go to `<run>/gemma_raw/`. Output is validated against the pydantic models in `schema.py`, with one retry that includes the validation error.
- **Config:** `atlas.config.Config` (pydantic) holds every default, mirrored in `config.default.yaml`. Pass it explicitly; there are no globals. Get the API key with `atlas.config.get_api_key()`, which checks `GEMMA4_API_KEY` first and falls back to `GEMINI_API_KEY`. The key is loaded from `.env`.
- **Labels and evaluation:** synthetic data carries a row-level `true_label`. A window's label is the failure type when at least 50% of its rows carry it, otherwise `healthy`. The `taxonomy.yaml` keys double as Gemma's allowed `failure_type` values (plus `other`) and as the evaluation labels.

## Testing

Tests never touch the network. Mock the Gemini client, and when testing `get_api_key`, patch `atlas.config.load_dotenv` so the real `.env` is never read. Every P0 module needs at least one fast pytest. The feature tests in SPEC §7.5 assert the expected feature signatures on synthetic injections.
