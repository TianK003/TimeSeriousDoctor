# NOTES

Decisions taken where SPEC.md was ambiguous (one line each, newest at the bottom).

- Time budget is **4 h** of implementation (not 5): MVP only — cover the main cases, skip edge cases.
- Product name is **TimeSeriousDoctor**; the Python package stays `atlas` so it matches SPEC.md everywhere.
- API key env var is `GEMMA4_API_KEY` (what our `.env` uses); `GEMINI_API_KEY` is accepted as a fallback. Use `atlas.config.get_api_key()`.
- `Dataset` and `RunResult` are dataclasses (they hold DataFrames/arrays); every JSON-facing model is pydantic v2.
- Gemma schemas (`GemmaClusterReport`, `GemmaBaselineAnswer`, `AtlasSummary`) live in `atlas/schema.py`, because `ClusterInfo` embeds the report.
- `ClusterInfo.image_path` is relative to the run folder (e.g. `images/cluster_0.png`); the pipeline writes PNGs straight into `<run>/images/`.
