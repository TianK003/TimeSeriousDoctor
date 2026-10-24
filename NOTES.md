# NOTES

Decisions taken where SPEC.md was ambiguous (one line each, newest at the bottom).

- Time budget is **4 h** of implementation (not 5): MVP only — cover the main cases, skip edge cases.
- Product name is **TimeSeriousDoctor**; the Python package stays `atlas` so it matches SPEC.md everywhere.
- API key env var is `GEMMA4_API_KEY` (what our `.env` uses); `GEMINI_API_KEY` is accepted as a fallback. Use `atlas.config.get_api_key()`.
- `Dataset` and `RunResult` are dataclasses (they hold DataFrames/arrays); every JSON-facing model is pydantic v2.
- Gemma schemas (`GemmaClusterReport`, `GemmaBaselineAnswer`, `AtlasSummary`) live in `atlas/schema.py`, because `ClusterInfo` embeds the report.
- `ClusterInfo.image_path` is relative to the run folder (e.g. `images/cluster_0.png`); the pipeline writes PNGs straight into `<run>/images/`.

## P0.2 — Gemini smoke test (2026-10-24)
Script: `scripts/smoke_gemini.py --env-file <path to .env>` (5 checks per model plus a `combined` check = the real P0.12 call; `--models` / `--checks` select subsets).

- **Backend: Gemini Developer API.** Client construction: `genai.Client(api_key=key)` with `key = os.environ.get("GEMMA4_API_KEY") or os.environ.get("GEMINI_API_KEY")` (google-genai 2.33.0). The Vertex AI express fallback (`genai.Client(vertexai=True, api_key=key)`) does NOT work with our key: `403 PERMISSION_DENIED` (the Vertex API is not enabled for the project). Do not build a Vertex path.
- **Model names that work:** `gemma-4-26b-a4b-it` and `gemma-4-31b-it` (exactly the SPEC names, passed without a `models/` prefix; `client.models.list()` shows them as `models/gemma-4-26b-a4b-it` and `models/gemma-4-31b-it`, the only two Gemma models listed). **Use `gemma-4-26b-a4b-it` as the default**: every check passed on it in every run. `gemma-4-31b-it` supports the same features but is unreliable right now (see below).
- **system_instruction: works** natively on both models (`GenerateContentConfig(system_instruction=...)`; "answer in UPPERCASE" was honoured, reply `PONG`). No need to prepend the system text to the user message.
- **Image input: works** on both: `types.Part.from_bytes(data=png, mime_type="image/png")` as the first item of `contents`, the text prompt second. Both models described the chart correctly (title, 10-12 unit baseline, spike near week 20).
- **JSON mode: works** (`response_mime_type="application/json"`): on 26b the raw text parsed with `json.loads` in all 3 runs, no code fences. **response_schema: works** with a pydantic class, and `resp.parsed` comes back as the pydantic instance. Also verified on 26b: image + system_instruction + `temperature=0.2` + JSON mime type + `response_schema` in ONE call, using the full SPEC `GemmaClusterReport` shape (nested `list[Cause]`, a `Literal` with 6 values, `list[str]`): valid output, 7.2 s, 288 output tokens.
  - **Recommendation for gemma.py:** `generate_content(model=..., contents=[image_part, user_text], config=GenerateContentConfig(system_instruction=..., temperature=0.2, response_mime_type="application/json", response_schema=GemmaClusterReport))`, then `GemmaClusterReport.model_validate_json(resp.text)`. Keep the SPEC's defensive parse anyway (strip fences, one retry with the error appended): on `gemma-4-31b-it` one `response_schema` reply failed pydantic with `Invalid JSON: trailing characters` (extra content after the object).
  - **Retry 429, 500 and 503 with backoff** (the SPEC already says 429/5xx). `gemma-4-31b-it` returned 500 INTERNAL or 503 "high demand" on 11 of 22 calls (all 5xx, never a 400 rejecting a feature), and failed calls took up to 88 s to come back, so set a client timeout if the UI must not hang. `gemma-4-26b-a4b-it` never returned a 5xx or 429.
- **Thinking:** Gemma 4 thinks by default on plain text, system_instruction, image and plain JSON-mode calls (35-750 thought tokens, reported as `thoughts_token_count`); calls with `response_schema` showed 0 thought tokens. `ThinkingConfig(thinking_budget=0)` is rejected (400 "Thinking budget is not supported for this model"); `ThinkingConfig(thinking_level="MINIMAL")` is accepted and cut an image description call on 26b from 15-17 s (650-750 thought tokens) to 1.4 s (0 thought tokens). With `response_schema` it is not needed; use `thinking_level="MINIMAL"` for free-text image calls (e.g. the P1 baseline) if latency matters. Only tested on an image + plain text prompt.
- **Latency, `gemma-4-26b-a4b-it`:** text 1.8-2.0 s; system_instruction 2.6-3.7 s; image (default thinking) 15-17 s; JSON mode 3.8-4.6 s; response_schema (flat) 1.0 s; image + system + schema (flat) 1.2-1.5 s; image + system + schema (full report shape) 7.2 s. **`gemma-4-31b-it`**, when it answered at all: text 18-34 s, image 42-46 s, JSON mode 14 s, response_schema 12-58 s. No 429 was seen (the rate limit was not deliberately tested).
- **31b tally over 5 runs (OK / 5xx / other):** text 4/3/0, system_instruction 1/3/0 (honoured when it answered), image 2/1/0, json_mode 1/3/0, response_schema 2/0/1 (malformed JSON), combined 0/1/0. Nothing was rejected as unsupported, so the same call shape should work on 31b when it is up.

### Results (final run, one script process per model)
Backend: Gemini Developer API (`genai.Client(api_key=key)`). In the note column `tok=` is prompt/output/thinking tokens.

| model | check | result | latency (s) | note |
|---|---|---|---|---|
| gemma-4-26b-a4b-it | text | OK | 1.8 | reply='pong'; tok=8/1/51 |
| gemma-4-26b-a4b-it | system_instruction | OK | 2.9 | reply='PONG'; instruction honoured; tok=18/2/102 |
| gemma-4-26b-a4b-it | image | OK | 15.3 | tok=266/38/651 This line chart shows weekly demand in units, which mostly fluctuates between 10 and 12 units but experiences a dramatic spike to appr... |
| gemma-4-26b-a4b-it | json_mode | OK | 3.8 | parsed keys=['answer', 'confidence']; raw fenced=False; tok=24/14/137 |
| gemma-4-26b-a4b-it | response_schema | OK | 1.0 | delayed_response 1.0; resp.parsed is Probe=True; tok=27/18/0 |
| gemma-4-26b-a4b-it | combined | OK | 1.2 | delayed_response 0.95; resp.parsed is Probe=True; tok=300/19/0 |
| gemma-4-31b-it | text | FAIL | 78.2 | ServerError: 503 UNAVAILABLE. This model is currently experiencing high demand... |
| gemma-4-31b-it | system_instruction | FAIL | 0.4 | ServerError: 500 INTERNAL. Internal error encountered. |
| gemma-4-31b-it | image | OK | 42.0 | tok=266/35/355 The line chart shows weekly demand over 30 weeks, characterized by steady fluctuations between 10 and 12 units with a significant spi... |
| gemma-4-31b-it | json_mode | OK | 14.2 | parsed keys=['answer', 'confidence']; raw fenced=False; tok=24/14/101 |
| gemma-4-31b-it | response_schema | FAIL | 45.8 | ValidationError: Invalid JSON: trailing characters at line 5 column 1 (model output, not an API error) |
| gemma-4-31b-it | combined | FAIL | 0.9 | ServerError: 500 INTERNAL. Internal error encountered. |
- Executive summary (P1.7): Gemma returns 504 DEADLINE_EXCEEDED for the text-only AtlasSummary call on 2026-10-24 afternoon (also with thinking_level=MINIMAL, 2k-char prompt), while cluster calls succeed. The pipeline falls back to a summary composed in code from the cluster reports (marked in the text); `scripts/refresh_summary.py --run runs/<name>` retries Gemma later.
- `pip-audit -r requirements.txt`: no known vulnerabilities (2026-10-24).
- Hosted demo (Streamlit Community Cloud) reads `runs/` from the repo, so `.gitignore` now also lets `runs/m5_features` (real M5 data, 7.8 MB) through next to `runs/synth_features`. Expert reviews saved on the hosted app are shared by all visitors and reset when the app restarts.
- FAQ page (`faq.py`, via `st.navigation` with a hidden menu): linked at the bottom of the sidebar, offers `docs/paper/main.pdf` and a zip of the LaTeX sources, and shows the `docs/figures` PNGs.
