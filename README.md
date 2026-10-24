# TimeSeriousDoctor

> Find the recurring ways forecasts fail across thousands of time series, group them automatically,
> and have Gemma 4 explain each failure pattern like a forecasting expert would.

## What it is and who it is for

For forecasting and data-science teams who run hundreds to thousands of forecasts. Aggregate metrics
(MAE, RMSE, MASE) say *how much* a model is wrong, not *how*. Lagging behind changes, missing peaks,
ignoring level shifts and over-smoothing are usually found by experts eyeballing plots, and that doesn't
scale.

TimeSeriousDoctor takes actuals and predictions for many series and:

1. computes scaled residuals;
2. keeps only the windows where the error is abnormal;
3. describes each window with expert failure features (and optionally MOMENT embeddings);
4. clusters the windows into recurring failure patterns;
5. asks **Gemma 4** to diagnose each cluster from a chart panel plus its numbers: what happens, likely causes, what to try, and which metric to monitor.

Everything is shown in a Streamlit "atlas". **Code measures, Gemma explains**: every number comes from
code, and Gemma only interprets the numbers and charts it is given.

> SPEC.md calls this project "Residual Atlas" (the Python package is named `atlas`).

## How to run it

Windows:

```
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

POSIX:

```
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set `GEMMA4_API_KEY` in `.env` (`GEMINI_API_KEY` also works). Then:

```
pytest                                    # tests, no network
python scripts/smoke_gemini.py            # optional: check your key against the Gemma 4 API
```

Demo pipeline (being built during the hackathon):

```
python scripts/make_synthetic.py --n 400 --out data/synthetic
python scripts/run_atlas.py --data data/synthetic --embedder features --run-name synth_features
python scripts/run_eval.py --runs runs/synth_features
streamlit run app.py
```

## Expertise: what knowledge is built in and where it comes from

- **Failure taxonomy** (`taxonomy.yaml`) has five failure types: delayed response, missed peak, unadapted shift, over-smoothing, healthy noise. Each one has a definition, a numeric signature, likely causes, fixes, and a metric to monitor. Our time-series expert wrote it from forecasting practice and refines it after seeing real clusters. It goes into every Gemma prompt.
- **Handcrafted failure features** (`atlas/features.py`) encode how an expert recognizes each failure: lag cross-correlation, peak excess, variance ratio, sign runs, error trend, directional accuracy.
- **MASE-style scaling** (Hyndman & Koehler, 2006) puts residuals on a common scale, where 1.0 is about a typical seasonal-naive error. This makes thresholds and clusters comparable across series.

## The model and its license

- **Gemma 4**: `gemma-4-26b-a4b-it` (fallback `gemma-4-31b-it`), called through the **Gemini API** with the `google-genai` SDK. Gemma 4 is released as open weights under [Apache 2.0](https://opensource.googleblog.com/2026/03/gemma-4-expanding-the-gemmaverse-with-apache-20.html), and API use is subject to the [Gemini API Terms of Service](https://ai.google.dev/gemini-api/terms). The integration lives in `atlas/gemma.py` and `atlas/prompts.py`: multimodal chart + numbers in, schema-validated JSON out, with disk cache and throttling.
- **MOMENT-1** (optional encoder): [AutonLab/MOMENT-1-small](https://huggingface.co/AutonLab/MOMENT-1-small) on Hugging Face. Its license is on the model card.

## Proof: better than the same model without it

_To be filled from `scripts/run_eval.py`._ Same Gemma 4 model, same taxonomy, same series. **Gemma alone**
looks at each full-series chart; **TimeSeriousDoctor** runs Gemma inside the pipeline.

| method | accuracy | macro-F1 | Gemma calls (whole dataset) |
|---|---|---|---|
| Gemma alone | – | – | 1 per series |
| TimeSeriousDoctor | – | – | 1 per cluster + 1 |

## More

- Architecture, module ownership and build priorities: see [SPEC.md](SPEC.md). Decisions and verified API facts: see [NOTES.md](NOTES.md).
- Disclaimer: automated diagnostics are suggestions to investigate, not guarantees and not financial advice. Verify before changing production models.
- License: Apache-2.0, see [LICENSE](LICENSE).
