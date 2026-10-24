# TimeSeriousDoctor

Forecasting teams judge models with aggregate metrics (MAE, RMSE, MASE), and those numbers hide *how*
a model fails: lagging behind changes, missing peaks, ignoring level shifts, over-smoothing. TimeSeriousDoctor
takes actuals and predictions for thousands of series, computes scaled residuals, keeps only the windows
where the error is abnormal, clusters them into recurring failure patterns, and asks Gemma 4 to name and
explain each pattern: what happens, likely causes, what to try, and which metric to monitor. Everything is
shown in a Streamlit "atlas". **Code measures, Gemma explains**: every number comes from code, and Gemma
only interprets the numbers and charts it is given.

> SPEC.md calls this project "Residual Atlas" (the Python package is still named `atlas`).

## Quick start

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

Then set `GEMMA4_API_KEY` in `.env` (`GEMINI_API_KEY` also works) and run the tests:

```
pytest
```

Planned commands (coming during the hackathon):

```
python scripts/make_synthetic.py --n 400 --out data/synthetic
python scripts/run_atlas.py --data data/synthetic --embedder features --run-name synth_features
streamlit run app.py
```

## More

- Architecture, module ownership and build priorities: see [SPEC.md](SPEC.md).
- Models: Gemma 4 via the Gemini API (explanations); MOMENT-1 time-series embeddings (optional, `requirements-moment.txt`).
- License: Apache-2.0, see [LICENSE](LICENSE).
