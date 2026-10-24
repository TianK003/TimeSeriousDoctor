import numpy as np
import pandas as pd
import pytest

from atlas.data.synthetic import DELAYS, FAILURE_TYPES, LABELS, MODEL_CARDS, make_synthetic
from atlas.schema import DATA_REQUIRED_COLUMNS, META_REQUIRED_COLUMNS, Dataset, load_dataset, save_dataset

HISTORY, FORECAST = 120, 182


@pytest.fixture(scope="module")
def synth() -> tuple[pd.DataFrame, pd.DataFrame]:
    return make_synthetic(n_series=100, history_len=HISTORY, forecast_len=FORECAST, seed=3)


def series_types(data: pd.DataFrame) -> dict[str, str]:
    """True type per series, read from the row labels."""
    out = {}
    for sid, s in data.groupby("series_id"):
        failing = set(s["true_label"]) - {"history", "healthy"}
        assert len(failing) <= 1, f"{sid} carries several failure types: {failing}"
        out[sid] = failing.pop() if failing else "healthy"
    return out


def segment(s: pd.DataFrame, label: str) -> slice:
    """Positional slice of the rows labeled `label` (asserts they are contiguous)."""
    idx = np.flatnonzero((s["true_label"] == label).to_numpy())
    assert len(idx) > 0 and idx[-1] - idx[0] + 1 == len(idx), "failure rows are not one contiguous run"
    return slice(idx[0], idx[-1] + 1)


def test_contract_shapes_and_columns(synth):
    data, meta = synth
    assert list(data.columns) == DATA_REQUIRED_COLUMNS + ["true_label"]
    assert list(meta.columns) == META_REQUIRED_COLUMNS
    assert len(data) == 100 * (HISTORY + FORECAST)
    assert data["series_id"].nunique() == 100 and meta["series_id"].is_unique
    assert list(meta["series_id"]) == [f"s{i:04d}" for i in range(100)]
    assert set(meta["model_card"]) <= set(MODEL_CARDS)
    assert pd.api.types.is_datetime64_any_dtype(data["ts"])
    assert data["is_forecast"].dtype == bool

    for _, s in data.groupby("series_id"):
        assert s["ts"].iloc[0] == pd.Timestamp("2023-01-01")
        assert (s["ts"].diff().dropna() == pd.Timedelta(days=1)).all()
        hist, fc = s[~s["is_forecast"]], s[s["is_forecast"]]
        assert len(hist) == HISTORY and len(fc) == FORECAST
        assert s["is_forecast"].tolist() == [False] * HISTORY + [True] * FORECAST
        assert hist["predicted"].isna().all() and (hist["true_label"] == "history").all()
        assert fc["predicted"].notna().all() and not (fc["true_label"] == "history").any()
        assert s["actual"].notna().all()


def test_load_dataset_accepts_output(synth, tmp_path):
    data, meta = synth
    save_dataset(Dataset(data=data, series_meta=meta), tmp_path / "ds")
    ds = load_dataset(tmp_path / "ds")
    assert len(ds.data) == len(data) and len(ds.series_meta) == len(meta)
    assert "true_label" in ds.data.columns
    assert ds.data["series_id"].nunique() == 100


@pytest.mark.parametrize(
    "n_series, mix, expected",
    [
        (100, None, {t: 20 for t in LABELS}),
        (
            50,
            {"healthy": 0.4, "delayed_response": 0.2, "missed_peak": 0.2, "unadapted_shift": 0.1, "over_smoothing": 0.1},
            {"healthy": 20, "delayed_response": 10, "missed_peak": 10, "unadapted_shift": 5, "over_smoothing": 5},
        ),
        (30, {"healthy": 0.5, "unadapted_shift": 0.5}, {"healthy": 15, "unadapted_shift": 15}),
    ],
)
def test_class_counts_exact(n_series, mix, expected):
    data, meta = make_synthetic(n_series=n_series, history_len=HISTORY, forecast_len=FORECAST, seed=1, mix=mix)
    counts = pd.Series(series_types(data)).value_counts().to_dict()
    assert counts == expected
    assert len(meta) == n_series


def test_class_counts_sum_when_not_divisible():
    data, _ = make_synthetic(n_series=13, history_len=HISTORY, forecast_len=FORECAST, seed=2)
    counts = pd.Series(series_types(data)).value_counts()
    assert counts.sum() == 13 and counts.min() >= 2 and counts.max() <= 3


def test_types_are_shuffled_and_cards_independent(synth):
    data, meta = synth
    types = series_types(data)
    order = [types[sid] for sid in meta["series_id"]]
    assert order != sorted(order)  # not grouped by class
    # model card carries no information about the label: every type sees several different cards
    cards = meta.set_index("series_id")["model_card"]
    for t in LABELS:
        assert cards[[sid for sid, ty in types.items() if ty == t]].nunique() >= 3


def test_labels_form_one_contiguous_forecast_segment(synth):
    data, _ = synth
    for sid, s in data.groupby("series_id"):
        s = s.reset_index(drop=True)
        labels = s["true_label"]
        assert (labels[:HISTORY] == "history").all()
        failing = set(labels) - {"history", "healthy"}
        if not failing:
            assert (labels[HISTORY:] == "healthy").all()
            continue
        (label,) = failing
        seg = segment(s, label)
        assert seg.start >= HISTORY, "segment starts inside history"
        assert 42 <= seg.stop - seg.start <= 91
        assert (labels[HISTORY:seg.start] == "healthy").all() and (labels[seg.stop:] == "healthy").all()


def test_injected_failures_are_visible(synth):
    """Residuals inside the segment dwarf the healthy noise outside it, and have the intended signature."""
    data, _ = synth
    types = series_types(data)
    min_ratio = {}
    for sid, label in types.items():
        if label == "healthy":
            continue
        s = data[data["series_id"] == sid].reset_index(drop=True)
        seg = segment(s, label)
        r =(s["actual"] - s["predicted"]).to_numpy()
        inside = np.zeros(len(s), dtype=bool)
        inside[seg] = True
        outside = (s["is_forecast"].to_numpy()) & ~inside
        ratio = np.abs(r[inside]).mean() / np.abs(r[outside]).mean()
        min_ratio[label] = min(min_ratio.get(label, np.inf), ratio)

        if label == "unadapted_shift":  # persistent bias, then healthy catch-up
            assert abs(r[inside].mean()) > 10 * abs(r[outside].mean())
            assert np.sign(r[inside]).mean() * np.sign(r[inside].mean()) > 0.9
        elif label == "missed_peak":  # sparse positive spikes the model never sees
            assert r[inside].max() > 5 * r[outside].std()
            assert (r[inside] > 2.5 * r[outside].std()).sum() >= 4
        elif label == "delayed_response":  # p_t is a lagged copy of a_t, lag in DELAYS, never 7
            a, p = s["actual"].to_numpy(), s["predicted"].to_numpy()
            idx = np.arange(seg.start, seg.stop)
            err = {k: np.abs(p[idx] - a[idx - k]).mean() for k in range(1, 9)}
            assert min(err, key=err.get) in DELAYS
            assert err[min(err, key=err.get)] < 0.5 * np.abs(p[idx] - a[idx]).mean()
    assert set(min_ratio) == set(FAILURE_TYPES)
    for label, ratio in min_ratio.items():
        assert ratio > 1.5, f"{label}: weakest injected series is only {ratio:.2f}x the healthy error"


def test_healthy_error_level(synth):
    """Healthy forecast rows: p = a + N(0, 0.35 sigma), i.e. scaled abs error near 0.25."""
    data, _ = synth
    types = series_types(data)
    errs = []
    for sid, label in types.items():
        if label != "healthy":
            continue
        s = data[data["series_id"] == sid]
        hist_a = s.loc[~s["is_forecast"], "actual"].to_numpy()
        scale = np.abs(hist_a[7:] - hist_a[:-7]).mean()
        fc = s[s["is_forecast"]]
        errs.append(np.abs(fc["actual"] - fc["predicted"]).mean() / scale)
    assert 0.18 < np.mean(errs) < 0.35


def test_same_seed_identical_and_other_seed_differs():
    kw = dict(n_series=10, history_len=HISTORY, forecast_len=FORECAST)
    d1, m1 = make_synthetic(seed=7, **kw)
    d2, m2 = make_synthetic(seed=7, **kw)
    pd.testing.assert_frame_equal(d1, d2)
    pd.testing.assert_frame_equal(m1, m2)
    d3, _ = make_synthetic(seed=8, **kw)
    assert not d1["actual"].equals(d3["actual"])


def test_unknown_mix_key_rejected():
    with pytest.raises(ValueError, match="unknown types"):
        make_synthetic(n_series=5, history_len=HISTORY, forecast_len=FORECAST, mix={"healthy": 0.5, "bogus": 0.5})
