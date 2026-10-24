import numpy as np
import pandas as pd

from atlas.residuals import compute_scaled_residuals, make_windows, window_matrix
from atlas.schema import Dataset

SEASON = 2
A_HIST = [10, 12, 11, 15, 13, 14, 20, 18]  # seasonal diffs 1,3,2,1,7,4 -> mean 3
B_HIST = [5, 7, 4]  # < 2*season history -> lag-1 diffs 2,3 -> mean 2.5
A_LABELS = ["healthy", "delayed_response", "delayed_response", "delayed_response"] + ["healthy"] * 5
B_LABELS = ["missed_peak"] + ["healthy"] * 4


def make_dataset(with_labels: bool = True) -> Dataset:
    """Series A: 8 history + 9 forecast rows (r=+6). Series B: 3 history + 5 forecast rows (r=-2.5)."""
    a_fc = 30.0 + np.arange(9)
    a = pd.DataFrame(
        {
            "series_id": "A",
            "ts": pd.date_range("2024-01-01", periods=17, freq="D"),
            "actual": np.r_[A_HIST, a_fc].astype(float),
            "predicted": np.r_[np.array(A_HIST, dtype=float), a_fc - 6.0],
            "is_forecast": [False] * 8 + [True] * 9,
        }
    )
    b_fc = 10.0 + np.arange(5)
    b = pd.DataFrame(
        {
            "series_id": "B",
            "ts": pd.date_range("2024-03-01", periods=8, freq="D"),
            "actual": np.r_[B_HIST, b_fc].astype(float),
            "predicted": np.r_[[np.nan] * 3, b_fc + 2.5],
            "is_forecast": [False] * 3 + [True] * 5,
        }
    )
    if with_labels:
        a["true_label"] = ["history"] * 8 + A_LABELS
        b["true_label"] = ["history"] * 3 + B_LABELS
    data = pd.concat([b, a], ignore_index=True)  # deliberately unsorted: B before A
    meta = pd.DataFrame({"series_id": ["A", "B"], "model_card": ["m", "m"]})
    return Dataset(data=data, series_meta=meta)


def test_scale_is_seasonal_naive_mae_on_history():
    res = compute_scaled_residuals(make_dataset(), season=SEASON)
    assert (res.loc[res.series_id == "A", "scale"] == 3.0).all()
    assert np.isclose(res.loc[res.series_id == "A", "hist_mean"], 113 / 8).all()


def test_scale_falls_back_to_lag1_for_short_history():
    res = compute_scaled_residuals(make_dataset(), season=SEASON)
    assert (res.loc[res.series_id == "B", "scale"] == 2.5).all()


def test_scale_guard_for_flat_series():
    n_hist, n_fc = 10, 4
    data = pd.DataFrame(
        {
            "series_id": "F",
            "ts": pd.date_range("2024-01-01", periods=n_hist + n_fc, freq="D"),
            "actual": 100.0,
            "predicted": 100.0,
            "is_forecast": [False] * n_hist + [True] * n_fc,
        }
    )
    ds = Dataset(data, pd.DataFrame({"series_id": ["F"], "model_card": ["m"]}))
    res = compute_scaled_residuals(ds, season=SEASON)
    assert np.allclose(res["scale"], 5.0)  # 0.05 * mean|a|, seasonal MAE is 0


def test_frame_layout_and_order():
    res = compute_scaled_residuals(make_dataset(), season=SEASON)
    assert list(res.columns) == [
        "series_id", "ts", "is_forecast", "actual", "predicted", "step", "fstep",
        "scale", "hist_mean", "r_s", "a_s", "p_s", "true_label",
    ]
    assert len(res) == 25
    assert list(res.index) == list(range(25))
    assert list(res.series_id.unique()) == ["A", "B"]
    for _, g in res.groupby("series_id"):
        assert g.ts.is_monotonic_increasing
        assert list(g.step) == list(range(len(g)))
    assert "true_label" not in compute_scaled_residuals(make_dataset(with_labels=False), season=SEASON)


def test_residual_sign_and_scaling():
    res = compute_scaled_residuals(make_dataset(), season=SEASON)
    a_fc = res[(res.series_id == "A") & res.is_forecast]
    assert np.allclose(a_fc.r_s, 2.0)  # actual above prediction -> positive: (30+i - (24+i)) / 3
    assert np.allclose(a_fc.a_s, (a_fc.actual - 113 / 8) / 3)
    assert np.allclose(a_fc.p_s, (a_fc.predicted - 113 / 8) / 3)
    b_fc = res[(res.series_id == "B") & res.is_forecast]
    assert np.allclose(b_fc.r_s, -1.0)  # over-prediction -> negative
    b_hist = res[(res.series_id == "B") & ~res.is_forecast]
    assert b_hist.r_s.isna().all() and b_hist.p_s.isna().all()  # predicted is NaN
    assert b_hist.a_s.notna().all()


def test_fstep_numbering():
    res = compute_scaled_residuals(make_dataset(), season=SEASON)
    for sid, n_hist, n_fc in [("A", 8, 9), ("B", 3, 5)]:
        g = res[res.series_id == sid]
        assert (g.fstep[:n_hist] == -1).all()
        assert list(g.fstep[n_hist:]) == list(range(n_fc))


def test_windows_count_ids_bounds_and_trailing_partial_dropped():
    res = compute_scaled_residuals(make_dataset(), season=SEASON)
    w = make_windows(res, window_len=4, stride=2)
    # A: 9 forecast rows -> starts 0,2,4 (start 6 would need 10 rows); B: 5 rows -> start 0 only
    assert list(w.columns) == ["window_id", "series_id", "start_idx", "end_idx", "start_ts", "end_ts", "true_label"]
    assert list(w.window_id) == ["A#0", "A#2", "A#4", "B#0"]
    assert list(w.series_id) == ["A", "A", "A", "B"]
    assert list(w.start_idx) == [0, 2, 4, 0]
    assert list(w.end_idx) == [4, 6, 8, 4]
    a_ts = pd.date_range("2024-01-01", periods=17, freq="D")
    assert w.start_ts[1] == a_ts[8 + 2] and w.end_ts[1] == a_ts[8 + 5]  # end_ts = last row of the window
    b_ts = pd.date_range("2024-03-01", periods=8, freq="D")
    assert w.start_ts[3] == b_ts[3] and w.end_ts[3] == b_ts[6]


def test_windows_skip_series_shorter_than_window():
    res = compute_scaled_residuals(make_dataset(), season=SEASON)
    w = make_windows(res, window_len=6, stride=3)  # B has only 5 forecast rows
    assert set(w.series_id) == {"A"}
    assert list(w.window_id) == ["A#0", "A#3"]


def test_window_true_label_majority_rule():
    res = compute_scaled_residuals(make_dataset(), season=SEASON)
    w = make_windows(res, window_len=4, stride=2)
    # A#0: 3/4 delayed; A#2: exactly 2/4 delayed (>= 50%); A#4: none; B#0: 1/4 missed_peak (< 50%)
    assert list(w.true_label) == ["delayed_response", "delayed_response", "healthy", "healthy"]


def test_windows_without_labels_have_no_true_label_column():
    res = compute_scaled_residuals(make_dataset(with_labels=False), season=SEASON)
    w = make_windows(res, window_len=4, stride=2)
    assert "true_label" not in w.columns


def test_window_matrix_plain_matches_forecast_rows():
    res = compute_scaled_residuals(make_dataset(), season=SEASON)
    w = make_windows(res, window_len=4, stride=2)
    m = window_matrix(res, w, "r_s")
    assert m.shape == (4, 4) and m.dtype == np.float64
    assert np.allclose(m[:3], 2.0) and np.allclose(m[3], -1.0)
    actual = window_matrix(res, w, "actual")
    assert list(actual[1]) == [32.0, 33.0, 34.0, 35.0]  # A forecast rows fstep 2..5


def test_window_matrix_before_context_pulls_history_and_nan_before_series_start():
    res = compute_scaled_residuals(make_dataset(), season=SEASON)
    w = make_windows(res, window_len=4, stride=2)
    m = window_matrix(res, w, "actual", before=3)
    assert m.shape == (4, 7)
    assert list(m[0]) == [14.0, 20.0, 18.0, 30.0, 31.0, 32.0, 33.0]  # last 3 history rows, then window
    assert list(m[1][:3]) == [18.0, 30.0, 31.0]  # context may start inside the forecast region
    # B#0 starts at step 3 and asks for 5 rows before: steps -2..2 -> two NaN, then B history
    mb = window_matrix(res, w, "actual", before=5)
    assert np.isnan(mb[3][:2]).all()
    assert list(mb[3][2:]) == [5.0, 7.0, 4.0, 10.0, 11.0, 12.0, 13.0]


def test_window_matrix_after_context_nan_past_series_end():
    res = compute_scaled_residuals(make_dataset(), season=SEASON)
    w = make_windows(res, window_len=4, stride=2)
    m = window_matrix(res, w, "actual", before=1, after=3)
    assert m.shape == (4, 8)
    # A#4 covers fsteps 4..7 -> last real row (fstep 8 = 38.0) is the first "after" value, then 2 NaN
    assert list(m[2][:5]) == [33.0, 34.0, 35.0, 36.0, 37.0]
    assert m[2][5] == 38.0 and np.isnan(m[2][6:]).all()
    # context never leaks across series: B#0 'after' is NaN past B's last row (fstep 4 = 14.0)
    assert m[3][5] == 14.0 and np.isnan(m[3][6:]).all()


def test_window_matrix_empty_windows():
    res = compute_scaled_residuals(make_dataset(), season=SEASON)
    w = make_windows(res, window_len=100, stride=10)
    assert len(w) == 0
    assert window_matrix(res, w, "r_s", before=2, after=1).shape == (0, 3)
