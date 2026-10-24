"""Signatures of known failure types on handcrafted features (SPEC 7.5)."""

import time

import numpy as np
import pandas as pd

from atlas.features import FEATURE_DOCS, FEATURES, compute_features

L = 28
ML = 7
N = 200


def _signal(n: int, seed: int = 0) -> np.ndarray:
    """Smooth seasonal (period 12, not 7, so lag 4 has no in-range alias) + ramp + slow wander.

    Returns [n, L + 4*ML]: window starts at column 2*ML, so there are 2*ML columns of context
    on each side (enough for a_ctx and for building lagged predictions).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(L + 4 * ML)[None, :]
    phase = rng.uniform(0, 2 * np.pi, (n, 1))
    amp = rng.uniform(0.8, 1.6, (n, 1))
    slope = rng.uniform(0.0, 0.06, (n, 1))
    wander = np.cumsum(rng.normal(0, 0.12, (n, t.shape[1])), axis=1)
    wander = np.convolve(wander.ravel(), np.ones(3) / 3, mode="same").reshape(n, -1)
    return amp * np.sin(2 * np.pi * t / 12 + phase) + slope * t + wander


def _views(sig: np.ndarray, lag: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(window, a_ctx) from sig; `lag` shifts the window start back (lag > 0 = earlier values)."""
    s0 = 2 * ML - lag
    return sig[:, s0:s0 + L], sig[:, 2 * ML - ML:2 * ML - ML + L + 2 * ML]


def _run(a: np.ndarray, p: np.ndarray, a_ctx: np.ndarray) -> pd.DataFrame:
    return compute_features(a - p, a, p, a_ctx, max_lag=ML)


def test_columns_and_docs() -> None:
    assert FEATURES == ["abs_err", "bias", "best_lag", "lag_gain", "slope_corr",
                        "peak_excess", "var_ratio", "sign_run", "r_trend", "dir_acc",
                        "r_acf_season", "spike_ratio"]
    assert set(FEATURE_DOCS) == set(FEATURES)
    sig = _signal(10)
    a, a_ctx = _views(sig)
    f = _run(a, a.copy(), a_ctx)
    assert list(f.columns) == FEATURES
    assert len(f) == 10
    assert np.isfinite(f.to_numpy()).all()


def test_delayed_response_has_positive_lag_and_gain() -> None:
    sig = _signal(N, seed=1)
    rng = np.random.default_rng(11)
    a, a_ctx = _views(sig)
    p, _ = _views(sig, lag=4)  # p_t = a_{t-4}
    p = p + rng.normal(0, 0.05, p.shape)
    f = _run(a, p, a_ctx)
    assert f["best_lag"].isin([3, 4, 5]).mean() > 0.95
    assert f["best_lag"].median() == 4
    assert (f["lag_gain"] > 0.1).mean() > 0.9
    assert f["slope_corr"].median() > 0.3  # residual proportional to slope
    assert f["dir_acc"].median() < 0.9


def test_unadapted_shift_is_one_sided() -> None:
    sig = _signal(N, seed=2)
    rng = np.random.default_rng(12)
    p, a_ctx = _views(sig)
    a = p + 5.0  # actual stepped up, forecast stayed at the old level
    p = p + rng.normal(0, 0.2, p.shape)
    a_ctx = a_ctx + 5.0
    f = _run(a, p, a_ctx)
    assert (f["sign_run"] >= 0.8).all()
    assert (f["bias"].abs() > 2).all()
    assert (f["bias"] > 0).all()


def test_over_smoothing_has_low_var_ratio() -> None:
    sig = _signal(N, seed=3)
    a, a_ctx = _views(sig)
    f = _run(a, 0.2 * a, a_ctx)
    assert (f["var_ratio"] < 0.6).all()
    np.testing.assert_allclose(f["var_ratio"], 0.2, atol=1e-9)


def test_missed_peaks_have_peak_excess() -> None:
    sig = _signal(N, seed=4)
    rng = np.random.default_rng(14)
    p, a_ctx = _views(sig)
    p = p + rng.normal(0, 0.05, p.shape)
    a = _views(sig)[0].copy()
    for i in range(N):
        a[i, rng.choice(L, 3, replace=False)] += 5.0  # spikes the forecast never sees
    f = _run(a, p, a_ctx)
    assert (f["peak_excess"] > 1).mean() > 0.95
    assert f["peak_excess"].median() > 2


def test_weekly_oscillating_residual_has_high_seasonal_acf() -> None:
    rng = np.random.default_rng(21)
    sig = _signal(N, seed=8)
    a, a_ctx = _views(sig)
    phase = rng.uniform(0, 2 * np.pi, (N, 1))
    weekly = 3.0 * np.sin(2 * np.pi * np.arange(L)[None, :] / 7 + phase)
    f = compute_features(weekly + rng.normal(0, 0.3, a.shape), a, a - weekly, a_ctx, max_lag=ML)
    assert (f["r_acf_season"] > 0.7).mean() > 0.95
    assert f["r_acf_season"].max() <= 1.0
    # a different season length is picked up through the `season` argument
    f14 = compute_features(weekly, a, a - weekly, a_ctx, max_lag=ML, season=14)
    assert (f14["r_acf_season"] > 0.7).all()  # period 7 repeats at lag 14 too
    f5 = compute_features(weekly, a, a - weekly, a_ctx, max_lag=ML, season=5)
    assert (f5["r_acf_season"] < 0.0).all()


def test_sparse_spikes_have_low_seasonal_acf_and_high_spike_ratio() -> None:
    rng = np.random.default_rng(22)
    sig = _signal(N, seed=9)
    a, a_ctx = _views(sig)
    r = rng.normal(0, 0.3, a.shape)
    for i in range(N):
        r[i, rng.choice(L, 3, replace=False)] += 6.0  # three sparse spikes
    f = compute_features(r, a, a - r, a_ctx, max_lag=ML)
    assert f["spike_ratio"].median() > 4.0
    assert f["r_acf_season"].median() < 0.4

    smooth = 3.0 * np.sin(2 * np.pi * np.arange(L)[None, :] / 7 + rng.uniform(0, 6.28, (N, 1)))
    fs = compute_features(smooth, a, a - smooth, a_ctx, max_lag=ML)
    assert fs["spike_ratio"].max() < 2.0  # sinusoid: max/mean = pi/2
    assert f["spike_ratio"].median() > 2 * fs["spike_ratio"].median()


def test_spike_ratio_and_acf_exact_values() -> None:
    a = np.zeros((2, L))
    a_ctx = np.zeros((2, L + 2 * ML))
    r = np.zeros((2, L))
    r[0, 10] = 4.0                 # one spike: max 4, mean 4/28
    r[1] = np.tile(np.arange(7.0), 4)  # period-7 sawtooth: acf at lag 7 is exactly 1
    f = compute_features(r, a, a - r, a_ctx, max_lag=ML)
    np.testing.assert_allclose(f["spike_ratio"].iloc[0], 4.0 / (4.0 / L + 1e-6))
    np.testing.assert_allclose(f["spike_ratio"].iloc[1], 6.0 / (3.0 + 1e-6))
    np.testing.assert_allclose(f["r_acf_season"].iloc[1], 1.0)
    assert f["r_acf_season"].iloc[0] < 0.0  # lone spike: negative-ish, never high


def test_healthy_noise_has_small_error_and_no_signature() -> None:
    sig = _signal(N, seed=5)
    rng = np.random.default_rng(15)
    a, a_ctx = _views(sig)
    p = a + rng.normal(0, 0.35, a.shape)
    f = _run(a, p, a_ctx)
    assert (f["abs_err"] < 0.5).all()
    assert f["best_lag"].median() == 0
    assert f["bias"].abs().max() < 0.4
    assert (f["sign_run"] < 0.5).all()


def test_nan_context_and_missing_values_stay_finite() -> None:
    sig = _signal(50, seed=6)
    rng = np.random.default_rng(16)
    a, a_ctx = _views(sig)
    a_ctx = a_ctx.copy()
    a_ctx[:20, :ML] = np.nan          # window at the start of a series: no left context
    a_ctx[20:40, -ML:] = np.nan       # window at the end: no right context
    a_ctx[40:, :] = np.nan            # no context at all...
    a_ctx[40:, ML:ML + L] = a[40:]    # ...except the window itself
    p = a + rng.normal(0, 0.3, a.shape)
    r = a - p
    f = compute_features(r, a, p, a_ctx, max_lag=ML)
    assert np.isfinite(f.to_numpy()).all()

    # missing predictions (r and p NaN) are ignored as well
    p2, r2 = p.copy(), r.copy()
    p2[:, 5:9] = np.nan
    r2[:, 5:9] = np.nan
    f2 = compute_features(r2, a, p2, a_ctx, max_lag=ML)
    assert np.isfinite(f2.to_numpy()).all()
    # entirely NaN predictions give neutral values rather than NaN
    f3 = compute_features(np.full_like(r, np.nan), a, np.full_like(p, np.nan), a_ctx, max_lag=ML)
    assert np.isfinite(f3.to_numpy()).all()
    assert (f3["r_acf_season"] == 0).all() and (f3["spike_ratio"] == 0).all()
    # a window shorter than the season gives a neutral acf
    f4 = compute_features(r[:, :5], a[:, :5], p[:, :5], a_ctx[:, :5 + 2 * ML], max_lag=ML)
    assert (f4["r_acf_season"] == 0).all() and np.isfinite(f4.to_numpy()).all()


def test_constant_series_gives_neutral_values() -> None:
    n = 5
    a = np.full((n, L), 3.0)
    a_ctx = np.full((n, L + 2 * ML), 3.0)
    p = np.full((n, L), 3.0)
    f = compute_features(a - p, a, p, a_ctx, max_lag=ML)
    assert np.isfinite(f.to_numpy()).all()
    assert (f["best_lag"] == 0).all()
    assert (f["lag_gain"] == 0).all()
    assert (f["slope_corr"] == 0).all()
    assert (f["var_ratio"] == 1.0).all()
    assert (f["sign_run"] == 0).all()  # all-zero residual: zeros break runs
    assert (f["r_acf_season"] == 0).all() and (f["spike_ratio"] == 0).all()


def test_exact_values_on_tiny_window() -> None:
    # L = 8, max_lag = 2: hand-checkable definitions
    L8, ml = 8, 2
    a = np.arange(8, dtype=float)[None, :]            # ramp 0..7
    a_ctx = np.arange(-2, 10, dtype=float)[None, :]   # ramp continued on both sides
    r = np.array([[1, 1, 1, -1, -1, 0, 1, 1]], dtype=float)
    p = a - r
    f = compute_features(r, a, p, a_ctx, max_lag=ml).iloc[0]
    assert f["abs_err"] == 7 / 8
    assert f["bias"] == 3 / 8
    assert f["sign_run"] == 3 / 8  # runs: 3, 2, (0 breaks), 2
    # peak = top round(0.15*8)=1 row -> t=7 (r=1); others mean = (1+1+1-1-1+0+1)/7 = 2/7
    np.testing.assert_allclose(f["peak_excess"], 1 - 2 / 7)
    # OLS slope of r over t (hand-computed), times L
    t = np.arange(8)
    slope = np.polyfit(t, r[0], 1)[0]
    np.testing.assert_allclose(f["r_trend"], slope * L8)
    # dp = da - dr = 1 - dr = 1, 1, 3, 1, 0, 0, 1; da = +1 always -> 2 of 7 steps disagree (dp == 0)
    np.testing.assert_allclose(f["dir_acc"], 5 / 7)


def test_lag_sign_convention() -> None:
    # prediction trails actual by 3: best_lag = +3; prediction leads: -3
    sig = _signal(N, seed=7)
    a, a_ctx = _views(sig)
    trailing, _ = _views(sig, lag=3)
    leading, _ = _views(sig, lag=-3)
    assert (_run(a, trailing, a_ctx)["best_lag"] == 3).all()
    assert (_run(a, leading, a_ctx)["best_lag"] == -3).all()


def test_speed_50k_windows() -> None:
    rng = np.random.default_rng(0)
    n = 50_000
    r = rng.normal(size=(n, L))
    a = rng.normal(size=(n, L))
    p = rng.normal(size=(n, L))
    a_ctx = rng.normal(size=(n, L + 2 * ML))
    t0 = time.perf_counter()
    f = compute_features(r, a, p, a_ctx, max_lag=ML)
    dt = time.perf_counter() - t0
    assert len(f) == n
    assert np.isfinite(f.to_numpy()).all()
    assert dt < 30.0
    print(f"compute_features on {n} windows: {dt:.2f}s")
