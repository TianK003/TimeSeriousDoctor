"""Handcrafted failure features.

Vectorized over windows (numpy, no per-window Python loops). All inputs are scaled
residual-frame matrices (SPEC section 7.5). NaN-safe: context outside a series and
missing predictions are ignored, and every output is finite (degenerate cases fall
back to a neutral value: correlations 0, var_ratio 1, dir_acc 0.5).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES: list[str] = [
    "abs_err", "bias", "best_lag", "lag_gain", "slope_corr",
    "peak_excess", "var_ratio", "sign_run", "r_trend", "dir_acc",
]

FEATURE_DOCS: dict[str, str] = {
    "abs_err": "Average size of the forecast error in the window, in units of the series' typical seasonal change. Overall severity.",
    "bias": "Average signed error (actual minus forecast). Positive = forecast too low (under-prediction), negative = forecast too high.",
    "best_lag": "Shift in time steps that best aligns the forecast with the actual curve. Positive = the forecast trails (arrives later than) the actual; 0 = aligned.",
    "lag_gain": "How much better the forecast matches the actual after the best shift than with no shift. Large = a clear delayed-response signature.",
    "slope_corr": "Correlation between the error and the actual's step-to-step change. Positive = errors appear when the actual is moving, as a lagging forecast produces.",
    "peak_excess": "Average error on the window's highest actual steps minus the average error elsewhere. Large positive = the forecast misses peaks.",
    "var_ratio": "Spread of the forecast divided by spread of the actual. Well below 1 = the forecast is flattened or over-smoothed; near 1 = similar variability.",
    "sign_run": "Longest streak of same-direction errors as a share of the window. Near 1 = the error is persistently one-sided (a level shift the model has not adapted to).",
    "r_trend": "Change of the error across the window (slope times window length). Positive = the error is growing, negative = shrinking.",
    "dir_acc": "Share of steps where the forecast moves in the same direction as the actual. 1 = tracks every up/down move, 0.5 = chance.",
}

_EPS_VAR = 1e-12  # variance below this counts as a constant series
_PEAK_SHARE = 0.15


def _masked_corr(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pearson correlation along the last axis over positions where both are finite.

    Constant series (or fewer than 3 valid positions) give 0, never NaN.
    """
    m = np.isfinite(x) & np.isfinite(y)
    cnt = m.sum(axis=1)
    safe = np.maximum(cnt, 1)
    xz = np.where(m, x, 0.0)
    yz = np.where(m, y, 0.0)
    dx = np.where(m, xz - xz.sum(axis=1, keepdims=True) / safe[:, None], 0.0)
    dy = np.where(m, yz - yz.sum(axis=1, keepdims=True) / safe[:, None], 0.0)
    cov = (dx * dy).sum(axis=1)
    vx = (dx * dx).sum(axis=1)
    vy = (dy * dy).sum(axis=1)
    ok = (cnt >= 3) & (vx > _EPS_VAR) & (vy > _EPS_VAR)
    denom = np.sqrt(np.where(ok, vx * vy, 1.0))
    return np.where(ok, cov / denom, 0.0)


def _nanmean(x: np.ndarray) -> np.ndarray:
    """Mean over finite entries along the last axis; 0 where there are none."""
    m = np.isfinite(x)
    cnt = m.sum(axis=1)
    return np.where(cnt > 0, np.where(m, x, 0.0).sum(axis=1) / np.maximum(cnt, 1), 0.0)


def _longest_same_sign_run(r: np.ndarray) -> np.ndarray:
    """Longest run of identical non-zero sign along the last axis (zero and NaN break runs)."""
    s = np.sign(np.nan_to_num(r, nan=0.0))
    run = (s[:, 0] != 0).astype(np.int64)
    best = run.copy()
    for t in range(1, s.shape[1]):
        nz = s[:, t] != 0
        run = np.where(nz & (s[:, t] == s[:, t - 1]), run + 1, nz.astype(np.int64))
        best = np.maximum(best, run)
    return best.astype(np.float64)


def _ols_slope(r: np.ndarray) -> np.ndarray:
    """OLS slope of r over t = 0..L-1 using finite entries only; 0 if undetermined."""
    t = np.broadcast_to(np.arange(r.shape[1], dtype=np.float64), r.shape)
    m = np.isfinite(r)
    cnt = m.sum(axis=1)
    safe = np.maximum(cnt, 1)
    tm = np.where(m, t, 0.0).sum(axis=1, keepdims=True) / safe[:, None]
    rm = np.where(m, r, 0.0).sum(axis=1, keepdims=True) / safe[:, None]
    dt = np.where(m, t - tm, 0.0)
    dr = np.where(m, r - rm, 0.0)
    den = (dt * dt).sum(axis=1)
    ok = (cnt >= 2) & (den > _EPS_VAR)
    return np.where(ok, (dt * dr).sum(axis=1) / np.where(ok, den, 1.0), 0.0)


def _peak_excess(r: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Mean r on the top-15 % rows of a (at least 1) minus mean r on the other rows."""
    n, L = a.shape
    k = max(1, int(round(_PEAK_SHARE * L)))
    order = np.argsort(np.where(np.isfinite(a), -a, np.inf), axis=1, kind="stable")
    peak = np.zeros((n, L), dtype=bool)
    np.put_along_axis(peak, order[:, :k], True, axis=1)
    peak &= np.isfinite(a)
    rf = np.isfinite(r)
    pm, om = peak & rf, ~peak & rf
    pc, oc = pm.sum(axis=1), om.sum(axis=1)
    r0 = np.where(rf, r, 0.0)
    peak_mean = (r0 * pm).sum(axis=1) / np.maximum(pc, 1)
    other_mean = (r0 * om).sum(axis=1) / np.maximum(oc, 1)
    return np.where((pc > 0) & (oc > 0), peak_mean - other_mean, 0.0)


def _var_ratio(p: np.ndarray, a: np.ndarray) -> np.ndarray:
    """std(p) / std(a) over positions where both are finite; 1.0 if std(a) < 1e-6."""
    m = np.isfinite(p) & np.isfinite(a)
    cnt = np.maximum(m.sum(axis=1), 1)

    def std(x: np.ndarray) -> np.ndarray:
        xz = np.where(m, x, 0.0)
        d = np.where(m, xz - xz.sum(axis=1, keepdims=True) / cnt[:, None], 0.0)
        return np.sqrt((d * d).sum(axis=1) / cnt)

    sa = std(a)
    return np.where(sa < 1e-6, 1.0, std(p) / np.where(sa < 1e-6, 1.0, sa))


def _dir_acc(p: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Share of steps t = 1..L-1 with sign(dp) == sign(da); 0.5 if no valid step."""
    dp, da = np.diff(p, axis=1), np.diff(a, axis=1)
    m = np.isfinite(dp) & np.isfinite(da)
    cnt = m.sum(axis=1)
    hit = ((np.sign(dp) == np.sign(da)) & m).sum(axis=1)
    return np.where(cnt > 0, hit / np.maximum(cnt, 1), 0.5)


def compute_features(
    r: np.ndarray, a: np.ndarray, p: np.ndarray, a_ctx: np.ndarray, max_lag: int = 7
) -> pd.DataFrame:
    """Failure features for n windows.

    `r`, `a`, `p` are [n, L] window matrices of r_s, a_s, p_s; `a_ctx` is [n, L + 2*max_lag]
    (a_s from `max_lag` rows before to `max_lag` rows after the window, NaN outside the series,
    so `a_ctx[:, max_lag + t]` is `a[:, t]`). Returns a frame with columns FEATURES, one row
    per window, all finite.
    """
    r, a, p, a_ctx = (np.asarray(x, dtype=np.float64) for x in (r, a, p, a_ctx))
    n, L = r.shape
    if max_lag < 1:
        raise ValueError("max_lag must be >= 1")
    if a.shape != (n, L) or p.shape != (n, L):
        raise ValueError(f"r, a, p must share shape [n, L]; got {r.shape}, {a.shape}, {p.shape}")
    if a_ctx.shape != (n, L + 2 * max_lag):
        raise ValueError(f"a_ctx must be [n, L + 2*max_lag] = {(n, L + 2 * max_lag)}; got {a_ctx.shape}")

    # lag: corr(p_t, a_{t-k}) for k in [-max_lag, max_lag]; a_{t-k} = a_ctx[:, max_lag + t - k].
    # Candidates are ordered by |k| ascending so argmax's first-hit rule breaks ties to the smallest |k|.
    lags = np.array([0] + [s * k for k in range(1, max_lag + 1) for s in (1, -1)])
    corrs = np.empty((n, lags.size))
    for j, k in enumerate(lags):
        start = max_lag - int(k)
        corrs[:, j] = _masked_corr(p, a_ctx[:, start:start + L])
    best = corrs.argmax(axis=1)
    best_lag = lags[best].astype(np.float64)
    lag_gain = corrs[np.arange(n), best] - corrs[:, 0]

    # slope_corr: corr(r_t, a_t - a_{t-1}); a_{-1} comes from the context.
    a_prev = np.concatenate([a_ctx[:, max_lag - 1:max_lag], a[:, :-1]], axis=1)

    out = {
        "abs_err": _nanmean(np.abs(r)),
        "bias": _nanmean(r),
        "best_lag": best_lag,
        "lag_gain": lag_gain,
        "slope_corr": _masked_corr(r, a - a_prev),
        "peak_excess": _peak_excess(r, a),
        "var_ratio": _var_ratio(p, a),
        "sign_run": _longest_same_sign_run(r) / L,
        "r_trend": _ols_slope(r) * L,
        "dir_acc": _dir_acc(p, a),
    }
    return pd.DataFrame({name: out[name] for name in FEATURES})
