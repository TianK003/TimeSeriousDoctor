"""Scaled residuals and windowing.

Spec: SPEC.md §7.3. Forecast rows are assumed to form the tail of each series (history first).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from atlas.schema import Dataset

HEALTHY = "healthy"
HISTORY = "history"


def compute_scaled_residuals(ds: Dataset, season: int = 7) -> pd.DataFrame:
    """Build the residual frame `res`: one row per input row, sorted by series_id, ts.

    r = actual - predicted (positive = under-prediction). Per-series `scale` is the seasonal-naive
    MAE on history rows (lag-1 fallback if history < 2*season), floored at
    max(1e-6, 0.05 * mean|actual| over history). Scaled columns: r_s, a_s, p_s.
    """
    df = ds.data.sort_values(["series_id", "ts"], kind="stable").reset_index(drop=True)
    sid = df["series_id"]
    actual = df["actual"].astype(float)
    predicted = df["predicted"].astype(float)
    is_fc = df["is_forecast"].astype(bool)
    hist = ~is_fc

    step = df.groupby("series_id", sort=False).cumcount()
    fstep = (is_fc.astype(int).groupby(sid, sort=False).cumsum() - 1).where(is_fc, -1)

    grp = actual.groupby(sid, sort=False)
    mae_season = grp.diff(season).abs().where(hist).groupby(sid, sort=False).transform("mean")
    mae_lag1 = grp.diff(1).abs().where(hist).groupby(sid, sort=False).transform("mean")
    n_hist = hist.astype(int).groupby(sid, sort=False).transform("sum")
    scale = mae_season.where(n_hist >= 2 * season, mae_lag1)
    mean_abs = actual.abs().where(hist).groupby(sid, sort=False).transform("mean")
    hist_mean = actual.where(hist).groupby(sid, sort=False).transform("mean")
    scale = np.fmax(np.fmax(scale.to_numpy(), 1e-6), 0.05 * mean_abs.to_numpy())

    res = pd.DataFrame(
        {
            "series_id": sid,
            "ts": df["ts"],
            "is_forecast": is_fc,
            "actual": actual,
            "predicted": predicted,
            "step": step.astype(int),
            "fstep": fstep.astype(int),
            "scale": scale,
            "hist_mean": hist_mean.to_numpy(),
        }
    )
    res["r_s"] = (actual - predicted) / scale
    res["a_s"] = (actual - res["hist_mean"]) / scale
    res["p_s"] = (predicted - res["hist_mean"]) / scale
    if "true_label" in df.columns:
        res["true_label"] = df["true_label"]
    return res


def _layout(res: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per series (res is sorted, series contiguous): ids, flat offset, n rows, n forecast rows."""
    sid = res["series_id"].to_numpy()
    if len(sid) == 0:
        empty = np.zeros(0, dtype=np.int64)
        return sid, empty, empty, empty
    off = np.flatnonzero(np.r_[True, sid[1:] != sid[:-1]])
    n = np.diff(np.r_[off, len(sid)])
    nf = np.add.reduceat(res["is_forecast"].to_numpy().astype(np.int64), off)
    return sid[off], off, n, nf


def make_windows(res: pd.DataFrame, window_len: int = 28, stride: int = 14) -> pd.DataFrame:
    """Windows over forecast rows only; trailing partial windows are dropped."""
    ids, off, n, nf = _layout(res)
    n_win = np.where(nf >= window_len, (nf - window_len) // stride + 1, 0)
    total = int(n_win.sum())

    s_idx = np.repeat(np.arange(len(ids)), n_win)
    first_of_series = np.cumsum(n_win) - n_win
    start_idx = (np.arange(total) - np.repeat(first_of_series, n_win)) * stride
    end_idx = start_idx + window_len
    # flat row of each window's first forecast row (forecast rows are the series tail)
    flat_first = off[s_idx] + (n - nf)[s_idx] + start_idx
    flat_last = flat_first + window_len - 1

    series_ids = ids[s_idx]
    ts = res["ts"]
    windows = pd.DataFrame(
        {
            "window_id": [f"{s}#{i}" for s, i in zip(series_ids, start_idx)],
            "series_id": series_ids,
            "start_idx": start_idx.astype(int),
            "end_idx": end_idx.astype(int),
            "start_ts": ts.iloc[flat_first].reset_index(drop=True),
            "end_ts": ts.iloc[flat_last].reset_index(drop=True),
        }
    )

    if "true_label" in res.columns:
        codes, uniques = pd.factorize(res["true_label"].astype(object))
        in_window = codes[flat_first[:, None] + np.arange(window_len)[None, :]]
        label = np.full(total, HEALTHY, dtype=object)
        best = np.zeros(total, dtype=np.int64)
        for k, name in enumerate(uniques):
            if name in (HEALTHY, HISTORY):
                continue
            count = (in_window == k).sum(axis=1)
            take = (2 * count >= window_len) & (count > best)
            label[take] = name
            best[take] = count[take]
        windows["true_label"] = label
    return windows


def window_matrix(
    res: pd.DataFrame, windows: pd.DataFrame, col: str, before: int = 0, after: int = 0
) -> np.ndarray:
    """Values of `res[col]` per window, shape [n_windows, before + L + after] (float64).

    Extends `before` rows before and `after` rows after each window in series step order (so the
    context may come from history rows); positions outside the series are NaN.
    """
    lens = (windows["end_idx"] - windows["start_idx"]).to_numpy()
    window_len = int(lens[0]) if len(lens) else 0
    if (lens != window_len).any():
        raise ValueError("windows must all have the same length")

    ids, off, n, nf = _layout(res)
    s = pd.Index(ids).get_indexer(windows["series_id"])
    if (s < 0).any():
        raise ValueError("windows reference series_ids missing from res")

    first_step = (n - nf)[s] + windows["start_idx"].to_numpy()  # step of each window's first row
    pos = first_step[:, None] + np.arange(-before, window_len + after)[None, :]
    valid = (pos >= 0) & (pos < n[s][:, None])
    flat = np.where(valid, off[s][:, None] + pos, 0)
    out = res[col].to_numpy(dtype=float)[flat]
    out[~valid] = np.nan
    return out
