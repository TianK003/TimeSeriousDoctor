"""Anomaly gate.

Spec: SPEC.md §7.4. Scores each window from its scaled residuals and flags the ones worth explaining.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from atlas.config import GateConfig
from atlas.residuals import window_matrix


def add_scores(windows: pd.DataFrame, res: pd.DataFrame) -> pd.DataFrame:
    """Copy of `windows` with score_abs = mean|r_s|, score_bias = |mean r_s|, score_max = max|r_s| per window."""
    r = window_matrix(res, windows, "r_s")
    count = np.sum(~np.isnan(r), axis=1)
    total = np.nansum(r, axis=1)
    total_abs = np.nansum(np.abs(r), axis=1)
    safe = np.maximum(count, 1)
    out = windows.copy()
    out["score_abs"] = np.where(count > 0, total_abs / safe, np.nan)
    out["score_bias"] = np.where(count > 0, np.abs(total / safe), np.nan)
    peak = np.max(np.where(np.isnan(r), -np.inf, np.abs(r)), axis=1) if r.shape[1] else np.zeros(len(r))
    out["score_max"] = np.where(count > 0, peak, np.nan)
    return out


def gate_windows(windows: pd.DataFrame, cfg: GateConfig) -> np.ndarray:
    """Boolean mask of windows worth explaining (NaN scores are never gated)."""
    score_abs = windows["score_abs"].to_numpy(dtype=float)
    if cfg.mode == "threshold":
        score_bias = windows["score_bias"].to_numpy(dtype=float)
        score_max = windows["score_max"].to_numpy(dtype=float)
        gated = (score_abs > cfg.t_abs) | (score_bias > cfg.t_bias) | (score_max > cfg.t_peak)
    else:
        finite = score_abs[~np.isnan(score_abs)]
        if len(finite) == 0:
            gated = np.zeros(len(score_abs), dtype=bool)
        else:
            gated = score_abs >= np.quantile(finite, 1.0 - cfg.q)
    n, g = len(gated), int(gated.sum())
    share = 100.0 * g / n if n else 0.0
    print(f"gate: {n} windows, {g} gated ({share:.1f}%)")
    return gated
