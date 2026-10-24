import numpy as np
import pandas as pd

from atlas.config import GateConfig
from atlas.gate import add_scores, gate_windows
from atlas.residuals import compute_scaled_residuals, make_windows
from atlas.schema import Dataset


def make_alternating_dataset() -> Dataset:
    """One flat series (scale = 0.05 * 100 = 5) whose forecast residual alternates +10, -10, +10, -20."""
    n_hist, n_fc = 10, 4
    actual = np.r_[np.full(n_hist, 100.0), np.full(n_fc, 100.0)]
    predicted = np.r_[np.full(n_hist, 100.0), 100.0 - np.array([10.0, -10.0, 10.0, -20.0])]
    data = pd.DataFrame(
        {
            "series_id": "F",
            "ts": pd.date_range("2024-01-01", periods=n_hist + n_fc, freq="D"),
            "actual": actual,
            "predicted": predicted,
            "is_forecast": [False] * n_hist + [True] * n_fc,
        }
    )
    return Dataset(data, pd.DataFrame({"series_id": ["F"], "model_card": ["m"]}))


def test_add_scores_abs_and_bias():
    res = compute_scaled_residuals(make_alternating_dataset(), season=2)
    windows = make_windows(res, window_len=4, stride=2)
    scored = add_scores(windows, res)
    assert "score_abs" not in windows.columns  # input is not mutated
    assert list(scored.columns)[: len(windows.columns)] == list(windows.columns)
    # r_s = [2, -2, 2, -4]: mean|r_s| = 2.5, |mean r_s| = 0.5
    assert np.isclose(scored.score_abs[0], 2.5)
    assert np.isclose(scored.score_bias[0], 0.5)
    assert np.isclose(scored.score_max[0], 4.0)
    cols = list(scored.columns)
    assert cols.index("score_abs") < cols.index("score_bias") < cols.index("score_max")


def test_gate_threshold_mode_is_strict_and_uses_either_score(capsys):
    w = pd.DataFrame(
        {
            "score_abs": [0.5, 1.2, 0.9, 1.0, 0.1],
            "score_bias": [0.1, 0.1, 0.8, 0.75, np.nan],
            "score_max": [1.0, 1.0, 1.0, 1.0, np.nan],
        }
    )
    gated = gate_windows(w, GateConfig(mode="threshold", t_abs=1.0, t_bias=0.75, t_peak=3.0))
    assert gated.dtype == bool
    assert list(gated) == [False, True, True, False, False]  # strict >, and NaN never gates
    assert "gate: 5 windows, 2 gated (40.0%)" in capsys.readouterr().out


def test_gate_quantile_mode_takes_top_share(capsys):
    w = pd.DataFrame({"score_abs": np.arange(1.0, 9.0), "score_bias": 0.0, "score_max": 99.0})
    gated = gate_windows(w, GateConfig(mode="quantile", q=0.25))
    assert list(np.flatnonzero(gated)) == [6, 7]  # top 25% of 8 windows
    assert "gate: 8 windows, 2 gated (25.0%)" in capsys.readouterr().out


def test_gate_threshold_mode_catches_a_short_spike(capsys):
    """A single +5 spike in a 28-row window has mean|r_s| = bias = 0.18 (under both thresholds) but max 5."""
    n = 28
    r_s = np.zeros(n)
    r_s[10] = 5.0
    res = pd.DataFrame(
        {
            "series_id": "S",
            "step": np.arange(n),
            "fstep": np.arange(n),
            "is_forecast": True,
            "r_s": r_s,
        }
    )
    healthy = res.assign(series_id="H", r_s=np.where(np.arange(n) % 2 == 0, 0.3, -0.3))
    res = pd.concat([res, healthy], ignore_index=True).sort_values(["series_id", "step"], ignore_index=True)
    windows = pd.DataFrame(
        {"series_id": ["H", "S"], "start_idx": [0, 0], "end_idx": [n, n]}
    )
    scored = add_scores(windows, res)
    assert np.isclose(scored.score_max[1], 5.0) and scored.score_abs[1] < 1.0 and scored.score_bias[1] < 0.75
    gated = gate_windows(scored, GateConfig(mode="threshold"))
    assert list(gated) == [False, True]  # healthy window not gated, spike window gated by score_max
    assert not gate_windows(scored, GateConfig(mode="threshold", t_peak=6.0))[1]
