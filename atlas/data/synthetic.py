"""Synthetic forecasts with injected, labeled failures.

Owner: C. Task: P0.4, P0.10. Spec: SPEC.md §7.1.

Every series gets exactly one type (healthy or one of four failures). A failure is injected only inside one
random contiguous segment of 42-91 forecast days; outside it the predictions are healthy. Row labels:
`history` before the forecast, `healthy` in the forecast, the failure type inside the segment.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FAILURE_TYPES = ["delayed_response", "missed_peak", "unadapted_shift", "over_smoothing"]
LABELS = ["healthy", *FAILURE_TYPES]
DEFAULT_MIX = {label: 0.2 for label in LABELS}
MODEL_CARDS = ["AutoETS", "LightGBM on lag features", "Prophet", "AutoARIMA", "Seasonal naive"]
START_DATE = "2023-01-01"
DELAYS = (2, 3, 4, 5)  # never 7: that equals the weekly season and is invisible
PRED_NOISE = 0.35  # healthy p = actual + N(0, PRED_NOISE * sigma)
SEGMENT_LEN = (42, 91)  # inclusive range, in forecast days


def _allocate(n_series: int, mix: dict[str, float], rng: np.random.Generator) -> list[str]:
    """Exact per-type counts (largest remainder) in shuffled order."""
    unknown = set(mix) - set(LABELS)
    if unknown:
        raise ValueError(f"unknown types in mix: {sorted(unknown)}; allowed: {LABELS}")
    if any(v < 0 for v in mix.values()) or sum(mix.values()) <= 0:
        raise ValueError("mix shares must be non-negative and sum to > 0")
    total = sum(mix.values())
    exact = {k: n_series * v / total for k, v in mix.items()}
    counts = {k: int(np.floor(x)) for k, x in exact.items()}
    by_remainder = sorted(exact, key=lambda k: exact[k] - counts[k], reverse=True)
    for k in by_remainder[: n_series - sum(counts.values())]:
        counts[k] += 1
    labels = [k for k, c in counts.items() for _ in range(c)]
    rng.shuffle(labels)
    return labels


def _make_series(
    rng: np.random.Generator, label: str, history_len: int, forecast_len: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One series: returns (actual, predicted, row_labels), all of length history_len + forecast_len."""
    n = history_len + forecast_len
    t = np.arange(n)

    # actuals: level + linear trend + weekly seasonality (+ optional yearly term) + Gaussian noise.
    # level is 14-19 sigma so the residual scale guard (0.05 * mean|a|) stays below the seasonal-diff scale.
    sigma = rng.uniform(2.0, 6.0)
    level = sigma * rng.uniform(14.0, 19.0)
    slope = level * rng.uniform(-0.15, 0.15) / n
    amp = sigma * rng.uniform(3.0, 5.0)
    phase1, phase2 = rng.uniform(0.0, 2 * np.pi, size=2)
    weekly = amp * np.sin(2 * np.pi * t / 7 + phase1) + 0.3 * amp * np.sin(4 * np.pi * t / 7 + phase2)
    yearly = np.zeros(n)
    if rng.random() < 0.5:
        yearly = level * rng.uniform(0.02, 0.05) * np.sin(2 * np.pi * t / 365.25 + rng.uniform(0.0, 2 * np.pi))
    base = level + slope * t + yearly  # everything except weekly season and noise
    noise = rng.normal(0.0, sigma, n)
    pred_noise = rng.normal(0.0, PRED_NOISE * sigma, n)
    actual = base + weekly + noise
    row_labels = np.array(["history"] * history_len + ["healthy"] * forecast_len, dtype=object)

    if label == "healthy":
        predicted = actual + pred_noise
    else:
        seg_len = min(int(rng.integers(SEGMENT_LEN[0], SEGMENT_LEN[1] + 1)), forecast_len)
        start = history_len + int(rng.integers(0, forecast_len - seg_len + 1))
        end = start + seg_len
        seg = slice(start, end)
        row_labels[seg] = label

        if label == "delayed_response":
            k = int(rng.choice(DELAYS))
            ramp = np.zeros(n)  # slope change inside the segment, held afterwards
            ramp[seg] = rng.choice([-1.0, 1.0]) * rng.uniform(0.10, 0.25) * sigma * np.arange(1, seg_len + 1)
            ramp[end:] = ramp[end - 1]
            actual = actual + ramp
            predicted = actual + pred_noise
            predicted[seg] = actual[start - k : end - k] + pred_noise[seg]
        elif label == "missed_peak":
            spikes = np.zeros(n)
            n_spikes = int(rng.integers(4, 9))
            edges = np.linspace(0, seg_len, n_spikes + 1).astype(int)  # one spike per slot, no overlaps
            for lo, hi in zip(edges[:-1], edges[1:]):
                width = min(int(rng.integers(1, 4)), max(hi - lo, 1))  # slots are >= 5 days for the default segments
                pos = start + lo + int(rng.integers(0, max(hi - lo - width + 1, 1)))
                spikes[pos : pos + width] = rng.uniform(3.0, 6.0) * sigma
            actual = actual + spikes
            predicted = actual + pred_noise
            predicted[seg] = (actual - spikes)[seg] + pred_noise[seg]
        elif label == "unadapted_shift":
            delta = rng.choice([-1.0, 1.0]) * rng.uniform(4.0, 8.0) * sigma
            actual = actual + np.where(t >= start, delta, 0.0)
            predicted = actual + pred_noise
            predicted[seg] = actual[seg] - delta + pred_noise[seg]
        elif label == "over_smoothing":
            alpha = rng.uniform(0.0, 0.3)
            predicted = actual + pred_noise
            predicted[seg] = base[seg] + alpha * weekly[seg] + pred_noise[seg]
        else:
            raise ValueError(f"unknown label: {label}")

    predicted[:history_len] = np.nan
    return actual, predicted, row_labels


def make_synthetic(
    n_series: int = 400,
    history_len: int = 365,
    forecast_len: int = 182,
    seed: int = 0,
    mix: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (data, series_meta) in the SPEC §5 contract, with a row-level `true_label` column."""
    rng = np.random.default_rng(seed)
    labels = _allocate(n_series, mix or DEFAULT_MIX, rng)
    n = history_len + forecast_len
    ts = pd.date_range(START_DATE, periods=n, freq="D")

    frames, cards = [], []
    for i, label in enumerate(labels):
        actual, predicted, row_labels = _make_series(rng, label, history_len, forecast_len)
        frames.append(
            pd.DataFrame(
                {
                    "series_id": f"s{i:04d}",
                    "ts": ts,
                    "actual": actual,
                    "predicted": predicted,
                    "is_forecast": np.arange(n) >= history_len,
                    "true_label": row_labels,
                }
            )
        )
        cards.append(MODEL_CARDS[int(rng.integers(len(MODEL_CARDS)))])  # independent of the label

    data = pd.concat(frames, ignore_index=True)
    series_meta = pd.DataFrame({"series_id": [f"s{i:04d}" for i in range(len(labels))], "model_card": cards})
    return data, series_meta
