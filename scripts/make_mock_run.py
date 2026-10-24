"""Write a small fake run for UI development.

    python scripts/make_mock_run.py [--out runs/mock]

Writes ~20 series x (60 history + 84 forecast) daily rows, the windows frame with every base
column (gate scores, the 10 feature columns, embedding row, cluster, 2D coordinates, true label),
3 clusters with canned Gemma reports plus an unclustered bucket without one, placeholder PNGs,
an atlas summary and a canned eval.json. It never calls Gemma and needs no API key (SPEC.md §6).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:  # `python scripts/make_mock_run.py` puts scripts/ first, not the repo root
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from atlas.config import Config  # noqa: E402
from atlas.schema import (  # noqa: E402
    CLUSTER_NOISE,
    CLUSTER_NOT_GATED,
    AtlasSummary,
    Cause,
    ClusterInfo,
    GemmaClusterReport,
    RunResult,
    save_run,
)

# Hardcoded on purpose: this script must not import atlas.features (names from SPEC §7.5).
FEATURES = [
    "abs_err", "bias", "best_lag", "lag_gain", "slope_corr",
    "peak_excess", "var_ratio", "sign_run", "r_trend", "dir_acc",
]
MODEL_CARDS = [
    "AutoETS", "LightGBM on lag features", "Prophet", "AutoARIMA", "Seasonal naive",
]
N_HISTORY, N_FORECAST, WINDOW_LEN, STRIDE = 60, 84, 28, 14
SEG_LEN_RANGE = (42, 57)     # failure segment length in forecast rows
SEG_START_RANGE = (14, 29)   # failure segment start in forecast rows
LABEL_PLAN = (
    ["delayed_response"] * 6 + ["unadapted_shift"] * 4 + ["over_smoothing"] * 3 + ["healthy"] * 7
)
N_NOISE_WINDOWS = 4          # healthy windows the gate lets through (-> cluster -1)
N_MISSED_WINDOWS = 2         # failing windows the gate misses (-> not gated)

# Mean feature profile per window kind: (abs_err, bias, best_lag, lag_gain, slope_corr,
# peak_excess, var_ratio, sign_run, r_trend, dir_acc)
PROFILES: dict[str, list[float]] = {
    "delayed_response": [1.7, 0.15, 3.0, 0.55, 0.25, 0.35, 0.95, 6.0, 0.20, 0.55],
    "unadapted_shift": [2.6, 2.20, 0.0, 0.05, 0.80, 0.10, 1.00, 22.0, 0.02, 0.70],
    "over_smoothing": [1.4, 0.00, 0.0, 0.05, 0.60, 1.40, 0.35, 3.0, 0.00, 0.72],
    "noise": [1.15, 0.20, 1.0, 0.10, 0.30, 0.40, 0.90, 4.0, 0.05, 0.60],
    "healthy": [0.45, 0.00, 0.0, 0.03, 0.15, 0.10, 1.00, 3.0, 0.00, 0.80],
}


# ------------------------------------------------------------------------- data

def _make_series(
    rng: np.random.Generator, series_id: str, kind: str, seg: tuple[int, int]
) -> pd.DataFrame:
    n = N_HISTORY + N_FORECAST
    t = np.arange(n)
    level = rng.uniform(80, 200)
    trend = rng.uniform(-0.05, 0.12)
    sigma = rng.uniform(2.0, 4.0)
    season = rng.uniform(8, 18) * np.sin(2 * np.pi * (t + rng.integers(0, 7)) / 7)
    base = level + trend * t + season
    actual = base + rng.normal(0, sigma, n)

    s0, s1 = N_HISTORY + seg[0], N_HISTORY + seg[1]
    delta = 0.0
    if kind == "delayed_response":  # a bump the forecast only reacts to 3 days late
        actual[s0:s1] += 6 * sigma * np.sin(np.linspace(0, np.pi, s1 - s0))
    elif kind == "unadapted_shift":  # level shift that persists after the segment
        delta = 6 * sigma
        actual[s0:] += delta

    predicted = actual + rng.normal(0, 0.35 * sigma, n)  # healthy forecast
    if kind == "delayed_response":
        predicted[s0:s1] = actual[s0 - 3:s1 - 3] + rng.normal(0, 0.35 * sigma, s1 - s0)
    elif kind == "unadapted_shift":
        predicted[s0:s1] = actual[s0:s1] - delta + rng.normal(0, 0.35 * sigma, s1 - s0)
    elif kind == "over_smoothing":
        alpha = rng.uniform(0.0, 0.3)
        predicted[s0:s1] = (level + trend * t + alpha * season)[s0:s1] + rng.normal(0, 0.35 * sigma, s1 - s0)
    predicted[:N_HISTORY] = np.nan

    row_label = np.array(["history"] * N_HISTORY + ["healthy"] * N_FORECAST, dtype=object)
    if kind != "healthy":
        row_label[s0:s1] = kind
    return pd.DataFrame(
        {
            "series_id": series_id,
            "ts": pd.date_range("2025-01-01", periods=n, freq="D"),
            "actual": actual.round(2),
            "predicted": predicted.round(2),
            "is_forecast": t >= N_HISTORY,
            "true_label": row_label,
        }
    )


def _make_windows(series: pd.DataFrame) -> list[dict]:
    rows = []
    fc = series[series["is_forecast"]].reset_index(drop=True)
    sid = fc["series_id"].iloc[0]
    for start in range(0, len(fc) - WINDOW_LEN + 1, STRIDE):
        end = start + WINDOW_LEN
        labels = fc["true_label"].iloc[start:end]
        top = labels.value_counts()
        label = top.index[0] if top.iloc[0] >= WINDOW_LEN / 2 else "healthy"
        rows.append(
            {
                "window_id": f"{sid}#{start}",
                "series_id": sid,
                "start_idx": start,
                "end_idx": end,
                "start_ts": fc["ts"].iloc[start],
                "end_ts": fc["ts"].iloc[end - 1],
                "true_label": label,
            }
        )
    return rows


def _assign_clusters(windows: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Decide gated/cluster per window and the kind of feature profile each window gets."""
    failing = windows.index[windows["true_label"] != "healthy"].to_numpy()
    healthy = windows.index[windows["true_label"] == "healthy"].to_numpy()
    missed = set(rng.choice(failing, size=N_MISSED_WINDOWS, replace=False))
    noisy = set(rng.choice(healthy, size=N_NOISE_WINDOWS, replace=False))

    profile, gated = [], []
    for i, label in windows["true_label"].items():
        if label != "healthy" and i not in missed:
            profile.append(label)
            gated.append(True)
        elif i in noisy:
            profile.append("noise")
            gated.append(True)
        else:
            profile.append("healthy")
            gated.append(False)
    windows["profile"] = profile
    windows["gated"] = gated

    # cluster ids 0..n-1 ordered by size desc (SPEC §6); ties by label name
    sizes = windows[windows["gated"] & (windows["profile"] != "noise")]["profile"].value_counts()
    order = sorted(sizes.index, key=lambda k: (-sizes[k], k))
    cid = {kind: i for i, kind in enumerate(order)}
    windows["cluster"] = [
        CLUSTER_NOT_GATED if not g else (CLUSTER_NOISE if p == "noise" else cid[p])
        for g, p in zip(windows["gated"], windows["profile"])
    ]
    return windows


def _add_features(windows: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    mat = np.array([PROFILES[p] for p in windows["profile"]], dtype=float)
    mat = mat * rng.normal(1.0, 0.12, mat.shape) + rng.normal(0.0, 0.02, mat.shape)
    mat[:, FEATURES.index("best_lag")] = np.round(mat[:, FEATURES.index("best_lag")]) + 0.0
    mat[:, FEATURES.index("sign_run")] = np.round(mat[:, FEATURES.index("sign_run")]) + 0.0
    mat[:, FEATURES.index("dir_acc")] = np.clip(mat[:, FEATURES.index("dir_acc")], 0, 1)
    feats = pd.DataFrame(mat.round(3), columns=FEATURES, index=windows.index)
    windows = pd.concat([windows, feats], axis=1)
    windows["score_abs"] = windows["abs_err"]
    windows["score_bias"] = windows["bias"].abs()
    return windows


def _embed_and_project(windows: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
    gated = windows["gated"].to_numpy()
    emb_row = np.full(len(windows), -1, dtype=int)
    emb_row[gated] = np.arange(gated.sum())
    windows["emb_row"] = emb_row

    x = windows.loc[gated, FEATURES].to_numpy(dtype=float)
    embeddings = ((x - x.mean(axis=0)) / (x.std(axis=0) + 1e-9)).astype(np.float32)

    n_clusters = int(windows["cluster"].max()) + 1
    x2d = np.full(len(windows), np.nan)
    y2d = np.full(len(windows), np.nan)
    for i in windows.index[gated]:
        c = int(windows.at[i, "cluster"])
        if c == CLUSTER_NOISE:
            cx, cy, spread = 0.0, 0.0, 2.2
        else:
            angle = 2 * np.pi * c / max(n_clusters, 1)
            cx, cy, spread = 4.5 * np.cos(angle), 4.5 * np.sin(angle), 0.8
        x2d[i] = cx + rng.normal(0, spread)
        y2d[i] = cy + rng.normal(0, spread)
    windows["x2d"], windows["y2d"] = x2d, y2d
    return windows, embeddings


# ------------------------------------------------------------------ canned Gemma text

def _report(kind: str, med: dict[str, float], base: dict[str, float]) -> GemmaClusterReport:
    def f(name: str) -> str:
        return f"{name} {med[name]:.2f} vs {base[name]:.2f} across all anomalous windows"

    if kind == "delayed_response":
        return GemmaClusterReport(
            failure_type="delayed_response",
            name="Forecast lags actuals by days",
            summary=(
                "The forecast follows the actuals with a delay of about three days, so every turning "
                "point is reported late. Errors are sizeable but mostly cancel out in sign."
            ),
            evidence=[
                f"Best cross-correlation lag is {med['best_lag']:.0f} days with a large lag gain: {f('lag_gain')}.",
                f"Residual sign runs are long ({f('sign_run')}), as expected after every turn.",
                "In the example plot the predicted line is a time-shifted copy of the actual line.",
            ],
            likely_causes=[
                Cause(
                    cause="Model leans on lagged targets",
                    why="A forecast built from recent values can only reproduce a change after it already happened.",
                ),
                Cause(
                    cause="Too much smoothing of the input",
                    why="A long moving average delays the signal by roughly half its window.",
                ),
            ],
            suggested_fixes=[
                "Add leading indicators (price, promotion, calendar) so the model can anticipate changes.",
                "Shorten the smoothing window or the lag set and compare MASE on the affected windows.",
            ],
            metric_to_monitor="Lag-adjusted MASE: best-lag gain per series, alert above 0.4",
            confidence=0.86,
        )
    if kind == "unadapted_shift":
        return GemmaClusterReport(
            failure_type="unadapted_shift",
            name="Stuck at the old level",
            summary=(
                "After a level shift in the data, the forecast stays at the old level for weeks. "
                "The error is large and has the same sign throughout."
            ),
            evidence=[
                f"Residuals are strongly one-sided: {f('bias')}.",
                f"Very long same-sign runs: {f('sign_run')}.",
                f"The forecast shape still tracks the actuals ({f('slope_corr')}); it is only shifted.",
            ],
            likely_causes=[
                Cause(
                    cause="Model refit too rarely",
                    why="The level estimate is frozen between refits, so a structural break is not absorbed.",
                ),
                Cause(
                    cause="Regime change not in the features",
                    why="A new assortment, price level or store opening shifted demand and nothing signals it.",
                ),
            ],
            suggested_fixes=[
                "Refit more often or add a level-adaptive component (e.g. exponential reweighting of recent data).",
                "Add change-point detection that triggers an early refit.",
            ],
            metric_to_monitor="Rolling 28-day bias (mean scaled residual), alert when |bias| stays above 0.75",
            confidence=0.91,
        )
    return GemmaClusterReport(
        failure_type="over_smoothing",
        name="Dampened seasonal swings",
        summary=(
            "The forecast is much flatter than reality: weekly peaks and troughs are cut down, "
            "so the largest deviations occur on peak days."
        ),
        evidence=[
            f"Forecast variance is far below actual variance: {f('var_ratio')}.",
            f"Errors concentrate on peaks: {f('peak_excess')}.",
            "In the example plot the predicted line is a smoothed version of the actual line.",
        ],
        likely_causes=[
            Cause(
                cause="Over-regularised seasonal component",
                why="A strong penalty or heavy shrinkage pulls seasonal amplitudes toward zero.",
            ),
            Cause(
                cause="Seasonality estimated on a mix of series",
                why="Pooling averages out series with different weekly profiles.",
            ),
        ],
        suggested_fixes=[
            "Loosen the seasonal regularisation or fit seasonality per series.",
            "Compare peak-day MASE against a seasonal-naive benchmark.",
        ],
        metric_to_monitor="Forecast-to-actual variance ratio, alert below 0.6",
        confidence=0.78,
    )


def _summary(clusters: list[ClusterInfo], n_series: int, n_gated: int) -> AtlasSummary:
    named = [c for c in clusters if c.report is not None]
    names = ", ".join(f"'{c.report.name}'" for c in named)
    return AtlasSummary(
        headline="Three recurring forecast failures explain most anomalous windows",
        summary=(
            f"Across {n_series} series, {n_gated} windows were flagged as anomalous and "
            f"{sum(c.size for c in named)} of them fall into {len(named)} clear failure patterns: {names}. "
            "The largest group is a delayed response to turning points. "
            "A small unclustered remainder does not match any pattern and should be reviewed by hand."
        ),
        actions=[
            "Review the delayed-response series first; adding leading indicators is the cheapest fix.",
            "Refit the models behind the level-shift cluster and add change-point triggers.",
            "Relax seasonal regularisation for the dampened-seasonality series.",
        ],
    )


def _eval() -> dict:
    classes = ["delayed_response", "unadapted_shift", "over_smoothing", "healthy"]
    return {
        "run": "mock",
        "embedder": "features",
        "gate": {"precision": 0.9, "recall": 0.8, "f1": 0.85},
        "clustering": {
            "ari": 0.7, "nmi": 0.75, "purity": 0.8, "noise_share": 0.1,
            "confusion": {
                "0": {"delayed_response": 40, "healthy": 3},
                "1": {"unadapted_shift": 28},
                "2": {"over_smoothing": 19, "healthy": 2},
                "-1": {"healthy": 6, "delayed_response": 2},
            },
        },
        "naming_accuracy": 0.75,
        "series_accuracy": 0.8,
        "comparison": {
            "n_series": 40,
            "pipeline": {
                "accuracy": 0.8, "macro_f1": 0.78,
                "per_class": dict(zip(classes, [0.9, 0.85, 0.75, 0.7])),
                "gemma_calls": 7,
            },
            "gemma_alone": {
                "accuracy": 0.45, "macro_f1": 0.4,
                "per_class": dict(zip(classes, [0.5, 0.4, 0.35, 0.55])),
                "gemma_calls": 40,
            },
            "gemma_calls_full_dataset": {"pipeline": 7, "gemma_alone": 400},
        },
    }


# ---------------------------------------------------------------------------- images

def _placeholder_png(cluster: ClusterInfo, data: pd.DataFrame, windows: pd.DataFrame, path: Path) -> None:
    win = windows[windows["window_id"] == cluster.medoid_window_ids[0]].iloc[0]
    fc = data[(data["series_id"] == win["series_id"]) & data["is_forecast"]].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 3), dpi=100)
    ax.plot(fc["ts"], fc["actual"], color="#1f77b4", label="actual")
    ax.plot(fc["ts"], fc["predicted"], color="#d62728", label="predicted")
    ax.axvspan(win["start_ts"], win["end_ts"], color="#999999", alpha=0.2, label="window")
    ax.set_title(f"Cluster {cluster.cluster_id}: {cluster.report.name} (mock placeholder, {win['series_id']})")
    ax.legend(loc="upper left", fontsize=8)
    ax.tick_params(axis="x", labelrotation=30, labelsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------------------------ main

def make_mock_run(out: Path, seed: int = 0) -> Path:
    out = Path(out)
    rng = np.random.default_rng(seed)

    plan = list(rng.permutation(LABEL_PLAN))
    frames, window_rows = [], []
    for k, kind in enumerate(plan):
        seg_start = int(rng.integers(*SEG_START_RANGE))
        seg_len = int(rng.integers(*SEG_LEN_RANGE))
        frame = _make_series(rng, f"S{k + 1:02d}", str(kind), (seg_start, seg_start + seg_len))
        frames.append(frame)
        window_rows += _make_windows(frame)
    data = pd.concat(frames, ignore_index=True)
    windows = pd.DataFrame(window_rows)

    windows = _assign_clusters(windows, rng)
    windows = _add_features(windows, rng)
    windows, embeddings = _embed_and_project(windows, rng)

    meta = pd.DataFrame(
        {
            "series_id": [f"S{k + 1:02d}" for k in range(len(plan))],
            "model_card": [MODEL_CARDS[int(i)] for i in rng.integers(0, len(MODEL_CARDS), len(plan))],
            "group": [("A", "B", "C")[int(i)] for i in rng.integers(0, 3, len(plan))],
        }
    )
    card_of = dict(zip(meta["series_id"], meta["model_card"]))

    gated_df = windows[windows["gated"]]
    base_median = {f: float(gated_df[f].median()) for f in FEATURES}
    kind_of_cluster = {
        int(c): windows[windows["cluster"] == c]["profile"].iloc[0]
        for c in sorted(windows["cluster"].unique()) if c >= 0
    }

    clusters: list[ClusterInfo] = []
    for cid in [*kind_of_cluster, CLUSTER_NOISE]:
        members = windows[windows["cluster"] == cid]
        centroid = embeddings[members["emb_row"].to_numpy()].mean(axis=0)
        dist = np.linalg.norm(embeddings[members["emb_row"].to_numpy()] - centroid, axis=1)
        medoids = members["window_id"].to_numpy()[np.argsort(dist)[:3]].tolist()
        mix = members["series_id"].map(card_of).value_counts(normalize=True)
        med = {f: float(members[f].median()) for f in FEATURES}
        clusters.append(
            ClusterInfo(
                cluster_id=cid,
                size=len(members),
                n_series=int(members["series_id"].nunique()),
                medoid_window_ids=medoids,
                feature_median=med,
                feature_median_all_gated=base_median,
                model_card_dist={k: float(v) for k, v in mix.items()},
                image_path=None if cid == CLUSTER_NOISE else f"images/cluster_{cid}.png",
                report=None if cid == CLUSTER_NOISE else _report(kind_of_cluster[cid], med, base_median),
            )
        )

    windows = windows.drop(columns=["profile"])
    windows = windows[[
        "window_id", "series_id", "start_idx", "end_idx", "start_ts", "end_ts", "true_label",
        "score_abs", "score_bias", "gated", *FEATURES, "emb_row", "cluster", "x2d", "y2d",
    ]]

    run = RunResult(
        name=out.name,
        config=Config(embedder="features").model_dump(mode="json"),
        data=data,
        series_meta=meta,
        windows=windows,
        embeddings=embeddings,
        clusters=clusters,
        summary=_summary(clusters, len(plan), int(windows["gated"].sum())),
        eval=_eval(),
    )
    save_run(run, out)
    for c in clusters:
        if c.image_path is not None:
            _placeholder_png(c, data, windows, out / c.image_path)
    return out


def main(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(description="Write a small fake run for UI development.")
    parser.add_argument("--out", type=Path, default=Path("runs/mock"), help="run folder to write")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    out = make_mock_run(args.out, args.seed)
    print(f"wrote mock run to {out}")
    return out


if __name__ == "__main__":
    main()
