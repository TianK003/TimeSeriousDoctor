import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import pytest

from atlas.plots import CLUSTER_COLORS, atlas_scatter, baseline_series_png, cluster_panel_png, series_figure
from atlas.schema import CLUSTER_NOISE, ClusterInfo, RunResult

N_HIST, N_FC, L, STRIDE = 56, 56, 28, 14
SERIES = ["s0", "s1", "s2"]
FEATURES = ["abs_err", "bias", "best_lag", "lag_gain", "slope_corr", "peak_excess",
            "var_ratio", "sign_run", "r_trend", "dir_acc"]


def make_series(sid: str, seed: int) -> pd.DataFrame:
    """Data rows of one series, then the `res` columns derived from them (SPEC 7.3 scaling)."""
    rng = np.random.default_rng(seed)
    n = N_HIST + N_FC
    t = np.arange(n)
    actual = 100 + 10 * np.sin(2 * np.pi * t / 7) + rng.normal(0, 1, n)
    predicted = np.roll(actual, 2) + rng.normal(0, 0.5, n)  # lagged forecast = "delayed_response"
    is_fc = t >= N_HIST
    predicted = np.where(is_fc, predicted, np.nan)
    return pd.DataFrame({
        "series_id": sid, "ts": pd.date_range("2024-01-01", periods=n, freq="D"),
        "actual": actual, "predicted": predicted, "is_forecast": is_fc,
        "true_label": np.where(is_fc, "delayed_response", "history"),
    })


def to_res(data: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, g in data.groupby("series_id"):
        g = g.sort_values("ts").reset_index(drop=True).copy()
        hist = g[~g["is_forecast"]]
        scale = max(np.abs(hist["actual"].to_numpy()[7:] - hist["actual"].to_numpy()[:-7]).mean(),
                    1e-6, 0.05 * np.abs(g["actual"]).mean())
        hist_mean = hist["actual"].mean()
        g["step"] = np.arange(len(g))
        g["fstep"] = np.where(g["is_forecast"], np.cumsum(g["is_forecast"]) - 1, -1)
        g["scale"], g["hist_mean"] = scale, hist_mean
        g["r_s"] = (g["actual"] - g["predicted"]) / scale
        g["a_s"] = (g["actual"] - hist_mean) / scale
        g["p_s"] = (g["predicted"] - hist_mean) / scale
        out.append(g)
    cols = ["series_id", "ts", "is_forecast", "actual", "predicted", "step", "fstep", "scale",
            "hist_mean", "r_s", "a_s", "p_s", "true_label"]
    return pd.concat(out, ignore_index=True)[cols]


def make_windows(res: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sid, g in res[res["is_forecast"]].groupby("series_id"):
        g = g.sort_values("fstep").reset_index(drop=True)
        for start in range(0, len(g) - L + 1, STRIDE):
            rows.append({
                "window_id": f"{sid}#{start}", "series_id": sid, "start_idx": start, "end_idx": start + L,
                "start_ts": g["ts"].iloc[start], "end_ts": g["ts"].iloc[start + L - 1],
                "true_label": "delayed_response",
            })
    w = pd.DataFrame(rows)
    n = len(w)
    w["score_abs"] = np.linspace(1.0, 3.0, n)
    w["score_bias"] = 0.1
    w["gated"] = np.arange(n) != 1  # one window not gated
    for f in FEATURES:
        w[f] = np.linspace(0.5, 2.0, n)
    w["emb_row"] = np.where(w["gated"], np.cumsum(w["gated"]) - 1, -1)
    w["cluster"] = np.where(~w["gated"], -2, np.where(np.arange(n) % 4 == 3, CLUSTER_NOISE, np.arange(n) % 2))
    rng = np.random.default_rng(1)
    w["x2d"], w["y2d"] = rng.normal(size=n), rng.normal(size=n)
    return w


@pytest.fixture(scope="module")
def world():
    data = pd.concat([make_series(s, i) for i, s in enumerate(SERIES)], ignore_index=True)
    res = to_res(data)
    windows = make_windows(res)
    cluster = ClusterInfo(
        cluster_id=0, size=int((windows["cluster"] == 0).sum()), n_series=3,
        medoid_window_ids=["s0#0", "s1#14", "s2#28"],
        feature_median={f: 1.0 for f in FEATURES}, feature_median_all_gated={f: 0.5 for f in FEATURES},
        model_card_dist={"AutoETS": 1.0},
    )
    run = RunResult(
        name="t", config={}, data=data, series_meta=pd.DataFrame({"series_id": SERIES, "model_card": "AutoETS"}),
        windows=windows, embeddings=np.zeros((0, 0), dtype=np.float32), clusters=[cluster],
    )
    return data, res, windows, cluster, run


def png_shape(path):
    return mpimg.imread(path).shape  # (height, width, channels)


def test_cluster_panel_size_and_path(world, tmp_path):
    _, res, windows, cluster, _ = world
    out = tmp_path / "deep" / "dir" / "cluster_0.png"
    assert cluster_panel_png(cluster, windows, res, out) == out
    assert out.exists()
    assert png_shape(out)[:2] == (900, 1200)


def test_cluster_panel_context_from_history_and_few_medoids(world, tmp_path):
    _, res, windows, cluster, _ = world
    one = cluster.model_copy(update={"medoid_window_ids": ["s0#0"]})  # start_idx 0: context is all history
    out = cluster_panel_png(one, windows, res, tmp_path / "one.png", context=14)
    assert png_shape(out)[:2] == (900, 1200)


def test_cluster_panel_noise_cluster_and_deterministic_sampling(world, tmp_path):
    _, res, windows, cluster, _ = world
    # 108 noise members (> 40) -> sampled with a fixed seed, so two renders are identical.
    big = pd.concat(
        [windows.assign(window_id=windows["window_id"] + (f"_{i}" if i else ""), cluster=CLUSTER_NOISE)
         for i in range(12)],
        ignore_index=True,
    )  # first copy keeps the medoid ids
    noise = cluster.model_copy(update={"cluster_id": CLUSTER_NOISE, "size": len(big)})
    a = cluster_panel_png(noise, big, res, tmp_path / "a.png")
    b = cluster_panel_png(noise, big, res, tmp_path / "b.png")
    assert png_shape(a)[:2] == (900, 1200)
    assert a.read_bytes() == b.read_bytes()


def test_baseline_series_png(world, tmp_path):
    data, *_ = world
    series = data[data["series_id"] == "s0"]
    png = baseline_series_png(series)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"

    out = tmp_path / "base.png"
    png2 = baseline_series_png(series, path=out, title="Series s0 (AutoETS)")
    assert out.read_bytes() == png2
    assert png_shape(out)[:2] == (450, 1000)

    no_fc = series.assign(is_forecast=False)  # no forecast region: must not crash
    assert baseline_series_png(no_fc)[:4] == b"\x89PNG"


def test_series_figure(world):
    *_, run = world
    fig = series_figure("s1", run)
    w = run.windows
    n_gated = int(((w["series_id"] == "s1") & w["gated"]).sum())
    assert n_gated > 0
    assert len(fig.data) == n_gated + 2  # one fill per gated window + actual + predicted
    assert len(fig.layout.shapes) == 1  # forecast region
    names = [t.name for t in fig.data]
    assert names[-2:] == ["actual", "predicted"]
    first = next(t for t in fig.data if t.fill == "toself")
    assert "s1#" in first.text and "cluster" in first.text and "abs_err" in first.text
    assert fig.to_json()  # timestamps in traces/shapes serialize


def test_series_figure_colors(world):
    *_, run = world
    w = run.windows[(run.windows["series_id"] == "s0") & run.windows["gated"]]
    fills = [t for t in series_figure("s0", run).data if t.fill == "toself"]
    assert len(fills) == len(w)
    for t, (_, row) in zip(fills, w.iterrows()):
        expected = "rgba(140,140,140" if row["cluster"] < 0 else None
        if expected:
            assert t.fillcolor.startswith(expected)
        else:
            h = CLUSTER_COLORS[int(row["cluster"]) % len(CLUSTER_COLORS)].lstrip("#")
            r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
            assert t.fillcolor.startswith(f"rgba({r},{g},{b}")


def test_atlas_scatter(world):
    *_, run = world
    w = run.windows
    fig = atlas_scatter(w)
    g = w[w["gated"]]
    assert len(fig.data) == g["cluster"].nunique()
    assert sum(len(t.x) for t in fig.data) == len(g)
    assert {t.name for t in fig.data} == {str(c) for c in g["cluster"].unique()}
    assert len(atlas_scatter(w.assign(gated=False)).data) == 0
