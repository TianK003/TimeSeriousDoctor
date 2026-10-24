"""t-SNE projection, latent-space figures and the light/dark theme helper (Task 15)."""

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from atlas.cluster import project_tsne
from atlas.plots import atlas_scatter, cluster_name_map, latent_figure, series_figure, window_figure
from atlas.schema import load_run
from atlas.ui_theme import DARK_CSS, plotly_template
from scripts.make_mock_run import main


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    out = main(["--out", str(tmp_path_factory.mktemp("runs") / "mock")])
    return load_run(out)


def blobs(n=60, dim=5, seed=0):
    rng = np.random.default_rng(seed)
    return np.vstack([rng.normal(0, 0.3, (n // 2, dim)), rng.normal(6, 0.3, (n - n // 2, dim))])


# ---------------------------------------------------------------- project_tsne

def test_tsne_shape_and_finite():
    out = project_tsne(blobs(60))
    assert out.shape == (60, 2)
    assert np.isfinite(out).all()


def test_tsne_separates_two_blobs():
    out = project_tsne(blobs(60))
    a, b = out[:30].mean(axis=0), out[30:].mean(axis=0)
    assert np.linalg.norm(a - b) > 3 * max(out[:30].std(), out[30:].std())


def test_tsne_tiny_input_is_zeros():
    out = project_tsne(np.ones((4, 3)))
    assert out.shape == (4, 2) and (out == 0).all()
    assert project_tsne(np.zeros((0, 3))).shape == (0, 2)


def test_tsne_cap_keeps_kept_rows_and_nans_the_rest():
    X = blobs(80)
    keep = np.zeros(80, dtype=bool)
    keep[::4] = True  # 20 rows
    out = project_tsne(X, max_points=35, seed=1, keep=keep)
    assert out.shape == (80, 2)
    finite = np.isfinite(out).all(axis=1)
    assert finite.sum() == 35
    assert finite[keep].all()  # every kept row is projected
    assert np.isnan(out[~finite]).all()
    # same seed -> same sample
    again = project_tsne(X, max_points=35, seed=1, keep=keep)
    assert np.array_equal(np.isfinite(again).all(axis=1), finite)


def test_tsne_cap_without_keep_and_keep_larger_than_cap():
    X = blobs(60)
    assert np.isfinite(project_tsne(X, max_points=25)).all(axis=1).sum() == 25
    keep = np.ones(60, dtype=bool)  # all rows kept: the cap never drops them
    assert np.isfinite(project_tsne(X, max_points=25, keep=keep)).all()


def test_tsne_nan_and_single_feature_input():
    X = blobs(40)
    X[3, 1] = np.nan
    assert np.isfinite(project_tsne(X)).all()
    assert np.isfinite(project_tsne(X[:, :1].copy())).all()


# ---------------------------------------------------------------- latent_figure

def test_latent_figure_cluster_traces_and_customdata(run):
    w = run.windows
    coords = np.random.default_rng(0).normal(size=(len(w), 2))
    names = cluster_name_map(run.clusters)
    fig = latent_figure(w, coords, cluster_names=names)

    gated = w[w["gated"].astype(bool)]
    n_clusters = gated["cluster"].nunique()
    assert len(fig.data) == 1 + n_clusters  # normal + one per cluster
    assert fig.data[0].name == "normal (not anomalous)"
    assert fig.layout.xaxis.title.text == "t-SNE 1" and fig.layout.yaxis.title.text == "t-SNE 2"
    assert fig.layout.height == 600
    assert not fig.layout.xaxis.showticklabels

    ids = np.concatenate([np.asarray(t.customdata) for t in fig.data])
    assert sorted(ids) == sorted(w["window_id"])  # every window drawn once, customdata = window_id
    cid0 = min(c for c in gated["cluster"].unique() if c >= 0)
    assert any(t.name == f"{cid0}: {names[cid0]}" for t in fig.data)
    assert any(t.name == "unclustered" for t in fig.data)
    hover = fig.data[1].text[0]
    assert "series:" in hover and "abs_err:" in hover and "true label:" in hover


def test_latent_figure_highlight_adds_one_trace(run):
    w = run.windows
    coords = np.random.default_rng(0).normal(size=(len(w), 2))
    base = latent_figure(w, coords)
    sid = w["series_id"].iloc[0]
    fig = latent_figure(w, coords, highlight_series=sid)
    assert len(fig.data) == len(base.data) + 1
    top = fig.data[-1]
    assert top.name == "selected series"
    assert set(top.customdata) == set(w.loc[w["series_id"] == sid, "window_id"])
    assert top.marker.line.width and top.marker.size > fig.data[1].marker.size
    # a series that is not in the frame draws nothing extra
    assert len(latent_figure(w, coords, highlight_series="nope").data) == len(base.data)


def test_latent_figure_skips_nan_coords_and_color_modes(run):
    w = run.windows
    coords = np.random.default_rng(0).normal(size=(len(w), 2))
    coords[::3] = np.nan
    fig = latent_figure(w, coords, color_by="true_label")
    n_drawn = sum(len(t.customdata) for t in fig.data)
    assert n_drawn == int(np.isfinite(coords).all(axis=1).sum())
    names = {t.name for t in fig.data}
    assert "normal (not anomalous)" in names and names & set(w["true_label"].astype(str))

    by_series = latent_figure(w, coords, color_by="series_id", highlight_series=w["series_id"].iloc[0])
    assert by_series.data[-1].name == "selected series"

    no_label = latent_figure(w.drop(columns="true_label"), coords, color_by="true_label")  # falls back to cluster
    assert len(no_label.data) > 1

    with pytest.raises(ValueError):
        latent_figure(w, coords[:-1])


def test_latent_figure_dark_template(run):
    w = run.windows
    coords = np.random.default_rng(0).normal(size=(len(w), 2))
    fig = latent_figure(w, coords, highlight_series=w["series_id"].iloc[0], template="plotly_dark")
    assert fig.layout.template.layout.paper_bgcolor == "rgb(17,17,17)"
    assert fig.data[-1].marker.line.color == "white"


# ---------------------------------------------------------------- window_figure

def test_window_figure_for_mock_window(run):
    w = run.windows
    row = w[w["gated"].astype(bool) & (w["cluster"] >= 0)].iloc[0]
    fig = window_figure(row["window_id"], run, context=14)
    assert [t.name for t in fig.data] == ["actual", "predicted"]
    cid = int(row["cluster"])
    name = run.cluster(cid).report.name
    assert fig.layout.title.text == f"{row['window_id']} — cluster {cid} ({name})"
    assert len(fig.layout.shapes) == 1  # shaded window region
    n_pts = len(fig.data[0].x)
    assert n_pts == (row["end_idx"] - row["start_idx"]) + 14  # start_idx >= 14 in the mock, so full context

    first = w.sort_values("start_idx").iloc[0]  # start_idx 0: context comes from history rows
    fig0 = window_figure(first["window_id"], run, context=14)
    assert len(fig0.data[0].x) == (first["end_idx"] - first["start_idx"]) + 14

    noise = w[w["cluster"] == -1].iloc[0]["window_id"]
    assert "unclustered" in window_figure(noise, run).layout.title.text
    normal = w[~w["gated"].astype(bool)].iloc[0]["window_id"]
    assert "normal" in window_figure(normal, run, template="plotly_dark").layout.title.text


# ---------------------------------------------------------------- template kwarg on existing figures

def test_existing_figures_accept_template(run):
    sid = run.windows["series_id"].iloc[0]
    assert series_figure(sid, run, template="plotly_dark").layout.template.layout.paper_bgcolor == "rgb(17,17,17)"
    assert atlas_scatter(run.windows, template="plotly_dark").layout.template.layout.paper_bgcolor == "rgb(17,17,17)"
    assert series_figure(sid, run).layout.template.layout.paper_bgcolor == "white"  # default stays light
    assert atlas_scatter(run.windows).layout.template.layout.paper_bgcolor == "white"


# ---------------------------------------------------------------- theme helper

def test_plotly_template():
    assert plotly_template(True) == "plotly_dark"
    assert plotly_template(False) == "plotly_white"


def test_dark_css_targets_stable_selectors():
    for sel in ['[data-testid="stSidebar"]', '[data-testid="stHeader"]', '[data-testid="stMetricValue"]',
                "#0E1117", "#FAFAFA", "#262730"]:
        assert sel in DARK_CSS
    assert "{{" not in DARK_CSS and "<" not in DARK_CSS  # f-string fully rendered, no markup injected


def theme_app():
    import streamlit as st

    from atlas.ui_theme import apply_theme, theme_toggle

    dark = theme_toggle()
    apply_theme(dark)
    st.write("dark" if dark else "light")


def test_theme_toggle_flips_and_injects_css():
    at = AppTest.from_function(theme_app, default_timeout=30).run()
    assert not at.exception
    assert at.toggle[0].label == "🌙 Dark mode" and at.toggle[0].value is False
    assert at.markdown[-1].value == "light"
    assert not any("<style>" in m.value for m in at.markdown)

    at.toggle[0].set_value(True).run()
    assert not at.exception
    assert at.toggle[0].value is True  # the relabel must not reset the widget
    assert at.toggle[0].label == "☀️ Light mode"
    assert at.markdown[-1].value == "dark"
    assert any("<style>" in m.value and "#0E1117" in m.value for m in at.markdown)

    at.toggle[0].set_value(False).run()
    assert at.toggle[0].value is False and at.toggle[0].label == "🌙 Dark mode"
    assert not any("<style>" in m.value for m in at.markdown)
