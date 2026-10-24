"""Streamlit entry point.

Atlas, Cluster, Series and Evaluation tabs over a precomputed run folder (SPEC.md §7.12).
Runs are produced by the CLI (`python scripts/run_atlas.py`, or `python scripts/make_mock_run.py`
for a fake one); the app only reads them. Directory with runs: env var ATLAS_RUNS_DIR (default `runs`).
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from atlas.schema import CLUSTER_NOISE, ClusterInfo, RunResult, list_runs, load_run

DISCLAIMER = (
    "Automated diagnostics — suggestions to investigate, not guarantees; "
    "verify before changing production models."
)
NO_REPORT = "Explanation unavailable — Gemma call failed or was skipped."
BADGE_COLORS = {
    "delayed_response": "blue",
    "missed_peak": "orange",
    "unadapted_shift": "red",
    "over_smoothing": "violet",
    "healthy_noise": "green",
    "other": "gray",
}
CARDS_PER_ROW = 3

_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|<>~$:&])")


def md_escape(text: str) -> str:
    """Backslash-escape markdown/HTML/LaTeX syntax so model-written text renders as plain text."""
    return _MD_SPECIAL.sub(r"\\\1", str(text))


def runs_dir() -> Path:
    return Path(os.environ.get("ATLAS_RUNS_DIR", "runs"))


@st.cache_data(show_spinner="Loading run ...")
def cached_load_run(path: str, mtime: float) -> RunResult:
    """Cache key is the run path (mtime only invalidates when the run is regenerated in place)."""
    return load_run(Path(path))


def load_selected_run(path: Path) -> RunResult | None:
    try:
        return cached_load_run(str(path), (path / "clusters.json").stat().st_mtime)
    except Exception as exc:  # missing/corrupt files: message, not a stack trace
        st.error(f"Could not load run '{path.name}': {type(exc).__name__}: {exc}")
        return None


def cluster_title(c: ClusterInfo) -> str:
    if c.cluster_id == CLUSTER_NOISE:
        return "Unclustered windows"
    return c.report.name if c.report is not None else f"Cluster {c.cluster_id} (no explanation)"


def failure_badge(c: ClusterInfo) -> None:
    if c.report is not None:
        st.badge(c.report.failure_type.replace("_", " "), color=BADGE_COLORS.get(c.report.failure_type, "gray"))
    elif c.cluster_id == CLUSTER_NOISE:
        st.badge("look manually", color="gray")


def sidebar(runs: list[Path]) -> RunResult | None:
    st.sidebar.title("TimeSeriousDoctor")
    if not runs:
        st.sidebar.warning(f"No runs found in `{runs_dir()}`.")
        st.sidebar.markdown(
            "Create a fake run for the demo:\n\n"
            "```\npython scripts/make_mock_run.py\n```\n"
            "or analyse a dataset with `python scripts/run_atlas.py`."
        )
        return None
    by_name = {p.name: p for p in runs}
    name = st.sidebar.selectbox("Run", list(by_name))
    run = load_selected_run(by_name[name])
    if run is None:
        return None
    wins = run.windows
    st.sidebar.markdown("**Run info**")
    st.sidebar.markdown(
        f"- Embedder: `{run.config.get('embedder', 'unknown')}`\n"
        f"- Series: {run.data['series_id'].nunique()}\n"
        f"- Windows: {len(wins)} ({int(wins['gated'].sum())} anomalous)\n"
        f"- Clusters: {sum(c.cluster_id >= 0 for c in run.clusters)}"
    )
    return run


# ---------------------------------------------------------------------------- Atlas

def _open_cluster(cluster_id: int) -> None:
    st.session_state["cluster_id"] = cluster_id
    st.session_state["opened_cluster"] = cluster_id


def tab_atlas(run: RunResult) -> None:
    wins = run.windows
    if run.summary is not None:
        st.subheader(md_escape(run.summary.headline))
        st.markdown(md_escape(run.summary.summary))
        if run.summary.actions:
            st.markdown("**Recommended actions**")
            for action in run.summary.actions:
                st.markdown(f"- {md_escape(action)}")
        st.divider()

    cols = st.columns(5)
    cols[0].metric("Series", run.data["series_id"].nunique())
    cols[1].metric("Windows", len(wins))
    cols[2].metric("Anomalous", int(wins["gated"].sum()))
    cols[3].metric("Clusters", sum(c.cluster_id >= 0 for c in run.clusters))
    cols[4].metric("Unclustered", int((wins["cluster"] == CLUSTER_NOISE).sum()))

    opened = st.session_state.get("opened_cluster")
    if opened is not None:
        st.success(f"Cluster {opened} selected — open the **Cluster** tab to read the explanation.")

    st.subheader("Failure clusters")
    clusters = [c for c in run.clusters if c.cluster_id >= 0]
    bucket = [c for c in run.clusters if c.cluster_id == CLUSTER_NOISE and c.size > 0]
    cards = clusters + bucket
    if not cards:
        st.info("No clusters in this run: no window passed the anomaly gate or none could be grouped.")
        return
    for i in range(0, len(cards), CARDS_PER_ROW):
        for col, c in zip(st.columns(CARDS_PER_ROW), cards[i:i + CARDS_PER_ROW]):
            with col, st.container(border=True):
                st.markdown(f"#### {md_escape(cluster_title(c))}")
                failure_badge(c)
                conf = f" · confidence {c.report.confidence:.0%}" if c.report is not None else ""
                st.caption(f"{c.size} windows · {c.n_series} series{conf}")
                if c.report is not None:
                    st.markdown(md_escape(c.report.summary))
                elif c.cluster_id == CLUSTER_NOISE:
                    st.markdown("Anomalous windows that fit no pattern. Review these by hand.")
                else:
                    st.warning(NO_REPORT)
                st.button("Open", key=f"open_{c.cluster_id}", on_click=_open_cluster, args=(c.cluster_id,))


# -------------------------------------------------------------------------- Cluster

def _feature_table(c: ClusterInfo) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "this cluster": c.feature_median,
            "all anomalous": c.feature_median_all_gated,
        }
    )
    df["difference"] = df["this cluster"] - df["all anomalous"]
    df.index.name = "feature"
    return df.round(3)


def _model_mix_figure(c: ClusterInfo) -> go.Figure:
    mix = sorted(c.model_card_dist.items(), key=lambda kv: -kv[1])
    fig = go.Figure(go.Bar(x=[k for k, _ in mix], y=[v * 100 for _, v in mix], marker_color="#4c78a8"))
    fig.update_layout(
        yaxis_title="Share of windows (%)", xaxis_title=None, margin=dict(l=10, r=10, t=10, b=10), height=300
    )
    return fig


def tab_cluster(run: RunResult) -> None:
    if not run.clusters:
        st.info("This run has no clusters.")
        return
    ids = [c.cluster_id for c in run.clusters]
    if st.session_state.get("cluster_id") not in ids:
        st.session_state["cluster_id"] = ids[0]
    cid = st.selectbox(
        "Cluster",
        ids,
        key="cluster_id",
        format_func=lambda i: f"{i}: {cluster_title(run.cluster(i))}" if i >= 0 else cluster_title(run.cluster(i)),
    )
    c = run.cluster(cid)

    st.header(md_escape(cluster_title(c)))
    failure_badge(c)
    st.caption(f"{c.size} windows · {c.n_series} series")

    r = c.report
    if r is None:
        if c.cluster_id == CLUSTER_NOISE:
            st.info("Gated windows that fit no cluster. There is no automatic explanation: review them manually.")
        else:
            st.warning(NO_REPORT)
    else:
        st.markdown(md_escape(r.summary))
        st.markdown(f"**Confidence:** {r.confidence:.0%}")
        st.markdown("**Evidence**")
        for item in r.evidence:
            st.markdown(f"- {md_escape(item)}")
        st.markdown("**Likely causes**")
        for k, cause in enumerate(r.likely_causes, start=1):
            st.markdown(f"{k}. **{md_escape(cause.cause)}** — {md_escape(cause.why)}")
        st.markdown("**Suggested fixes**")
        for fix in r.suggested_fixes:
            st.markdown(f"- {md_escape(fix)}")
        st.markdown(f"**Metric to monitor:** {md_escape(r.metric_to_monitor)}")

    st.subheader("Example windows")
    img = run.path / c.image_path if (run.path is not None and c.image_path) else None
    if img is not None and img.exists():
        st.image(str(img), width="stretch")
    elif c.image_path:
        st.warning(f"Cluster image not found: {c.image_path}")
    else:
        st.info("No image for this cluster.")

    left, right = st.columns(2)
    with left:
        st.subheader("Features: cluster vs all anomalous")
        if c.feature_median:
            st.dataframe(_feature_table(c), width="stretch")
        else:
            st.info("No feature medians stored for this cluster.")
    with right:
        st.subheader("Model mix")
        if c.model_card_dist:
            st.plotly_chart(_model_mix_figure(c), width="stretch")
        else:
            st.info("No model information for this cluster.")

    st.subheader("Member series")
    members = run.windows[run.windows["cluster"] == cid]
    table = (
        members.groupby("series_id").size().rename("windows").sort_values(ascending=False).reset_index()
    )
    st.dataframe(table, width="stretch", hide_index=True)


# --------------------------------------------------------------- later-wave tabs

def tab_series(run: RunResult) -> None:
    st.info("Coming soon")


def tab_evaluation(run: RunResult) -> None:
    st.info("Coming soon")


# ------------------------------------------------------------------------------ main

def main() -> None:
    st.set_page_config(page_title="TimeSeriousDoctor", layout="wide")
    st.title("TimeSeriousDoctor")
    st.caption("Failure atlas for forecasting models: where, how and why forecasts go wrong.")

    run = sidebar(list_runs(runs_dir()))
    if run is None:
        st.info("Select or create a run in the sidebar to see the atlas.")
    else:
        atlas, cluster, series, evaluation = st.tabs(["Atlas", "Cluster", "Series", "Evaluation"])
        with atlas:
            tab_atlas(run)
        with cluster:
            tab_cluster(run)
        with series:
            tab_series(run)
        with evaluation:
            tab_evaluation(run)

    st.divider()
    st.caption(DISCLAIMER)


main()
