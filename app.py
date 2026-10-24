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
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from atlas.plots import atlas_scatter, series_figure
from atlas.schema import CLUSTER_NOISE, CLUSTER_NOT_GATED, ClusterInfo, RunResult, list_runs, load_run

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
PIPELINE_COLOR = "#4c78a8"
GEMMA_ALONE_COLOR = "#e45756"

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

    if {"x2d", "y2d"} <= set(wins.columns) and wins["x2d"].notna().any():
        st.subheader("Atlas map")
        st.caption("Each dot is an anomalous window; windows with a similar failure pattern sit close together.")
        st.plotly_chart(atlas_scatter(wins), width="stretch")


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


def _show_series(key: str) -> None:
    st.session_state["series_id"] = st.session_state[key]
    st.session_state["opened_series"] = st.session_state[key]


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
    if not table.empty:
        key = f"member_series_{cid}"
        counts = dict(zip(table["series_id"], table["windows"]))
        pick, button = st.columns([3, 1], vertical_alignment="bottom")
        pick.selectbox(
            "Member series", table["series_id"].tolist(), key=key, format_func=lambda s: f"{s} ({counts[s]} windows)"
        )
        button.button("Show in Series tab", key=f"show_series_{cid}", on_click=_show_series, args=(key,))
        opened = st.session_state.get("opened_series")
        if opened is not None:
            st.success(f"Series {opened} selected — open the **Series** tab to see its forecast and windows.")


# --------------------------------------------------------------------------- Series

def _window_cluster_name(run: RunResult, cluster_id: int) -> str:
    if cluster_id == CLUSTER_NOT_GATED:
        return "not anomalous"
    try:
        return cluster_title(run.cluster(cluster_id))
    except KeyError:
        return f"Cluster {cluster_id}"


def _series_windows_table(run: RunResult, series_id: str) -> pd.DataFrame:
    w = run.windows[run.windows["series_id"] == series_id].sort_values("start_idx")
    err = "abs_err" if "abs_err" in w.columns else "score_abs"
    bias = "bias" if "bias" in w.columns else "score_bias"
    return pd.DataFrame(
        {
            "window_id": w["window_id"],
            "start_ts": w["start_ts"],
            "cluster": w["cluster"],
            "cluster name": [_window_cluster_name(run, int(c)) for c in w["cluster"]],
            "abs_err": w[err].round(3),
            "bias": w[bias].round(3),
        }
    )


def tab_series(run: RunResult) -> None:
    gated = run.windows[run.windows["gated"].astype(bool)].groupby("series_id").size()
    counts = {sid: int(gated.get(sid, 0)) for sid in run.data["series_id"].unique()}
    ids = sorted(counts, key=lambda sid: (-counts[sid], sid))
    if not ids:
        st.info("This run has no series.")
        return
    if st.session_state.get("series_id") not in ids:
        st.session_state["series_id"] = ids[0]
    sid = st.selectbox("Series", ids, key="series_id", format_func=lambda s: f"{s} ({counts[s]} anomalous)")

    cards = run.series_meta.loc[run.series_meta["series_id"] == sid, "model_card"]
    n_windows = int((run.windows["series_id"] == sid).sum())
    model = f"Model: {cards.iloc[0]} · " if len(cards) else ""
    st.caption(md_escape(f"{model}{counts[sid]} anomalous of {n_windows} windows"))
    st.plotly_chart(series_figure(sid, run), width="stretch")

    st.subheader("Windows")
    table = _series_windows_table(run, sid)
    if table.empty:
        st.info("No forecast windows for this series.")
    else:
        st.dataframe(table, width="stretch", hide_index=True)


# ---------------------------------------------------------------------- Evaluation

def _fmt(value: float | None, pct: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:.0%}" if pct else f"{value:.2f}"


def _label(name: str) -> str:
    return name.replace("_", " ")


def _comparison_figure(comp: dict) -> go.Figure:
    pipe, alone = comp["pipeline"], comp["gemma_alone"]
    classes = list(pipe.get("per_class", {}))
    classes += [c for c in alone.get("per_class", {}) if c not in classes]

    def accuracies(m: dict) -> list[float | None]:
        vals = [m.get("accuracy")] + [m.get("per_class", {}).get(c) for c in classes]
        return [None if v is None else v * 100 for v in vals]

    cats = ["overall"] + [_label(c) for c in classes]
    fig = go.Figure()
    for name, m, color in (
        ("TimeSeriousDoctor", pipe, PIPELINE_COLOR),
        ("Gemma alone", alone, GEMMA_ALONE_COLOR),
    ):
        vals = accuracies(m)
        fig.add_trace(go.Bar(
            name=name, x=cats, y=vals, marker_color=color,
            text=[None if v is None else f"{v:.0f}%" for v in vals], textposition="outside",
        ))
    fig.update_layout(
        barmode="group", yaxis_title="Accuracy (%)", yaxis_range=[0, 112], xaxis_title=None,
        margin=dict(l=10, r=10, t=30, b=10), height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def _headline(comp: dict) -> None:
    pipe, alone = comp["pipeline"], comp["gemma_alone"]
    n = comp.get("n_series")
    sample = f"{n}-series sample" if n is not None else "sample"
    st.subheader("TimeSeriousDoctor vs Gemma alone")
    st.caption(
        f"Same Gemma model on the same {sample} of series with known failures: diagnosing each series' "
        "failure type with the full pipeline vs asking Gemma on the raw forecast plot."
    )

    left, right = st.columns(2)
    p_acc, g_acc = pipe.get("accuracy"), alone.get("accuracy")
    delta = f"{(p_acc - g_acc) * 100:+.0f} points vs Gemma alone" if None not in (p_acc, g_acc) else None
    left.metric("TimeSeriousDoctor accuracy", _fmt(p_acc, pct=True), delta=delta)
    right.metric("Gemma alone accuracy", _fmt(g_acc, pct=True))
    st.caption(f"Macro-F1: {_fmt(pipe.get('macro_f1'))} vs {_fmt(alone.get('macro_f1'))}")

    full = comp.get("gemma_calls_full_dataset") or {}
    c_sample, c_full = st.columns(2)
    for col, label, p_calls, g_calls in (
        (c_sample, f"Gemma calls on the {sample}", pipe.get("gemma_calls"), alone.get("gemma_calls")),
        (c_full, "Gemma calls on the full dataset", full.get("pipeline"), full.get("gemma_alone")),
    ):
        if p_calls is None or g_calls is None:
            continue
        ratio = f"{g_calls / p_calls:.0f}x fewer calls" if p_calls > 0 and g_calls > p_calls else None
        col.metric(
            label, f"{p_calls} vs {g_calls}", delta=ratio, delta_color="off", delta_arrow="off",
            help="TimeSeriousDoctor vs Gemma alone",
        )
    st.plotly_chart(_comparison_figure(comp), width="stretch")


def _pipeline_metrics(ev: dict) -> None:
    gate = ev.get("gate") or {}
    clu = ev.get("clustering") or {}
    st.subheader("Anomaly gate")
    cols = st.columns(3)
    cols[0].metric("Precision", _fmt(gate.get("precision")), help="Share of flagged windows that really fail.")
    cols[1].metric("Recall", _fmt(gate.get("recall")), help="Share of failing windows that were flagged.")
    cols[2].metric("F1", _fmt(gate.get("f1")))

    st.subheader("Clustering")
    cols = st.columns(4)
    cols[0].metric("ARI", _fmt(clu.get("ari")), help="Adjusted Rand index vs the true failure types (1 = perfect).")
    cols[1].metric("NMI", _fmt(clu.get("nmi")), help="Normalized mutual information vs the true failure types.")
    cols[2].metric(
        "Purity", _fmt(clu.get("purity"), pct=True), help="Share of windows in the majority true type of their cluster."
    )
    cols[3].metric("Noise share", _fmt(clu.get("noise_share"), pct=True), help="Gated windows that fit no cluster.")

    st.subheader("Diagnosis")
    cols = st.columns(2)
    cols[0].metric(
        "Naming accuracy", _fmt(ev.get("naming_accuracy"), pct=True),
        help="Clusters whose Gemma failure type matches the majority true type.",
    )
    cols[1].metric(
        "Series accuracy", _fmt(ev.get("series_accuracy"), pct=True),
        help="Series whose failure type is diagnosed correctly via their cluster.",
    )


def _confusion_figure(run: RunResult, confusion: dict) -> go.Figure:
    cids = sorted(confusion, key=lambda k: (int(k) < 0, int(k)))  # unclustered bucket last
    labels = sorted({lab for row in confusion.values() for lab in row}, key=lambda x: (x == "healthy", x))
    rows = []
    for k in cids:
        try:
            title = cluster_title(run.cluster(int(k)))
            rows.append(f"{k}: {title}" if int(k) >= 0 else title)
        except KeyError:
            rows.append(f"Cluster {k}")
    z = [[confusion[k].get(lab, 0) for lab in labels] for k in cids]
    fig = px.imshow(
        z, x=[_label(lab) for lab in labels], y=rows, text_auto=True, aspect="auto",
        color_continuous_scale="Blues", labels=dict(x="true label", y="cluster", color="windows"),
    )
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=120 + 60 * len(rows))
    return fig


def tab_evaluation(run: RunResult) -> None:
    ev = run.eval
    if not ev:
        st.info("No eval.json for this run — run scripts/run_eval.py")
        return

    comp = ev.get("comparison")
    if comp and comp.get("pipeline") and comp.get("gemma_alone"):
        _headline(comp)
        st.divider()
    _pipeline_metrics(ev)

    confusion = (ev.get("clustering") or {}).get("confusion")
    if confusion:
        st.subheader("Clusters vs true failure types")
        st.caption("Windows per cluster and true label; a clean heatmap has one dark cell per row.")
        st.plotly_chart(_confusion_figure(run, confusion), width="stretch")


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
