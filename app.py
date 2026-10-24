"""Streamlit entry point.

Atlas, Latent space, Cluster, Series and Evaluation tabs over a precomputed run folder (SPEC.md §7.12).
Runs are produced by the CLI (`python scripts/run_atlas.py`, or `python scripts/make_mock_run.py`
for a fake one); the app only reads them, except the expert review in the Cluster tab, which writes
`expert_feedback.json` into the run folder. Directory with runs: env var ATLAS_RUNS_DIR (default `runs`).
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from atlas.cluster import embed_features, project_tsne
from atlas.features import FEATURES
from atlas.plots import atlas_scatter, cluster_name_map, latent_figure, series_figure, window_figure
from atlas.schema import CLUSTER_NOISE, CLUSTER_NOT_GATED, ClusterInfo, RunResult, list_runs, load_run
from atlas.taxonomy import load_taxonomy
from atlas.ui_theme import apply_theme, plotly_template, theme_toggle

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
EXPERT_FILE = "expert_feedback.json"
EXPERT_VERDICTS = {"Confirm": "confirmed", "Reject": "rejected", "Rename": "renamed"}  # radio label -> status
EXPERT_STATUSES = set(EXPERT_VERDICTS.values())
MOCK_RUN = "mock"
LATENT_MODES = {  # radio label -> mode
    "All windows (feature space)": "all",
    "Anomalous windows (cluster embedding)": "anomalous",
}
LATENT_COLOR_LABELS = {"cluster": "Cluster", "true_label": "True label (synthetic data)", "series_id": "Series"}
PIPELINE_COLOR = "#4c78a8"
GEMMA_ALONE_COLOR = "#e45756"

_MD_SPECIAL = re.compile(r"([\\`*_{}\[\]()#+\-.!|<>~$:&])")


def md_escape(text: str) -> str:
    """Backslash-escape markdown/HTML/LaTeX syntax so model-written text renders as plain text."""
    return _MD_SPECIAL.sub(r"\\\1", str(text))


def runs_dir() -> Path:
    return Path(os.environ.get("ATLAS_RUNS_DIR", "runs"))


def run_stamp(path: Path) -> float:
    """Newest mtime of any file in the run folder (eval.json, summary.json, expert_feedback.json, ...)."""
    return max((p.stat().st_mtime for p in Path(path).iterdir() if p.is_file()), default=0.0)


@st.cache_data(show_spinner="Loading run ...")
def cached_load_run(path: str, stamp: float) -> RunResult:
    """Cache key is the run path plus the newest file mtime, so files written later show up."""
    return load_run(Path(path))


def load_selected_run(path: Path) -> RunResult | None:
    try:
        return cached_load_run(str(path), run_stamp(path))
    except Exception as exc:  # missing/corrupt files: message, not a stack trace
        st.error(f"Could not load run '{path.name}': {type(exc).__name__}: {exc}")
        return None


def expert_entry(run: RunResult, cluster_id: int) -> dict | None:
    """The saved expert verdict for a cluster (run.expert_feedback), or None."""
    entry = (run.expert_feedback or {}).get(str(cluster_id))
    return entry if isinstance(entry, dict) and entry.get("status") in EXPERT_STATUSES else None


def gemma_title(c: ClusterInfo) -> str:
    if c.cluster_id == CLUSTER_NOISE:
        return "Unclustered windows"
    return c.report.name if c.report is not None else f"Cluster {c.cluster_id} (no explanation)"


def cluster_title(c: ClusterInfo, expert: dict | None = None) -> str:
    """Display name: the expert's name for a renamed cluster, else Gemma's."""
    if expert is not None and expert["status"] == "renamed" and expert.get("name"):
        return str(expert["name"])
    return gemma_title(c)


def run_title(run: RunResult, c: ClusterInfo) -> str:
    return cluster_title(c, expert_entry(run, c.cluster_id))


def failure_badge(c: ClusterInfo, expert: dict | None = None) -> None:
    ftype = c.report.failure_type if c.report is not None else None
    if expert is not None and expert["status"] != "rejected" and expert.get("failure_type"):
        ftype = expert["failure_type"]
    if ftype is not None:
        st.badge(str(ftype).replace("_", " "), color=BADGE_COLORS.get(ftype, "gray"))
    elif c.cluster_id == CLUSTER_NOISE:
        st.badge("look manually", color="gray")


def expert_badge(expert: dict | None, with_name: bool = False) -> None:
    """Badge for the saved expert verdict (nothing if there is none)."""
    if expert is None:
        return
    if expert["status"] == "confirmed":
        st.badge("✔ Expert-confirmed", color="green")
    elif expert["status"] == "rejected":
        st.badge("✖ Rejected by expert", color="red")
    else:
        name = str(expert.get("name", "")).replace("[", "(").replace("]", ")")  # brackets would end the badge
        st.badge(f"✎ Renamed by expert: {md_escape(name)}" if with_name else "✎ Renamed by expert", color="violet")


def default_run_name(runs: list[Path]) -> str:
    """Most recently modified real run; the mock demo run only when it is the only one."""
    real = [p for p in runs if p.name != MOCK_RUN]
    return max(real or runs, key=run_stamp).name


def sidebar(runs: list[Path]) -> RunResult | None:
    if not runs:
        st.sidebar.warning(f"No runs found in `{runs_dir()}`.")
        st.sidebar.markdown(
            "Create a fake run for the demo:\n\n"
            "```\npython scripts/make_mock_run.py\n```\n"
            "or analyse a dataset with `python scripts/run_atlas.py`."
        )
        return None
    by_name = {p.name: p for p in runs}
    if st.session_state.get("run_name") not in by_name:
        st.session_state["run_name"] = default_run_name(runs)
    name = st.sidebar.selectbox(
        "Run", list(by_name), key="run_name",
        format_func=lambda n: "mock (demo data)" if n == MOCK_RUN else n,
    )
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


def plot(fig: go.Figure, **kwargs):
    """st.plotly_chart that keeps the figure's own plotly template (theme=None) in light and dark mode."""
    return st.plotly_chart(fig, width="stretch", theme=None, **kwargs)


# ---------------------------------------------------------------------------- Atlas

def _open_cluster(cluster_id: int) -> None:
    st.session_state["cluster_id"] = cluster_id
    st.session_state["opened_cluster"] = cluster_id


def tab_atlas(run: RunResult, tpl: str) -> None:
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
                expert = expert_entry(run, c.cluster_id)
                st.markdown(f"#### {md_escape(cluster_title(c, expert))}")
                if expert is not None and expert["status"] == "renamed":
                    st.caption(f"Gemma: {md_escape(gemma_title(c))}")
                failure_badge(c, expert)
                expert_badge(expert)
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
        plot(atlas_scatter(wins, template=tpl))


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


def _model_mix_figure(c: ClusterInfo, tpl: str) -> go.Figure:
    mix = sorted(c.model_card_dist.items(), key=lambda kv: -kv[1])
    fig = go.Figure(go.Bar(x=[k for k, _ in mix], y=[v * 100 for _, v in mix], marker_color="#4c78a8"))
    fig.update_layout(
        yaxis_title="Share of windows (%)", xaxis_title=None, margin=dict(l=10, r=10, t=10, b=10), height=300,
        template=tpl,
    )
    return fig


def _goto_series(series_id: str) -> None:
    st.session_state["series_id"] = series_id
    st.session_state["opened_series"] = series_id


def _show_series(key: str) -> None:
    _goto_series(st.session_state[key])


def _save_expert_review(run_path: Path, c: ClusterInfo) -> None:
    """Button callback: upsert this cluster's verdict in expert_feedback.json, then drop the loader cache."""
    cid, ss = c.cluster_id, st.session_state
    status = EXPERT_VERDICTS[ss[f"expert_verdict_{cid}"]]
    gemma_name = c.report.name if c.report is not None else ""
    gemma_type = c.report.failure_type if c.report is not None else "other"
    entry = {
        "status": status,
        "name": ss[f"expert_name_{cid}"].strip() if status == "renamed" else gemma_name,
        "failure_type": gemma_type if status == "rejected" else ss[f"expert_type_{cid}"],
        "note": ss[f"expert_note_{cid}"].strip(),
        "gemma_name": gemma_name,
        "summary": c.report.summary if c.report is not None else "",
        "feature_median": c.feature_median,
        "feature_median_all_gated": c.feature_median_all_gated,
    }
    path = Path(run_path) / EXPERT_FILE
    feedback = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    feedback[str(cid)] = entry
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(feedback, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    cached_load_run.clear()
    ss["expert_saved"] = cid


def expert_review(run: RunResult, c: ClusterInfo, saved: dict | None) -> None:
    """Expert review box: confirm / reject / rename Gemma's diagnosis (SPEC P2.2)."""
    if c.cluster_id < 0 or run.path is None:  # the unclustered bucket has no diagnosis to review
        return
    cid = c.cluster_id
    types = list(load_taxonomy())
    if "other" not in types:
        types.append("other")
    default_type = (saved or {}).get("failure_type") or (c.report.failure_type if c.report is not None else "other")
    if default_type not in types:
        types.append(default_type)
    verdicts = list(EXPERT_VERDICTS) if c.report is not None else ["Rename"]  # nothing to confirm without a report
    saved_verdict = next((k for k, v in EXPERT_VERDICTS.items() if saved and v == saved["status"]), None)

    with st.container(border=True):
        st.markdown("#### Expert review")
        st.caption(
            "Confirm, reject or rename Gemma's diagnosis. Reviewed clusters become reference examples for Gemma "
            f"when you re-run with `--expert-from {run.path}`."
        )
        verdict = st.radio(
            "Verdict", verdicts, key=f"expert_verdict_{cid}", horizontal=True,
            index=verdicts.index(saved_verdict) if saved_verdict in verdicts else 0,
        )
        name = ""
        if verdict == "Rename":
            name = st.text_input(
                "New name", key=f"expert_name_{cid}", max_chars=80,
                value=(saved or {}).get("name", "") if saved_verdict == "Rename" else "",
            )
        if verdict != "Reject":
            st.selectbox("Failure type", types, key=f"expert_type_{cid}", index=types.index(default_type))
        st.text_area("Note (optional)", key=f"expert_note_{cid}", max_chars=500, value=(saved or {}).get("note", ""))
        st.button(
            "Save expert review", key=f"expert_save_{cid}", type="primary",
            disabled=verdict == "Rename" and not name.strip(),
            on_click=_save_expert_review, args=(run.path, c),
        )
        if st.session_state.pop("expert_saved", None) == cid:
            st.success("Saved. The verdict is shown on the cluster card and in the Cluster tab header.")


def tab_cluster(run: RunResult, tpl: str) -> None:
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
        format_func=lambda i: f"{i}: {run_title(run, run.cluster(i))}" if i >= 0 else run_title(run, run.cluster(i)),
    )
    c = run.cluster(cid)
    expert = expert_entry(run, cid)

    st.header(md_escape(cluster_title(c, expert)))
    failure_badge(c, expert)
    expert_badge(expert, with_name=True)
    if expert is not None and expert["status"] == "renamed":
        st.caption(f"Gemma's name: {md_escape(gemma_title(c))}")
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

    expert_review(run, c, expert)

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
            plot(_model_mix_figure(c, tpl))
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


# ------------------------------------------------------------------------ Latent space

def latent_coords(run: RunResult, mode: str) -> np.ndarray | None:
    """t-SNE coordinates [n_windows, 2] (NaN = not projected), or None if the run lacks the inputs.

    mode "all": every window from its 10 features; "anomalous": the cluster embedding of the gated windows.
    """
    w = run.windows
    if mode == "all":
        feats = [f for f in FEATURES if f in w.columns]  # older runs carry fewer features
        if not feats:
            return None
        return project_tsne(embed_features(w, feats), keep=w["gated"].to_numpy(dtype=bool))
    emb = np.asarray(run.embeddings)
    if emb.ndim != 2 or len(emb) == 0:
        return None
    xy = project_tsne(emb)
    rows = w["emb_row"].to_numpy(dtype=int)
    ok = (rows >= 0) & (rows < len(xy))
    out = np.full((len(w), 2), np.nan, dtype=np.float32)
    out[ok] = xy[rows[ok]]
    return out


@st.cache_data(show_spinner="Computing the t-SNE map (about 15 s for thousands of windows) ...")
def cached_tsne(_run: RunResult, path: str, mode: str, stamp: float) -> np.ndarray | None:
    """t-SNE is slow: cached per run folder + mode (+ newest file mtime); `_run` itself is not hashed."""
    return latent_coords(_run, mode)


def _cluster_names(run: RunResult) -> dict[int, str]:
    """Cluster id -> display name: Gemma's, or the expert's for a renamed cluster."""
    names = cluster_name_map(run.clusters)
    for c in run.clusters:
        if c.cluster_id >= 0 and expert_entry(run, c.cluster_id) is not None:
            names[c.cluster_id] = run_title(run, c)
    return names


def _selected_window(event, windows: pd.DataFrame) -> str | None:
    """window_id of the first selected point (its customdata) that exists in this run."""
    points = ((event or {}).get("selection") or {}).get("points") or []
    known = set(windows["window_id"])
    for p in points:
        cd = p.get("customdata")
        if isinstance(cd, (list, tuple)):
            cd = cd[0] if cd else None
        if isinstance(cd, str) and cd in known:
            return cd
    return None


def _window_detail(run: RunResult, window_id: str, tpl: str) -> None:
    row = run.windows[run.windows["window_id"] == window_id].iloc[0]
    sid, cid = str(row["series_id"]), int(row["cluster"])
    plot(window_figure(window_id, run, template=tpl))
    st.button(f"Show {sid} in Series tab", key="latent_show_series", on_click=_goto_series, args=(sid,))
    if cid == CLUSTER_NOT_GATED:
        st.markdown("Normal window: it did not pass the anomaly gate.")
        return
    try:
        c = run.cluster(cid)
    except KeyError:
        return
    expert = expert_entry(run, cid)
    st.markdown(f"**{md_escape(cluster_title(c, expert))}**")
    failure_badge(c, expert)
    expert_badge(expert)
    if c.report is not None:
        st.markdown(md_escape(c.report.summary))
    elif cid == CLUSTER_NOISE:
        st.markdown("Anomalous window that fits no pattern. Review it by hand.")
    else:
        st.warning(NO_REPORT)
    if cid >= 0:
        st.button("Open this cluster", key="latent_open_cluster", on_click=_open_cluster, args=(cid,))


def tab_latent(run: RunResult, tpl: str) -> None:
    wins = run.windows
    st.caption(
        "Each dot is one 28-step window of forecast errors, placed by t-SNE so that windows with a similar "
        "error pattern sit close together. Click a dot to see that window."
    )
    mode_label = st.radio("Show", list(LATENT_MODES), horizontal=True, key="latent_mode")
    coords = cached_tsne(run, str(run.path), LATENT_MODES[mode_label], run_stamp(run.path))
    if coords is None:
        st.info("This run has no stored features or embeddings for this view.")
        return

    ids, counts = _series_order(run)
    color_options = ["cluster", *(["true_label"] if "true_label" in wins.columns else []), "series_id"]
    c1, c2 = st.columns(2)
    color_by = c1.selectbox("Color by", color_options, key="latent_color", format_func=LATENT_COLOR_LABELS.get)
    highlight = c2.selectbox(
        "Highlight a series", [None, *ids], key="latent_highlight",
        format_func=lambda s: "(none)" if s is None else f"{s} ({counts[s]} anomalous)",
    )
    fig = latent_figure(
        wins, coords, color_by=color_by, cluster_names=_cluster_names(run), highlight_series=highlight, template=tpl
    )

    left, right = st.columns([3, 2])
    with left:
        event = plot(fig, key="latent_plot", on_select="rerun", selection_mode="points")
    with right:
        window_id = _selected_window(event, wins)
        if window_id is None:
            st.info("Click a point on the map to see that window's forecast and its cluster's explanation.")
        else:
            _window_detail(run, window_id, tpl)


# --------------------------------------------------------------------------- Series

def _window_cluster_name(run: RunResult, cluster_id: int) -> str:
    if cluster_id == CLUSTER_NOT_GATED:
        return "not anomalous"
    try:
        return run_title(run, run.cluster(cluster_id))
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


def _series_order(run: RunResult) -> tuple[list[str], dict[str, int]]:
    """Series ids sorted by number of anomalous windows (desc, then id) and the counts."""
    gated = run.windows[run.windows["gated"].astype(bool)].groupby("series_id").size()
    counts = {sid: int(gated.get(sid, 0)) for sid in run.data["series_id"].unique()}
    return sorted(counts, key=lambda sid: (-counts[sid], sid)), counts


def _series_latent(run: RunResult, series_id: str, tpl: str) -> None:
    """Small t-SNE map with this series' windows outlined."""
    coords = cached_tsne(run, str(run.path), "all", run_stamp(run.path))
    if coords is None:
        return
    fig = latent_figure(run.windows, coords, cluster_names=_cluster_names(run), highlight_series=series_id, template=tpl)
    fig.update_layout(
        title="Where this series sits in the latent space", height=340, showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    plot(fig)


def tab_series(run: RunResult, tpl: str) -> None:
    ids, counts = _series_order(run)
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
    plot(series_figure(sid, run, template=tpl))

    st.subheader("Windows")
    table = _series_windows_table(run, sid)
    left, right = st.columns([3, 2])
    with left:
        if table.empty:
            st.info("No forecast windows for this series.")
        else:
            st.dataframe(table, width="stretch", hide_index=True)
    with right:
        _series_latent(run, sid, tpl)


# ---------------------------------------------------------------------- Evaluation

def _fmt(value: float | None, pct: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:.0%}" if pct else f"{value:.2f}"


def _label(name: str) -> str:
    return name.replace("_", " ")


def _comparison_figure(comp: dict, tpl: str) -> go.Figure:
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
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), template=tpl,
    )
    return fig


def _headline(comp: dict, tpl: str) -> None:
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
    plot(_comparison_figure(comp, tpl))


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


def _confusion_figure(run: RunResult, confusion: dict, tpl: str) -> go.Figure:
    cids = sorted(confusion, key=lambda k: (int(k) < 0, int(k)))  # unclustered bucket last
    labels = sorted({lab for row in confusion.values() for lab in row}, key=lambda x: (x == "healthy", x))
    rows = []
    for k in cids:
        try:
            title = run_title(run, run.cluster(int(k)))
            rows.append(f"{k}: {title}" if int(k) >= 0 else title)
        except KeyError:
            rows.append(f"Cluster {k}")
    z = [[confusion[k].get(lab, 0) for lab in labels] for k in cids]
    fig = px.imshow(
        z, x=[_label(lab) for lab in labels], y=rows, text_auto=True, aspect="auto",
        color_continuous_scale="Blues", labels=dict(x="true label", y="cluster", color="windows"),
    )
    fig.update_layout(margin=dict(l=10, r=10, t=10, b=10), height=120 + 60 * len(rows), template=tpl)
    return fig


def tab_evaluation(run: RunResult, tpl: str) -> None:
    ev = run.eval
    if not ev:
        st.info("No eval.json for this run — run scripts/run_eval.py")
        return

    comp = ev.get("comparison")
    if comp and comp.get("pipeline") and comp.get("gemma_alone"):
        _headline(comp, tpl)
        st.divider()
    _pipeline_metrics(ev)

    confusion = (ev.get("clustering") or {}).get("confusion")
    if confusion:
        st.subheader("Clusters vs true failure types")
        st.caption("Windows per cluster and true label; a clean heatmap has one dark cell per row.")
        plot(_confusion_figure(run, confusion, tpl))


# ------------------------------------------------------------------------------ main

def main() -> None:
    st.set_page_config(page_title="TimeSeriousDoctor", layout="wide")
    st.title("TimeSeriousDoctor")
    st.caption("Failure atlas for forecasting models: where, how and why forecasts go wrong.")

    st.sidebar.title("TimeSeriousDoctor")
    dark = theme_toggle()
    apply_theme(dark)
    tpl = plotly_template(dark)

    run = sidebar(list_runs(runs_dir()))
    if run is None:
        st.info("Select or create a run in the sidebar to see the atlas.")
    else:
        atlas, latent, cluster, series, evaluation = st.tabs(
            ["Atlas", "Latent space", "Cluster", "Series", "Evaluation"]
        )
        with atlas:
            tab_atlas(run, tpl)
        with latent:
            tab_latent(run, tpl)
        with cluster:
            tab_cluster(run, tpl)
        with series:
            tab_series(run, tpl)
        with evaluation:
            tab_evaluation(run, tpl)

    st.divider()
    st.caption(DISCLAIMER)


main()
