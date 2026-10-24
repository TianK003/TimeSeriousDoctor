"""Matplotlib PNGs for Gemma, plotly figures for the UI.

Owner: C. Task: P0.11. Spec: SPEC.md §7.8.

The cluster panel and the baseline plot are drawn with matplotlib's Agg canvas directly (no pyplot),
so they are safe to call from Streamlit threads and leave no figures open.
"""

from __future__ import annotations

import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from atlas.schema import CLUSTER_NOISE, CLUSTER_NOT_GATED, ClusterInfo, RunResult

# d3 category10 without its grey, which is reserved for noise / unclustered windows.
CLUSTER_COLORS: list[str] = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b",
    "#e377c2", "#bcbd22", "#17becf", "#d62728",
]
NOISE_COLOR = "#8c8c8c"

ACTUAL_COLOR = "black"
PRED_COLOR = "#d62728"
CONTEXT_COLOR = "#9e9e9e"
WINDOW_COLOR = "#ffe9a8"
RESID_COLOR = "#1f6fb2"
MEDIAN_COLOR = "#0b2e59"


def _cluster_color(cluster_id: int) -> str:
    if cluster_id < 0:
        return NOISE_COLOR
    return CLUSTER_COLORS[cluster_id % len(CLUSTER_COLORS)]


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def _short(text: str, n: int = 28) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


def _png_bytes(fig: Figure) -> bytes:
    FigureCanvasAgg(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=fig.dpi)
    return buf.getvalue()


# ---------------------------------------------------------------- cluster panel (what Gemma sees)

def cluster_panel_png(
    cluster: ClusterInfo, windows: pd.DataFrame, res: pd.DataFrame, path: Path, context: int = 14
) -> Path:
    """1200x900 px, 2x2: up to 3 medoid windows (scaled actual vs predicted) + residuals of member windows."""
    path = Path(path)
    win_by_id = windows.set_index("window_id", drop=False)
    medoids = [win_by_id.loc[w] for w in cluster.medoid_window_ids[:3] if w in win_by_id.index]

    members = windows[windows["cluster"] == cluster.cluster_id]
    if len(members) > 40:
        members = members.iloc[np.random.default_rng(0).choice(len(members), size=40, replace=False)]
        members = members.sort_values(["series_id", "start_idx"])

    needed = {str(w["series_id"]) for w in medoids} | set(members["series_id"].astype(str))
    series = {
        sid: g.sort_values("step")
        for sid, g in res[res["series_id"].isin(needed)].groupby("series_id", sort=False)
    }

    fig = Figure(figsize=(12, 9), dpi=100, layout="constrained")
    axes = fig.subplots(2, 2).ravel()
    win_len = max([int(w["end_idx"] - w["start_idx"]) for w in medoids] or [28])

    for k, ax in enumerate(axes[:3]):
        if k >= len(medoids):
            ax.set_visible(False)
            continue
        w = medoids[k]
        sid = str(w["series_id"])
        g = series[sid]
        start, length = int(w["start_idx"]), int(w["end_idx"] - w["start_idx"])
        step0 = int(g.loc[g["fstep"] == start, "step"].iloc[0])
        seg = g[(g["step"] >= step0 - context) & (g["step"] < step0 + length)]
        x = seg["step"].to_numpy() - step0

        ax.axvspan(-context - 0.5, -0.5, color=CONTEXT_COLOR, alpha=0.18, lw=0)
        ax.axvspan(-0.5, length - 0.5, color=WINDOW_COLOR, alpha=0.6, lw=0)
        ax.plot(x, seg["a_s"].to_numpy(), color=ACTUAL_COLOR, lw=1.6)
        ax.plot(x, seg["p_s"].to_numpy(), color=PRED_COLOR, lw=1.6)
        ax.set_xlim(-context - 0.5, win_len - 0.5)
        ax.set_title(f"Example {k + 1} — series {_short(sid)}", fontsize=11)
        ax.set_xlabel("steps (0 = window start)", fontsize=9)
        ax.set_ylabel("scaled value", fontsize=9)

    # Panel 4: residual curves of member windows.
    ax = axes[3]
    curves = []
    for _, w in members.iterrows():
        g = series.get(str(w["series_id"]))
        if g is None:
            continue
        rows = g[(g["fstep"] >= w["start_idx"]) & (g["fstep"] < w["end_idx"])].sort_values("fstep")
        curves.append(rows["r_s"].to_numpy(dtype=float))
    if curves:
        n = max(len(c) for c in curves)
        mat = np.full((len(curves), n), np.nan)
        for i, c in enumerate(curves):
            mat[i, : len(c)] = c
        xs = np.arange(n)
        for row in mat:
            ax.plot(xs, row, color=RESID_COLOR, alpha=0.2, lw=0.8)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            median = np.nanmedian(mat, axis=0)
        ax.plot(xs, median, color=MEDIAN_COLOR, lw=2.6)
        ax.set_xlim(-0.5, n - 0.5)
    ax.axhline(0, color="#555555", lw=1.0, ls="--")
    ax.set_title("Residuals of member windows (r = actual − predicted)", fontsize=11)
    ax.set_xlabel("steps", fontsize=9)
    ax.set_ylabel("scaled value", fontsize=9)

    for ax in axes:
        ax.tick_params(labelsize=8)
        ax.grid(alpha=0.25, lw=0.5)

    handles = [
        Line2D([], [], color=ACTUAL_COLOR, lw=1.6, label="actual"),
        Line2D([], [], color=PRED_COLOR, lw=1.6, label="predicted"),
        Patch(color=CONTEXT_COLOR, alpha=0.18, label="context before window"),
        Patch(color=WINDOW_COLOR, alpha=0.6, label="window"),
        Line2D([], [], color=RESID_COLOR, alpha=0.5, lw=1.0, label="member window residual"),
        Line2D([], [], color=MEDIAN_COLOR, lw=2.6, label="median residual"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=6, fontsize=8, frameon=False)
    name = "Unclustered" if cluster.cluster_id == CLUSTER_NOISE else f"Cluster {cluster.cluster_id}"
    fig.suptitle(f"{name} — {cluster.size} windows from {cluster.n_series} series", fontsize=14, fontweight="bold")

    path.parent.mkdir(parents=True, exist_ok=True)
    FigureCanvasAgg(fig)
    fig.savefig(path, format="png", dpi=fig.dpi)
    return path


# ---------------------------------------------------------------- Gemma-alone baseline input

def baseline_series_png(series: pd.DataFrame, path: Path | None = None, title: str = "") -> bytes:
    """Full series, actual vs predicted in raw units, forecast region shaded. No residuals, no features."""
    series = series.sort_values("ts")
    fig = Figure(figsize=(10, 4.5), dpi=100, layout="constrained")
    ax = fig.subplots()

    fc = series[series["is_forecast"].astype(bool)]
    if len(fc):
        ax.axvspan(fc["ts"].iloc[0], fc["ts"].iloc[-1], color=CONTEXT_COLOR, alpha=0.22, lw=0)
    ax.plot(series["ts"], series["actual"], color=ACTUAL_COLOR, lw=1.4)
    ax.plot(series["ts"], series["predicted"], color=PRED_COLOR, lw=1.4)
    ax.set_xlabel("time", fontsize=9)
    ax.set_ylabel("value", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.25, lw=0.5)
    if title:
        ax.set_title(title, fontsize=12)

    handles = [
        Line2D([], [], color=ACTUAL_COLOR, lw=1.4, label="actual"),
        Line2D([], [], color=PRED_COLOR, lw=1.4, label="predicted"),
        Patch(color=CONTEXT_COLOR, alpha=0.22, label="forecast period"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, fontsize=8, frameon=False)

    png = _png_bytes(fig)
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(png)
    return png


# ---------------------------------------------------------------- plotly figures for the UI

def series_figure(series_id: str, run: RunResult, template: str = "plotly_white") -> go.Figure:
    """Full series actual/predicted, forecast region grey, gated windows shaded by cluster with hover."""
    s = run.data[run.data["series_id"] == series_id].sort_values("ts")
    fig = go.Figure()

    lo = float(np.nanmin(s[["actual", "predicted"]].to_numpy()))
    hi = float(np.nanmax(s[["actual", "predicted"]].to_numpy()))
    pad = 0.06 * (hi - lo or 1.0)
    lo, hi = lo - pad, hi + pad

    w = run.windows
    w = w[(w["series_id"] == series_id) & w["gated"].astype(bool)]
    err_col = "abs_err" if "abs_err" in w.columns else "score_abs"
    seen: set[int] = set()
    for _, row in w.iterrows():
        cid = int(row["cluster"])
        color = _cluster_color(cid)
        label = "unclustered" if cid == CLUSTER_NOISE else f"cluster {cid}"
        s0, e0 = row["start_ts"], row["end_ts"]
        fig.add_trace(go.Scatter(
            x=[s0, e0, e0, s0, s0], y=[lo, lo, hi, hi, lo],
            mode="lines", line=dict(width=0), fill="toself", fillcolor=_rgba(color, 0.25),
            hoveron="fills", hoverinfo="text",
            text=f"{row['window_id']}<br>cluster: {cid}<br>abs_err: {float(row[err_col]):.2f}",
            name=label, legendgroup=label, showlegend=cid not in seen,
        ))
        seen.add(cid)

    fig.add_trace(go.Scatter(x=s["ts"], y=s["actual"], mode="lines", name="actual",
                             line=dict(color=ACTUAL_COLOR, width=1.5)))
    fig.add_trace(go.Scatter(x=s["ts"], y=s["predicted"], mode="lines", name="predicted",
                             line=dict(color=PRED_COLOR, width=1.5)))

    fc = s[s["is_forecast"].astype(bool)]
    if len(fc):
        fig.add_shape(
            type="rect", xref="x", yref="paper", x0=fc["ts"].iloc[0], x1=fc["ts"].iloc[-1], y0=0, y1=1,
            fillcolor="rgba(128,128,128,0.12)", line_width=0, layer="below",
            label=dict(text="forecast", textposition="top left", font=dict(size=10, color="#666666")),
        )
    fig.update_layout(
        title=f"Series {series_id}", xaxis_title="time", yaxis_title="value",
        yaxis_range=[lo, hi], margin=dict(l=50, r=20, t=50, b=40), hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), template=template,
    )
    return fig


def atlas_scatter(windows: pd.DataFrame, template: str = "plotly_white") -> go.Figure:
    """2-D map of gated windows colored by cluster (cluster id cast to str for discrete colors)."""
    g = windows[windows["gated"].astype(bool) & windows["x2d"].notna() & windows["y2d"].notna()]
    fig = go.Figure()
    for cid in sorted(g["cluster"].unique()):
        sub = g[g["cluster"] == cid]
        fig.add_trace(go.Scatter(
            x=sub["x2d"], y=sub["y2d"], mode="markers", name=str(cid),
            marker=dict(color=_cluster_color(int(cid)), size=7, opacity=0.8),
            text=sub["window_id"], hovertemplate="%{text}<br>cluster " + str(cid) + "<extra></extra>",
        ))
    fig.update_layout(
        title="Failure atlas — gated windows", xaxis_title="", yaxis_title="",
        xaxis=dict(showticklabels=False), yaxis=dict(showticklabels=False),
        legend_title_text="cluster", margin=dict(l=20, r=20, t=50, b=20), template=template,
    )
    return fig


# ---------------------------------------------------------------- latent space (t-SNE) figures

TRUE_LABEL_COLORS = {
    "delayed_response": "#1f77b4", "missed_peak": "#ff7f0e", "unadapted_shift": "#2ca02c",
    "over_smoothing": "#9467bd", "healthy": "#8fbf8f",
}
NORMAL_COLOR = "#BDBDBD"
NORMAL_NAME = "normal (not anomalous)"
MAX_SERIES_LEGEND = 12


def cluster_name_map(clusters: list[ClusterInfo]) -> dict[int, str]:
    """cluster_id -> Gemma's short name, for clusters that have a report."""
    return {c.cluster_id: c.report.name for c in clusters if c.report is not None}


def _cluster_label(cid: int, names: dict[int, str] | None) -> str:
    base = "unclustered" if cid < 0 else f"cluster {cid}"
    name = (names or {}).get(cid)
    return f"{cid}: {name}" if name and cid >= 0 else base


def _hover(df: pd.DataFrame, names: dict[int, str] | None, err_col: str) -> list[str]:
    has_label = "true_label" in df.columns
    out = []
    for row in df.to_dict("records"):
        cid = int(row["cluster"])
        cname = (names or {}).get(cid)
        if cid == CLUSTER_NOISE:
            cl = "unclustered"
        elif cid < 0:
            cl = "normal (not gated)"
        else:
            cl = f"{cid}" + (f" ({cname})" if cname else "")
        lines = [f"<b>{row['window_id']}</b>", f"series: {row['series_id']}", f"cluster: {cl}",
                 f"abs_err: {float(row[err_col]):.2f}"]
        if has_label:
            lines.append(f"true label: {row['true_label']}")
        out.append("<br>".join(lines))
    return out


def latent_figure(
    windows: pd.DataFrame,
    coords: np.ndarray,
    color_by: str = "cluster",
    cluster_names: dict[int, str] | None = None,
    highlight_series: str | None = None,
    template: str = "plotly_white",
) -> go.Figure:
    """t-SNE scatter of windows (row-aligned with `coords` [n, 2]; NaN rows are skipped).

    Not-gated windows are small light-grey dots, gated ones are colored by `color_by` in
    {"cluster", "true_label", "series_id"}. `customdata` of every point is its window_id.
    """
    coords = np.asarray(coords, dtype=np.float64)
    if len(coords) != len(windows):
        raise ValueError(f"coords has {len(coords)} rows but windows has {len(windows)}")
    df = windows.reset_index(drop=True).assign(_x=coords[:, 0], _y=coords[:, 1])
    df = df[np.isfinite(df["_x"]) & np.isfinite(df["_y"])].reset_index(drop=True)
    err_col = "abs_err" if "abs_err" in df.columns else "score_abs"
    df = df.assign(_hover=_hover(df, cluster_names, err_col))
    if color_by == "true_label" and "true_label" not in df.columns:
        color_by = "cluster"
    dark = template.endswith("dark")
    gated = df["gated"].astype(bool)

    def scatter(sub: pd.DataFrame, name: str, color, size: int, opacity: float, **kw) -> go.Scatter:
        return go.Scatter(
            x=sub["_x"], y=sub["_y"], mode="markers", name=name, text=sub["_hover"], hoverinfo="text",
            customdata=sub["window_id"].to_numpy(), marker=dict(color=color, size=size, opacity=opacity),
            **kw,
        )

    fig = go.Figure()
    normal = df[~gated]
    if len(normal):
        fig.add_trace(scatter(normal, NORMAL_NAME, NORMAL_COLOR, 5, 0.45))

    anomalous = df[gated]
    if color_by == "series_id":
        ids = {s: i for i, s in enumerate(sorted(anomalous["series_id"].astype(str).unique()))}
        colors = [CLUSTER_COLORS[ids[str(s)] % len(CLUSTER_COLORS)] for s in anomalous["series_id"]]
        if len(anomalous):
            fig.add_trace(scatter(anomalous, "anomalous (colored by series)", colors, 8, 0.85))
    else:
        for key in sorted(anomalous[color_by].astype(int if color_by == "cluster" else str).unique()):
            sub = anomalous[anomalous[color_by].astype(int if color_by == "cluster" else str) == key]
            if color_by == "cluster":
                name, color = _cluster_label(int(key), cluster_names), _cluster_color(int(key))
            else:
                name = str(key)
                color = TRUE_LABEL_COLORS.get(name, NOISE_COLOR)
            fig.add_trace(scatter(sub, name, color, 8, 0.85))

    if highlight_series is not None:
        sel = df[df["series_id"].astype(str) == str(highlight_series)]
        if len(sel):
            colors = [_cluster_color(int(c)) if g else NORMAL_COLOR for c, g in zip(sel["cluster"], sel["gated"])]
            trace = scatter(sel, "selected series", colors, 14, 1.0)
            trace.marker.line = dict(color="white" if dark else "black", width=2)
            fig.add_trace(trace)

    fig.update_layout(
        title="Latent space (t-SNE) — each dot is one 28-step window", height=600, template=template,
        xaxis=dict(title="t-SNE 1", showticklabels=False, zeroline=False),
        yaxis=dict(title="t-SNE 2", showticklabels=False, zeroline=False),
        legend_title_text={"cluster": "cluster", "true_label": "true label"}.get(color_by, ""),
        margin=dict(l=20, r=20, t=50, b=40), hovermode="closest",
    )
    return fig


def window_figure(
    window_id: str, run: RunResult, context: int = 14, template: str = "plotly_white"
) -> go.Figure:
    """One window: actual vs predicted in raw units with `context` rows before, window region shaded."""
    row = run.windows[run.windows["window_id"] == window_id].iloc[0]
    sid = str(row["series_id"])
    s = run.data[run.data["series_id"] == sid].sort_values("ts").reset_index(drop=True)
    first_fc = int(np.flatnonzero(s["is_forecast"].to_numpy(dtype=bool))[0])
    pos0, pos1 = first_fc + int(row["start_idx"]), first_fc + int(row["end_idx"])
    seg = s.iloc[max(0, pos0 - context):pos1]
    t0, t1 = s["ts"].iloc[pos0], s["ts"].iloc[min(pos1, len(s)) - 1]

    cid = int(row["cluster"]) if "cluster" in row.index else CLUSTER_NOT_GATED
    name = cluster_name_map(run.clusters).get(cid)
    if cid == CLUSTER_NOISE:
        where = "unclustered"
    elif cid < 0:
        where = "normal (not gated)"
    else:
        where = f"cluster {cid}" + (f" ({name})" if name else "")

    fig = go.Figure()
    fig.add_shape(type="rect", xref="x", yref="paper", x0=t0, x1=t1, y0=0, y1=1,
                  fillcolor=_rgba(WINDOW_COLOR, 0.35), line_width=0, layer="below",
                  label=dict(text="window", textposition="top left", font=dict(size=10, color="#888888")))
    fig.add_trace(go.Scatter(x=seg["ts"], y=seg["actual"], mode="lines", name="actual",
                             line=dict(color="white" if template.endswith("dark") else ACTUAL_COLOR, width=1.8)))
    fig.add_trace(go.Scatter(x=seg["ts"], y=seg["predicted"], mode="lines", name="predicted",
                             line=dict(color=PRED_COLOR, width=1.8)))
    fig.update_layout(
        title=f"{window_id} — {where}", xaxis_title="time", yaxis_title="value", template=template,
        margin=dict(l=50, r=20, t=50, b=40), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig
