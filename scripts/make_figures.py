"""Explanatory figures from a real run, for the pitch deck and the README.

Owner: C. Task: 18.

    python scripts/make_figures.py --run runs/synth_features_v1 [--out docs/figures] [--only 1,3] [--tsne-cache tsne.npy]

Writes five 200 dpi PNGs (16:9, except the gallery) into --out:

    01_transformation.png        one real series walked through the pipeline, forecast -> diagnosis
    02_latent_space.png          t-SNE of all windows: injected truth next to the clusters found
    03_pipeline_vs_gemma_alone.png   accuracy / macro-F1 / per-class accuracy / Gemma calls (needs eval.json)
    04_llm_plus_expert.png       diagram: what the LLM gets from code and from the expert
    05_cluster_gallery.png       the cluster panels (what Gemma sees) of the four largest clusters

Everything is read from the run folder; residuals and the t-SNE map are recomputed from it. A figure whose
inputs are missing (no true_label, no eval comparison, no Gemma report) is skipped with a message.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.colors  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # `python scripts/make_figures.py` puts scripts/ first, not the repo root
    sys.path.insert(0, str(REPO_ROOT))

from atlas.cluster import embed_features, project_tsne  # noqa: E402
from atlas.features import FEATURE_DOCS, FEATURES  # noqa: E402
from atlas.plots import CLUSTER_COLORS, NOISE_COLOR  # noqa: E402
from atlas.residuals import compute_scaled_residuals  # noqa: E402
from atlas.schema import CLUSTER_NOISE, ClusterInfo, Dataset, RunResult, load_run  # noqa: E402

DPI = 200
FIG_W, FIG_H = 16.0, 9.0

# ---------------------------------------------------------------- style

INK = "#1B2430"        # primary text
INK2 = "#4A5568"       # secondary text, arrows
MUTED = "#718096"      # footers, units
GRID = "#E2E8F0"
PANEL_BG = "#F3F6FA"
NOT_ANOMALOUS = "#D3D9E1"
ACTUAL_COLOR = "#1B2430"
FORECAST_COLOR = "#D62728"
RESID_COLOR = "#1F6FB2"
FORECAST_SHADE = "#EAF0F8"

# the three kinds of knowledge in figure 4 (soft background / edge)
CODE_BG, CODE_EDGE = "#E3EEFB", "#2A78D6"
GEMMA_BG, GEMMA_EDGE = "#FFEFD2", "#D98200"
EXPERT_BG, EXPERT_EDGE = "#DDF3E7", "#1B9A63"

TSD_COLOR = "#1F5FBF"   # TimeSeriousDoctor in the comparison
BASE_COLOR = "#9AA5B4"  # Gemma alone

# failure types (truth and Gemma's names); deliberately not the cluster palette, so a reader does not
# look for a color match between "injected type" and "cluster id"
FAILURE_COLORS = {
    "delayed_response": "#0F8B8D",
    "missed_peak": "#C2185B",
    "unadapted_shift": "#4A55C8",
    "over_smoothing": "#E0A100",
    "healthy": "#9AD29A",
    "healthy_noise": "#9AD29A",
    "other": "#8C8C8C",
}
FAILURE_NAMES = {
    "delayed_response": "delayed response",
    "missed_peak": "missed peak",
    "unadapted_shift": "unadapted shift",
    "over_smoothing": "over-smoothing",
    "healthy": "healthy",
    "healthy_noise": "healthy noise",
    "other": "other",
}

PLAIN_FEATURES = {
    "abs_err": "Error size",
    "bias": "Bias (+ = too low)",
    "best_lag": "Forecast delay",
    "lag_gain": "Gain from shifting",
    "slope_corr": "Error tracks moves",
    "peak_excess": "Error at peaks",
    "var_ratio": "Forecast spread",
    "sign_run": "One-sided streak",
    "r_trend": "Error trend",
    "dir_acc": "Direction hit rate",
    "r_acf_season": "Weekly error cycle",
    "spike_ratio": "Error spikiness",
}


def set_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#B8C2CF",
        "axes.labelcolor": INK2,
        "axes.labelsize": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "text.color": INK,
        "axes.grid": False,
    })


def cluster_color(cid: int) -> str:
    return NOISE_COLOR if cid < 0 else CLUSTER_COLORS[cid % len(CLUSTER_COLORS)]


def cluster_name(c: ClusterInfo) -> str:
    """Gemma's short name, first letter capitalised (Gemma is inconsistent about it)."""
    if c.cluster_id == CLUSTER_NOISE:
        return "Unclustered (look manually)"
    if c.report is None:
        return f"Cluster {c.cluster_id} (no Gemma report)"
    name = c.report.name.strip()
    return name[0].upper() + name[1:]


def on_color(hex_color: str) -> str:
    """Text color (white or ink) with the better contrast on a filled background."""
    r, g, b = matplotlib.colors.to_rgb(hex_color)
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in (r, g, b)]
    lum = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    ink_lum = 0.02  # INK
    return "white" if 1.05 / (lum + 0.05) >= (lum + 0.05) / (ink_lum + 0.05) else INK


def pct(v: float) -> str:
    return f"{int(v * 100 + 0.5)}%"


# ---------------------------------------------------------------- shared drawing helpers

def new_figure(width: float = FIG_W, height: float = FIG_H) -> Figure:
    return plt.figure(figsize=(width, height), dpi=DPI, facecolor="white")


def header(fig: Figure, title: str, subtitle: str | None = None) -> None:
    fig.text(0.04, 0.965, title, fontsize=27, fontweight="bold", color=INK, ha="left", va="top")
    if subtitle:
        fig.text(0.04, 0.912, subtitle, fontsize=14, color=INK2, ha="left", va="top")


def footer(fig: Figure, run: RunResult) -> None:
    fig.text(0.04, 0.012, f"TimeSeriousDoctor  |  run {run.name}", fontsize=9, color=MUTED, ha="left", va="bottom")


def step_title(fig: Figure, x: float, y: float, n: int, text: str, sub: str | None = None) -> None:
    """Numbered badge + panel title (figure coordinates)."""
    fig.text(x, y, str(n), color="white", fontsize=12, fontweight="bold", ha="center", va="center",
             bbox=dict(boxstyle="circle,pad=0.32", fc=INK, ec="none"))
    fig.text(x + 0.016, y, text, fontsize=14, fontweight="bold", color=INK, ha="left", va="center")
    if sub:
        fig.text(x + 0.016, y - 0.026, sub, fontsize=10.5, color=MUTED, ha="left", va="center")


def fig_arrow(fig: Figure, pts: list[tuple[float, float]], color: str = INK2, lw: float = 2.4,
              style: str = "-", head: bool = True) -> None:
    """Polyline in figure coordinates, arrow head on the last segment."""
    body = pts[:-1] if head else pts
    if len(body) > 1:
        xs, ys = zip(*body)
        fig.add_artist(Line2D(xs, ys, transform=fig.transFigure, color=color, lw=lw, ls=style, solid_capstyle="butt"))
    if head:
        fig.add_artist(FancyArrowPatch(pts[-2], pts[-1], transform=fig.transFigure, arrowstyle="-|>",
                                       mutation_scale=20, lw=lw, color=color, ls=style, shrinkA=0, shrinkB=0))


def wrap_text(fig: Figure, text: str, size: float, width_in: float, weight: str = "normal") -> list[str]:
    """Greedy word wrap measured with the real font, so lines fit `width_in` inches."""
    renderer = fig.canvas.get_renderer()
    prop = FontProperties(family="DejaVu Sans", size=size, weight=weight)
    lines: list[str] = []
    cur = ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        w_px = renderer.get_text_width_height_descent(trial, prop, False)[0]
        if cur and w_px / fig.dpi > width_in:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


def axes_inches(fig: Figure, ax) -> tuple[float, float]:
    pos = ax.get_position()
    return pos.width * fig.get_figwidth(), pos.height * fig.get_figheight()


def text_width_in(fig: Figure, text: str, size: float, weight: str = "normal") -> float:
    prop = FontProperties(family="DejaVu Sans", size=size, weight=weight)
    return fig.canvas.get_renderer().get_text_width_height_descent(text, prop, False)[0] / fig.dpi


def round_box(fig: Figure, ax, x: float, y: float, w: float, h: float, radius_in: float, **kw) -> FancyBboxPatch:
    """Rounded box in the data coordinates of `ax` with true circular corners of `radius_in` inches.

    The axes limits must be set before calling this.
    """
    w_in, h_in = axes_inches(fig, ax)
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    rx = radius_in * (x1 - x0) / w_in
    aspect = ((y1 - y0) / h_in) / ((x1 - x0) / w_in)
    patch = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rx}", mutation_aspect=aspect, **kw)
    ax.add_patch(patch)
    return patch


# ---------------------------------------------------------------- run context

@dataclass
class Context:
    run: RunResult
    res: pd.DataFrame
    windows: pd.DataFrame
    feats: list[str]
    tsne_cache: Path | None = None
    _coords: np.ndarray | None = field(default=None, repr=False)

    @property
    def gated(self) -> np.ndarray:
        return self.windows["gated"].to_numpy(dtype=bool)

    @property
    def has_truth(self) -> bool:
        return "true_label" in self.windows.columns

    def coords(self) -> np.ndarray:
        """t-SNE of all windows [n, 2] (NaN = not projected). Cached on disk if --tsne-cache is given."""
        if self._coords is None:
            if self.tsne_cache is not None and self.tsne_cache.exists():
                coords = np.load(self.tsne_cache)
                if len(coords) == len(self.windows):
                    self._coords = coords
                    return coords
            X = embed_features(self.windows, self.feats)
            self._coords = project_tsne(X, keep=self.gated)
            if self.tsne_cache is not None:
                np.save(self.tsne_cache, self._coords)
        return self._coords


def load_context(run_dir: Path, tsne_cache: Path | None) -> Context:
    run = load_run(run_dir)
    res = compute_scaled_residuals(Dataset(run.data, run.series_meta), season=int(run.config.get("season", 7)))
    windows = run.windows.reset_index(drop=True)
    feats = [f for f in FEATURES if f in windows.columns]
    return Context(run=run, res=res, windows=windows, feats=feats, tsne_cache=tsne_cache)


def save(fig: Figure, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=DPI, facecolor="white")
    plt.close(fig)
    print(f"wrote {path}")
    return path


def scatter_map(ax) -> None:
    """Common frame for the t-SNE panels."""
    ax.set_xticks([])
    ax.set_yticks([])
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#CBD3DE")
    # no set_aspect: t-SNE axes carry no scale, so the map fills its panel
    ax.margins(0.04)


# ---------------------------------------------------------------- figure 1: from a forecast to a diagnosis

HISTORY_SHOWN = 70
TOP_FEATURES = 7


def pick_walkthrough(ctx: Context) -> tuple[ClusterInfo, str] | None:
    """A large, clean delayed-response / unadapted-shift cluster with a Gemma report, and its medoid window."""
    reported = [c for c in ctx.run.clusters if c.cluster_id >= 0 and c.report is not None and c.medoid_window_ids]
    wanted = [c for c in reported if c.report.failure_type in ("delayed_response", "unadapted_shift")]

    def purity(c: ClusterInfo) -> float:
        if not ctx.has_truth:
            return 1.0
        lab = ctx.windows.loc[ctx.windows["cluster"] == c.cluster_id, "true_label"]
        return float(lab.value_counts(normalize=True).iloc[0]) if len(lab) else 0.0

    for pool in (wanted, reported):
        pool = sorted(pool, key=lambda c: -c.size)
        clean = [c for c in pool if purity(c) >= 0.9]
        if clean or pool:
            c = (clean or pool)[0]
            return c, c.medoid_window_ids[0]
    return None


def robust_z(ctx: Context, values: dict[str, float] | pd.Series) -> pd.Series:
    """Distance from the median anomalous (gated) window, in robust units (IQR), clipped to +-5."""
    g = ctx.windows.loc[ctx.gated, ctx.feats]
    med = g.median()
    iqr = (g.quantile(0.75) - g.quantile(0.25)).replace(0.0, 1.0)
    v = pd.Series({f: float(values[f]) for f in ctx.feats})
    return ((v - med) / iqr).clip(-5, 5)


def fig_transformation(ctx: Context, out_dir: Path) -> Path | None:
    run = ctx.run
    picked = pick_walkthrough(ctx)
    if picked is None:
        print("skip 01_transformation: no cluster with a Gemma report")
        return None
    cluster, win_id = picked
    report = cluster.report
    color = cluster_color(cluster.cluster_id)
    win = ctx.windows[ctx.windows["window_id"] == win_id].iloc[0]
    win_idx = int(ctx.windows.index[ctx.windows["window_id"] == win_id][0])
    sid = str(win["series_id"])
    card = run.series_meta.set_index("series_id")["model_card"].get(sid, "")
    s = ctx.res[ctx.res["series_id"] == sid].sort_values("step")
    fc = s[s["is_forecast"]]
    seg = pd.concat([s[~s["is_forecast"]].tail(HISTORY_SHOWN), fc])
    sw = ctx.windows[ctx.windows["series_id"] == sid].sort_values("start_idx")

    fig = new_figure()
    header(fig, "From a forecast to a diagnosis",
           f"One real series ({sid}, forecast by {card}) followed through the pipeline. "
           f"It sits in cluster {cluster.cluster_id}, the most typical window of that cluster is highlighted.")
    footer(fig, run)

    t0, t1 = fc["ts"].iloc[0], fc["ts"].iloc[-1]
    day = pd.Timedelta(days=1)
    w0, w1 = win["start_ts"], win["end_ts"]

    # (a) actual vs forecast
    ax = fig.add_axes([0.055, 0.58, 0.425, 0.242])
    ax.axvspan(t0, t1 + day, color=FORECAST_SHADE, lw=0, zorder=0)
    ax.axvspan(w0, w1 + day, color=color, alpha=0.30, lw=0, zorder=1)
    ax.plot(seg["ts"], seg["actual"], color=ACTUAL_COLOR, lw=1.9, label="actual", zorder=3)
    ax.plot(fc["ts"], fc["predicted"], color=FORECAST_COLOR, lw=1.9, label="forecast", zorder=3)
    ax.set_xlim(seg["ts"].iloc[0], t1 + day)
    ylo, yhi = ax.get_ylim()
    ax.set_ylim(ylo, yhi + 0.12 * (yhi - ylo))
    ytxt = ax.get_ylim()[1] - 0.04 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.text(seg["ts"].iloc[0] + day * 2, ytxt, "history", fontsize=11, color=MUTED, va="top")
    ax.text(t0 + day * 2, ytxt, "forecast period (judged)", fontsize=11, color=MUTED, va="top")
    ax.text(w0 + (w1 - w0) / 2 + day / 2, ytxt, "selected\nwindow", fontsize=10.5, color=INK, va="top", ha="center",
            fontweight="bold", linespacing=1.0)
    ax.set_ylabel("value")
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.legend(loc="lower left", ncol=2, frameon=False, fontsize=11, handlelength=1.6)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    step_title(fig, 0.055, 0.866, 1, "The forecast", "actual against what the model predicted")

    # (b) scaled residuals, windows, gate
    axr = fig.add_axes([0.585, 0.70, 0.40, 0.122])
    axw = fig.add_axes([0.585, 0.632, 0.40, 0.062], sharex=axr)
    axr.axvspan(w0, w1 + day, color=color, alpha=0.30, lw=0)
    axr.axhline(0, color="#6B7686", lw=1.0)
    axr.plot(fc["ts"], fc["r_s"], color=RESID_COLOR, lw=1.6)
    axr.set_xlim(t0 - day, t1 + day)
    axr.set_ylabel("error r", labelpad=2)
    axr.tick_params(labelbottom=False)
    axr.grid(axis="y", color=GRID, lw=0.8)
    axr.set_axisbelow(True)

    axw.set_ylim(0, 2)
    axw.set_yticks([])
    for side in ("left", "top", "right"):
        axw.spines[side].set_visible(False)
    for _, w in sw.iterrows():
        row = int(w["start_idx"] // 14) % 2          # stride 14: neighbours overlap, so alternate two rows
        is_sel = w["window_id"] == win_id
        gated = bool(w["gated"])
        wc = cluster_color(int(w["cluster"])) if gated else "#C6CED9"
        x0 = mdates.date2num(w["start_ts"]) + 0.4
        wd = mdates.date2num(w["end_ts"]) + 1 - mdates.date2num(w["start_ts"]) - 0.8
        yb = 1.12 if row == 0 else 0.1
        round_box(fig, axw, x0, yb, wd, 0.8, 0.05, fc=wc if gated else "#F1F4F8",
                  ec=INK if is_sel else wc, lw=3.0 if is_sel else 1.8, zorder=3)
        if gated:   # tint the fill without washing out the edge
            axw.patches[-1].set_facecolor(matplotlib.colors.to_rgba(wc, 0.30))
        axw.text(x0 + wd / 2, yb + 0.4, f"{float(w['score_abs']):.1f}", ha="center", va="center", fontsize=11,
                 fontweight="bold" if gated else "normal", color=INK if gated else MUTED, zorder=4)
    axw.xaxis.set_major_formatter(mdates.ConciseDateFormatter(axw.xaxis.get_major_locator()))
    axw.tick_params(axis="x", labelsize=11)

    gate = run.config.get("gate", {})
    if gate.get("mode", "threshold") == "threshold":
        rule = (f"Gate keeps a window if mean |r| > {gate.get('t_abs', 1.0):g}, or |mean r| > {gate.get('t_bias', 0.75):g}, "
                f"or max |r| > {gate.get('t_peak', 3.0):g}")
    else:
        rule = f"Gate keeps the top {100 * gate.get('q', 0.25):.0f}% of windows by mean |r|"
    fig.text(0.585, 0.578, "Brackets: 28-step windows, number = mean |r|. Colored = kept, grey = dropped.",
             fontsize=10.5, color=INK2, ha="left", va="center")
    fig.text(0.585, 0.556, rule, fontsize=10.5, color=INK, fontweight="bold", ha="left", va="center")
    step_title(fig, 0.585, 0.866, 2, "Errors cut into windows, abnormal ones kept",
               "r = actual - forecast, in units of the series' typical seasonal change")

    # (c) feature fingerprint
    z_win = robust_z(ctx, win[ctx.feats])
    z_med = robust_z(ctx, cluster.feature_median)
    top = list(z_win.abs().sort_values(ascending=False).index[:TOP_FEATURES])
    axc = fig.add_axes([0.185, 0.085, 0.115, 0.29])
    ypos = np.arange(len(top))[::-1]
    axc.barh(ypos, z_win[top].to_numpy(), height=0.58, color=color, alpha=0.9, zorder=3)
    axc.scatter(z_med[top].to_numpy(), ypos, marker="D", s=46, color=INK, zorder=4)
    axc.axvline(0, color="#6B7686", lw=1.2, zorder=2)
    axc.set_xlim(-5.4, 5.4)
    axc.set_ylim(-0.6, len(top) - 0.4)
    axc.set_yticks([])
    axc.set_xticks([-4, 0, 4])
    axc.set_xticklabels(["lower", "typical", "higher"], fontsize=10.5)
    axc.grid(axis="x", color=GRID, lw=0.8)
    axc.set_axisbelow(True)
    axc.spines["left"].set_visible(False)
    ytr = axc.get_yaxis_transform()
    for y, f in zip(ypos, top):
        plain = PLAIN_FEATURES.get(f) or FEATURE_DOCS.get(f, f).split(".")[0][:24]
        axc.text(-0.07, y + 0.07, plain, transform=ytr, ha="right", va="center", fontsize=12, color=INK)
        axc.text(-0.07, y - 0.30, f, transform=ytr, ha="right", va="center", fontsize=9.5, color=MUTED)
        axc.text(1.06, y, f"{float(win[f]):.2f}", transform=ytr, ha="left", va="center", fontsize=12, color=INK,
                 fontweight="bold")
        axc.text(1.5, y, f"{ctx.windows.loc[ctx.gated, f].median():.2f}", transform=ytr, ha="left", va="center",
                 fontsize=12, color=MUTED)
    axc.text(1.06, len(top) - 0.35, "this\nwindow", transform=ytr, ha="left", va="bottom", fontsize=10.5,
             color=INK, fontweight="bold", linespacing=1.0)
    axc.text(1.5, len(top) - 0.35, "all anomalous\n(median)", transform=ytr, ha="left", va="bottom", fontsize=10.5,
             color=MUTED, linespacing=1.0)
    step_title(fig, 0.055, 0.468, 3, "Its failure fingerprint", "bar = this window, diamond = its cluster's median")

    # (d) t-SNE map with the window marked
    coords = ctx.coords()
    axd = fig.add_axes([0.445, 0.07, 0.225, 0.335])
    ok = np.isfinite(coords[:, 0])
    gated = ctx.gated
    clus = ctx.windows["cluster"].to_numpy()
    base = ok & ~gated
    axd.scatter(coords[base, 0], coords[base, 1], s=5, color=NOT_ANOMALOUS, alpha=0.7, lw=0, rasterized=True)
    other = ok & gated & (clus != cluster.cluster_id)
    axd.scatter(coords[other, 0], coords[other, 1], s=9, color=[cluster_color(int(c)) for c in clus[other]],
                alpha=0.28, lw=0, rasterized=True)
    mine = ok & gated & (clus == cluster.cluster_id)
    axd.scatter(coords[mine, 0], coords[mine, 1], s=16, color=color, alpha=0.9, lw=0, rasterized=True)
    axd.scatter([coords[win_idx, 0]], [coords[win_idx, 1]], s=520, marker="*", color="white", edgecolor=INK,
                linewidth=2.2, zorder=6)
    axd.scatter([coords[win_idx, 0]], [coords[win_idx, 1]], s=190, marker="*", color=color, zorder=7)
    scatter_map(axd)
    cx, cy = np.nanmedian(coords[mine, 0]), np.nanmedian(coords[mine, 1])
    axd.annotate(f"cluster {cluster.cluster_id}", (cx, cy), xytext=(-58, -34), textcoords="offset points",
                 ha="center", fontsize=12, fontweight="bold", color=INK,
                 arrowprops=dict(arrowstyle="-", color=INK2, lw=1.2, shrinkA=2, shrinkB=14),
                 bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=color, lw=2))
    step_title(fig, 0.445, 0.468, 4, "Where it lands", "every dot is a window; grey = normal")

    # (e) the cluster's Gemma report
    axe = fig.add_axes([0.695, 0.07, 0.29, 0.335])
    axe.axis("off")
    axe.set_xlim(0, 1)
    axe.set_ylim(0, 1)
    w_in, h_in = axes_inches(fig, axe)
    round_box(fig, axe, 0, 0, 1, 1, 0.12, fc="white", ec=color, lw=3.0)
    round_box(fig, axe, 0, 0.865, 1, 0.135, 0.12, fc=color, ec=color, lw=3.0)
    axe.add_patch(Rectangle((0.0, 0.865), 1, 0.05, fc=color, ec="none"))
    axe.text(0.035, 0.932, f"Cluster {cluster.cluster_id}  |  {cluster.size} windows from {cluster.n_series} series",
             fontsize=12, fontweight="bold", color=on_color(color), va="center")
    lh = 1.0 / h_in / 72.0  # one point of height in axes units

    def put(x: float, y: float, text: str, size: float, **kw) -> None:
        axe.text(x, y, text, fontsize=size, va="top", ha="left", **kw)

    body = 11.5
    y = 0.835
    for ln in wrap_text(fig, cluster_name(cluster), 17, w_in - 0.4, "bold"):
        put(0.035, y, ln, 17, fontweight="bold", color=INK)
        y -= 17 * 1.25 * lh
    y -= 5 * lh
    ft = report.failure_type
    chip = FAILURE_NAMES.get(ft, ft)
    axe.text(0.035, y, chip, fontsize=11.5, fontweight="bold", color="white", va="top", ha="left",
             bbox=dict(boxstyle="round,pad=0.28", fc=FAILURE_COLORS.get(ft, "#8C8C8C"), ec="none"))
    chip_w = text_width_in(fig, chip, 11.5, "bold") + 0.35
    axe.text(0.035 + chip_w / w_in + 0.03, y, f"confidence {report.confidence:.0%}", fontsize=11, color=MUTED,
             va="top", ha="left")
    y -= 11.5 * 2.35 * lh
    put(0.035, y, "Evidence", 12, fontweight="bold", color=INK2)
    y -= 12 * 1.5 * lh
    for item in report.evidence[:2]:
        lines = wrap_text(fig, item.rstrip("."), body, w_in - 0.66)
        axe.text(0.04, y, "•", fontsize=body, va="top", ha="left", color=color, fontweight="bold")
        for ln in lines:
            put(0.085, y, ln, body, color=INK)
            y -= body * 1.3 * lh
        y -= 3 * lh
    y -= 2 * lh
    put(0.035, y, "Top suggested fix", 12, fontweight="bold", color=INK2)
    y -= 12 * 1.5 * lh
    for ln in wrap_text(fig, report.suggested_fixes[0].rstrip("."), body, w_in - 0.5):
        put(0.035, y, ln, body, color=INK)
        y -= body * 1.3 * lh
    step_title(fig, 0.695, 0.468, 5, "Gemma explains the cluster", "from its chart, numbers and the expert taxonomy")

    # stage-to-stage arrows (figure coordinates)
    fig_arrow(fig, [(0.488, 0.70), (0.532, 0.70)])
    fig_arrow(fig, [(0.84, 0.538), (0.84, 0.518), (0.115, 0.518), (0.115, 0.497)])
    fig_arrow(fig, [(0.406, 0.24), (0.436, 0.24)])
    fig_arrow(fig, [(0.675, 0.24), (0.690, 0.24)])
    return save(fig, out_dir, "01_transformation.png")


# ---------------------------------------------------------------- figure 2: latent space

def legend_entries(handles: list[tuple[str, str, int | None]]) -> list[Line2D]:
    out = []
    for label, col, n in handles:
        text = label if n is None else f"{label}  ({n:,})"
        out.append(Line2D([], [], marker="o", ls="", markersize=10, markerfacecolor=col, markeredgecolor="none",
                          label=text))
    return out


def fig_latent_space(ctx: Context, out_dir: Path) -> Path | None:
    run = ctx.run
    coords = ctx.coords()
    ok = np.isfinite(coords[:, 0])
    gated = ctx.gated
    clus = ctx.windows["cluster"].to_numpy()
    n_plot = int(ok.sum())
    purity = (run.eval or {}).get("clustering", {}).get("purity")

    fig = new_figure()
    sub = f"Each dot is one 28-step window of forecast error, placed by its failure features. {n_plot:,} windows shown."
    if ctx.has_truth and purity is not None:
        sub = (f"Each dot is one 28-step window of forecast error. Clusters largely recover the injected failure types "
               f"(cluster purity {purity:.0%}).")
    header(fig, "Windows with the same failure sit together", sub)
    footer(fig, run)

    panels = ["truth", "cluster"] if ctx.has_truth else ["cluster"]
    n = len(panels)
    xs = [(0.04, 0.485), (0.525, 0.97)] if n == 2 else [(0.2, 0.8)]
    for (x0, x1), kind in zip(xs, panels):
        ax = fig.add_axes([x0, 0.385, x1 - x0, 0.42])
        base = ok & ~gated
        ax.scatter(coords[base, 0], coords[base, 1], s=11, color=NOT_ANOMALOUS, alpha=0.75, lw=0, rasterized=True)
        entries: list[tuple[str, str, int | None]] = []
        if kind == "truth":
            truth = ctx.windows["true_label"].to_numpy()
            order = ["delayed_response", "missed_peak", "unadapted_shift", "over_smoothing", "healthy"]
            for lab in order + sorted(set(truth[gated]) - set(order)):
                m = ok & gated & (truth == lab)
                if not m.any():
                    continue
                col = FAILURE_COLORS.get(lab, "#8C8C8C")
                ax.scatter(coords[m, 0], coords[m, 1], s=28, color=col, alpha=0.9, lw=0, rasterized=True)
                name = FAILURE_NAMES.get(lab, lab)
                entries.append((name + (" (flagged by the gate)" if lab == "healthy" else ""), col, int(m.sum())))
            title = "What was injected"
            subt = "colored by the true failure type"
        else:
            infos = sorted(run.clusters, key=lambda c: (c.cluster_id == CLUSTER_NOISE, c.cluster_id))
            for c in infos:
                m = ok & gated & (clus == c.cluster_id)
                if not m.any():
                    continue
                col = cluster_color(c.cluster_id)
                ax.scatter(coords[m, 0], coords[m, 1], s=28, color=col, alpha=0.9, lw=0, rasterized=True)
                lab = cluster_name(c) if c.cluster_id == CLUSTER_NOISE else f"{c.cluster_id}  {cluster_name(c)}"
                entries.append((lab, col, int(m.sum())))
            title = "What the pipeline found"
            subt = "colored by cluster, named by Gemma"
        if (ok & ~gated).any():
            entries.append(("normal window (not flagged)", NOT_ANOMALOUS, int((ok & ~gated).sum())))
        scatter_map(ax)
        ax.set_xlabel("t-SNE dimension 1", fontsize=10.5, color=MUTED)
        ax.set_ylabel("t-SNE dimension 2", fontsize=10.5, color=MUTED)
        fig.text(x0, 0.856, title, fontsize=17, fontweight="bold", color=INK, va="center", ha="left")
        fig.text(x0 + (text_width_in(fig, title, 17, "bold") + 0.25) / FIG_W, 0.856, subt, fontsize=12, color=MUTED,
                 va="center", ha="left")
        leg = fig.legend(handles=legend_entries(entries), loc="upper left", bbox_to_anchor=(x0 - 0.005, 0.345),
                         ncol=1, frameon=False, fontsize=11.5, labelspacing=0.32, columnspacing=1.2,
                         handletextpad=0.4, borderaxespad=0)
        for t in leg.get_texts():
            t.set_color(INK)
    return save(fig, out_dir, "02_latent_space.png")


# ---------------------------------------------------------------- figure 3: pipeline vs Gemma alone

def bar_labels(ax, bars, fmt, fontsize: float = 12, dy: float = 0.0, color: str = INK) -> None:
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + dy, fmt(h), ha="center", va="bottom", fontsize=fontsize,
                fontweight="bold", color=color)


def fig_pipeline_vs_gemma(ctx: Context, out_dir: Path) -> Path | None:
    run = ctx.run
    comp = (run.eval or {}).get("comparison")
    if not comp:
        print("skip 03_pipeline_vs_gemma_alone: no eval comparison in the run (run scripts/run_eval.py first)")
        return None
    pipe, base = comp["pipeline"], comp["gemma_alone"]
    n_series = int(comp.get("n_series", 0))
    model = run.config.get("gemma", {}).get("model", "Gemma")
    n_full = int(run.series_meta["series_id"].nunique())
    calls_full = comp.get("gemma_calls_full_dataset", {})

    fig = new_figure()
    header(fig, "Measuring first beats asking Gemma alone",
           f"Same model ({model}), same failure taxonomy, same {n_series} series. "
           "The pipeline adds measured evidence per cluster.")
    footer(fig, run)

    # panel 1: overall scores
    ax = fig.add_axes([0.055, 0.14, 0.235, 0.62])
    width = 0.36
    groups = np.arange(2)
    vals_p = [pipe["accuracy"], pipe["macro_f1"]]
    vals_b = [base["accuracy"], base["macro_f1"]]
    b1 = ax.bar(groups - width / 2 - 0.01, vals_p, width, color=TSD_COLOR, zorder=3)
    b2 = ax.bar(groups + width / 2 + 0.01, vals_b, width, color=BASE_COLOR, zorder=3)
    bar_labels(ax, b1, pct, 14, 0.012)
    bar_labels(ax, b2, pct, 14, 0.012)
    ax.set_xticks(groups)
    ax.set_xticklabels(["Accuracy", "Macro-F1"], fontsize=13)
    style_pct_axis(ax)
    fig.text(0.055, 0.80, "Overall score", fontsize=16, fontweight="bold", color=INK, va="center")
    fig.text(0.055, 0.772, "per series, higher is better", fontsize=11.5, color=MUTED, va="center")

    # panel 2: per class
    classes = [c for c in ["delayed_response", "missed_peak", "unadapted_shift", "over_smoothing", "healthy"]
               if c in pipe.get("per_class", {}) or c in base.get("per_class", {})]
    classes += [c for c in sorted(set(pipe.get("per_class", {})) | set(base.get("per_class", {}))) if c not in classes]
    ax2 = fig.add_axes([0.355, 0.14, 0.395, 0.62])
    xs = np.arange(len(classes))
    vp = [pipe["per_class"].get(c, 0.0) for c in classes]
    vb = [base["per_class"].get(c, 0.0) for c in classes]
    w2 = 0.44
    b3 = ax2.bar(xs - w2 / 2 - 0.005, vp, w2, color=TSD_COLOR, zorder=3)
    b4 = ax2.bar(xs + w2 / 2 + 0.005, vb, w2, color=BASE_COLOR, zorder=3)
    bar_labels(ax2, b3, pct, 10.5, 0.012)
    bar_labels(ax2, b4, pct, 10.5, 0.012)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([FAILURE_NAMES.get(c, c).replace("-", "-\n").replace(" ", "\n") for c in classes], fontsize=12)
    for tick, c in zip(ax2.get_xticklabels(), classes):
        tick.set_color(INK)
    style_pct_axis(ax2)
    fig.text(0.355, 0.80, "Accuracy per failure type", fontsize=16, fontweight="bold", color=INK, va="center")
    fig.text(0.355, 0.772, "share of series of that type diagnosed correctly", fontsize=11.5, color=MUTED, va="center")

    # panel 3: Gemma calls, log scale
    ax3 = fig.add_axes([0.82, 0.14, 0.15, 0.62])
    cp = calls_full.get("pipeline", pipe.get("gemma_calls"))
    cb = calls_full.get("gemma_alone", base.get("gemma_calls"))
    b5 = ax3.bar([0], [cp], 0.7, color=TSD_COLOR, zorder=3)
    b6 = ax3.bar([1], [cb], 0.7, color=BASE_COLOR, zorder=3)
    ax3.set_yscale("log")
    ax3.set_ylim(1, max(cb, cp) * 3.2)
    ax3.set_xlim(-0.6, 1.6)
    ax3.set_xticks([0, 1])
    ax3.set_xticklabels(["Pipeline", "Gemma\nalone"], fontsize=12.5)
    ax3.grid(axis="y", color=GRID, lw=0.8, which="major")
    ax3.set_axisbelow(True)
    ax3.spines["left"].set_color("#B8C2CF")
    ax3.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax3.tick_params(axis="y", labelsize=11)
    for b, v in ((b5[0], cp), (b6[0], cb)):
        ax3.text(b.get_x() + b.get_width() / 2, v * 1.12, f"{int(v):,}", ha="center", va="bottom", fontsize=14,
                 fontweight="bold", color=INK)
    if cp and cb:
        ax3.text(0.0, cp * 2.4, f"{cb / cp:.0f}x\nfewer", ha="center", va="bottom", fontsize=13,
                 fontweight="bold", color=TSD_COLOR, linespacing=1.0)
    fig.text(0.82, 0.80, "Gemma calls", fontsize=16, fontweight="bold", color=INK, va="center")
    fig.text(0.82, 0.772, f"for all {n_full} series, log scale", fontsize=11.5, color=MUTED, va="center")

    # legend (shared)
    handles = [Line2D([], [], marker="s", ls="", markersize=13, markerfacecolor=TSD_COLOR, markeredgecolor="none",
                      label="TimeSeriousDoctor (pipeline + Gemma)"),
               Line2D([], [], marker="s", ls="", markersize=13, markerfacecolor=BASE_COLOR, markeredgecolor="none",
                      label="Gemma alone (one chart per series)")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.04, 0.855), ncol=2, frameon=False, fontsize=13,
               columnspacing=2.2, handletextpad=0.3, borderaxespad=0)
    sample = (f"On the {n_series}-series sample the pipeline used {pipe.get('gemma_calls')} Gemma calls and Gemma alone "
              f"{base.get('gemma_calls')}. Pipeline calls grow with the number of clusters, Gemma-alone calls with the number of series.")
    fig.text(0.04, 0.062, sample, fontsize=11, color=INK2, ha="left", va="center")
    return save(fig, out_dir, "03_pipeline_vs_gemma_alone.png")


def style_pct_axis(ax) -> None:
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=11)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_color("#B8C2CF")
    ax.tick_params(axis="x", length=0)


# ---------------------------------------------------------------- figure 4: LLM + expert diagram (no data)

U = 10.0  # data units per inch on the diagram canvas (160 x 90 units on 16 x 9 in)


def diagram_box(fig: Figure, ax, x: float, y: float, w: float, h: float, title: str, body: str = "",
                fc: str = "white", ec: str = INK2, lw: float = 2.0, ls: str = "-", title_size: float = 13,
                body_size: float = 10.5, title_color: str = INK) -> None:
    round_box(fig, ax, x, y, w, h, 0.12, fc=fc, ec=ec, lw=lw, ls=ls, zorder=2)
    inner = w / U - 0.3
    body_lines = wrap_text(fig, body, body_size, inner) if body else []
    title_lines = wrap_text(fig, title, title_size, inner, "bold")
    th = title_size * 1.25 / 72 * U
    bh = body_size * 1.3 / 72 * U
    total = len(title_lines) * th + (0.35 + len(body_lines) * bh if body_lines else 0)
    cy = y + h / 2 + total / 2
    for ln in title_lines:
        ax.text(x + w / 2, cy, ln, ha="center", va="top", fontsize=title_size, fontweight="bold", color=title_color,
                zorder=3)
        cy -= th
    cy -= 0.35
    for ln in body_lines:
        ax.text(x + w / 2, cy, ln, ha="center", va="top", fontsize=body_size, color=INK2, zorder=3)
        cy -= bh


def diagram_arrow(ax, pts: list[tuple[float, float]], color: str = INK2, lw: float = 2.6, ls: str = "-") -> None:
    xs, ys = zip(*pts)
    if len(pts) > 2:
        ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, ls=ls, solid_capstyle="butt", zorder=1)
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>", mutation_scale=22, lw=lw, color=color, ls=ls,
                                 shrinkA=0, shrinkB=0, zorder=1))


def result_tile(fig: Figure, ax, x: float, y: float, w: float, h: float, accuracy: str, macro_f1: str, calls: str,
                n_series: int, color: str) -> None:
    """Headline numbers of one lane: big accuracy on the left, supporting numbers on the right."""
    round_box(fig, ax, x, y, w, h, 0.14, fc="white", ec=color, lw=3.0, zorder=2)
    cy = y + h / 2
    ax.text(x + 7.6, cy, accuracy, ha="center", va="center", fontsize=32, fontweight="bold", color=color, zorder=3)
    tx = x + 15.6
    ax.text(tx, cy + 3.5, "accuracy", ha="left", va="center", fontsize=11, color=MUTED, fontweight="bold", zorder=3)
    ax.text(tx, cy + 1.3, f"macro-F1 {macro_f1}", ha="left", va="center", fontsize=11.5, color=INK, zorder=3)
    ax.text(tx, cy - 0.9, f"{calls} Gemma calls", ha="left", va="center", fontsize=11.5, color=INK, fontweight="bold",
            zorder=3)
    ax.text(tx, cy - 3.0, f"for {n_series} series", ha="left", va="center", fontsize=10.5, color=MUTED, zorder=3)


def fig_llm_plus_expert(ctx: Context, out_dir: Path) -> Path | None:
    run = ctx.run
    ev = run.eval or {}
    comp = ev.get("comparison")
    n_win = len(ctx.windows)
    n_gated = int(ctx.gated.sum())
    n_clusters = sum(1 for c in run.clusters if c.cluster_id >= 0)
    n_feats = len(ctx.feats)
    n_full = int(run.series_meta["series_id"].nunique())

    fig = new_figure()
    header(fig, "How an LLM and an expert make forecast errors diagnosable",
           "Code measures the errors, Gemma explains them in plain language, the expert's knowledge steers both.")
    footer(fig, run)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 160)
    ax.set_ylim(0, 90)
    ax.axis("off")

    # lane backgrounds
    round_box(fig, ax, 2, 62.5, 156, 15.5, 0.2, fc="#F6F7F9", ec="#D5DBE3", lw=1.5, zorder=0)
    round_box(fig, ax, 2, 6.5, 156, 53.5, 0.2, fc="#F6F7F9", ec="#D5DBE3", lw=1.5, zorder=0)
    ax.text(4.5, 75.4, "A", fontsize=12, fontweight="bold", color="white", ha="center", va="center", zorder=3,
            bbox=dict(boxstyle="circle,pad=0.32", fc=INK2, ec="none"))
    ax.text(7, 75.4, "Gemma alone: one picture in, one guess out", fontsize=16, fontweight="bold", color=INK, va="center")
    ax.text(4.5, 57.2, "B", fontsize=12, fontweight="bold", color="white", ha="center", va="center", zorder=3,
            bbox=dict(boxstyle="circle,pad=0.32", fc=TSD_COLOR, ec="none"))
    ax.text(7, 57.2, "TimeSeriousDoctor: measure first, then explain, then let the expert correct",
            fontsize=16, fontweight="bold", color=INK, va="center")

    # lane A
    ya, ha = 64.0, 8.6
    diagram_box(fig, ax, 4, ya, 44, ha, "Full-series chart", "actual vs forecast in raw units. Nothing is measured.",
                fc="white", ec=INK2)
    diagram_box(fig, ax, 56, ya, 28, ha, "Gemma 4", "same model, same taxonomy in the prompt", fc=GEMMA_BG, ec=GEMMA_EDGE)
    diagram_box(fig, ax, 92, ya, 24, ha, "A guess", "failure type and explanation", fc="white", ec=INK2)
    diagram_arrow(ax, [(48, ya + ha / 2), (56, ya + ha / 2)])
    diagram_arrow(ax, [(84, ya + ha / 2), (92, ya + ha / 2)])

    # lane B, row 1: code measures
    round_box(fig, ax, 3, 42.7, 154, 13.4, 0.2, fc="#EEF4FC", ec="none", zorder=0)
    yb, hb, wb, gap = 44.8, 10.0, 26.0, 5.5
    steps = [
        ("Forecast series", "actual and predicted for every series"),
        ("Residuals and windows", "scaled errors cut into 28-step windows"),
        ("Anomaly gate", f"keeps the abnormal ones: {n_gated:,} of {n_win:,} windows"),
        ("Failure features", f"{n_feats} measured numbers per window: lag, bias, peak miss, flatness, ..."),
        ("Clusters", f"windows grouped into {n_clusters} recurring failure patterns"),
    ]
    for i, (t, b) in enumerate(steps):
        x = 4 + i * (wb + gap)
        diagram_box(fig, ax, x, yb, wb, hb, t, b, fc="white", ec=CODE_EDGE)
        if i:
            diagram_arrow(ax, [(x - gap, yb + hb / 2), (x, yb + hb / 2)])

    # lane B, row 2: evidence -> Gemma -> diagnosis
    round_box(fig, ax, 4, 9.5, 44, 29, 0.14, fc="white", ec=INK2, lw=2.0, zorder=1)
    ax.text(26, 36.4, "Evidence handed over per cluster", fontsize=12.5, fontweight="bold", color=INK, ha="center",
            va="center", zorder=3)
    minis = [
        (29.3, "Cluster chart panel", CODE_BG, CODE_EDGE, "-"),
        (23.1, "Measured feature table", CODE_BG, CODE_EDGE, "-"),
        (16.9, "Expert taxonomy: definitions, signatures, causes, fixes", EXPERT_BG, EXPERT_EDGE, "-"),
        (10.7, "Expert verdicts as few-shot examples", EXPERT_BG, EXPERT_EDGE, (0, (4, 2))),
    ]
    for y, label, fc, ec, ls in minis:
        round_box(fig, ax, 6, y, 40, 5.4, 0.1, fc=fc, ec=ec, lw=1.8, ls=ls, zorder=2)
        lines = wrap_text(fig, label, 11, 3.8)
        ax.text(26, y + 2.7, "\n".join(lines), ha="center", va="center", fontsize=11, color=INK, zorder=3,
                linespacing=1.15)
    diagram_arrow(ax, [(26, 43.2), (26, 38.4)])
    diagram_box(fig, ax, 56, 23, 28, 12.5, "Gemma 4", "reads the chart, the numbers and the taxonomy; answers in schema-validated JSON",
                fc=GEMMA_BG, ec=GEMMA_EDGE)
    diagram_box(fig, ax, 92, 23, 24, 12.5, "Diagnosis per cluster", "name, evidence, likely causes, fixes, metric to monitor",
                fc=GEMMA_BG, ec=GEMMA_EDGE)
    diagram_arrow(ax, [(48, 29.25), (56, 29.25)])
    diagram_arrow(ax, [(84, 29.25), (92, 29.25)])
    # row 1 -> row 2 connector (the clusters feed the evidence pack)
    diagram_arrow(ax, [(4 + 4 * (wb + gap) + wb / 2, yb), (4 + 4 * (wb + gap) + wb / 2, 41.0), (26, 41.0), (26, 38.4)])

    # expert review and the feedback loop
    diagram_box(fig, ax, 92, 8.2, 24, 9.0, "Expert review", "confirm, reject or rename", fc=EXPERT_BG, ec=EXPERT_EDGE)
    diagram_arrow(ax, [(104, 23), (104, 17.2)])
    diagram_arrow(ax, [(92, 13.4), (46, 13.4)], color=EXPERT_EDGE, ls=(0, (5, 3)))
    ax.text(69, 15.0, "next run: verdicts become few-shot examples", ha="center", va="bottom", fontsize=10.5,
            color=EXPERT_EDGE, fontweight="bold", style="italic")

    # results next to each lane's output
    if comp:
        p, b = comp["pipeline"], comp["gemma_alone"]
        calls = comp.get("gemma_calls_full_dataset", {})
        diagram_arrow(ax, [(116, ya + ha / 2), (122, ya + ha / 2)])
        result_tile(fig, ax, 122, ya + ha / 2 - 5.3, 35, 10.6, pct(b["accuracy"]), pct(b["macro_f1"]),
                    f"{calls.get('gemma_alone', b['gemma_calls'])}", n_full, BASE_COLOR)
        diagram_arrow(ax, [(116, 29.25), (122, 29.25)])
        result_tile(fig, ax, 122, 29.25 - 5.3, 35, 10.6, pct(p["accuracy"]), pct(p["macro_f1"]),
                    f"{calls.get('pipeline', p['gemma_calls'])}", n_full, TSD_COLOR)
        ax.text(139.5, 21.6, f"accuracy and macro-F1 on {comp.get('n_series', '')} series,", ha="center", va="center",
                fontsize=9.5, color=MUTED)
        ax.text(139.5, 20.0, "same model, same taxonomy", ha="center", va="center", fontsize=9.5, color=MUTED)

    # legend
    items = [("Code measures", CODE_BG, CODE_EDGE, "every number comes from code"),
             ("Gemma explains", GEMMA_BG, GEMMA_EDGE, "reads evidence, never computes"),
             ("Expert knowledge", EXPERT_BG, EXPERT_EDGE, "taxonomy, review, examples")]
    x = 4.0
    for name, fc, ec, sub in items:
        round_box(fig, ax, x, 3.2, 3.6, 2.2, 0.05, fc=fc, ec=ec, lw=1.8, zorder=2)
        ax.text(x + 4.6, 4.3, name, fontsize=12, fontweight="bold", color=INK, va="center")
        ax.text(x + 4.6 + U * text_width_in(fig, name, 12, "bold") + 1.0, 4.3, sub, fontsize=11, color=INK2,
                va="center")
        x += 4.6 + U * (text_width_in(fig, name, 12, "bold") + text_width_in(fig, sub, 11)) + 8.0
    return save(fig, out_dir, "04_llm_plus_expert.png")


# ---------------------------------------------------------------- figure 5: cluster gallery

def fig_cluster_gallery(ctx: Context, out_dir: Path) -> Path | None:
    run = ctx.run
    clusters = [c for c in run.clusters if c.cluster_id >= 0 and c.image_path and (run.path / c.image_path).exists()]
    clusters = sorted(clusters, key=lambda c: -c.size)[:4]
    if not clusters:
        print("skip 05_cluster_gallery: no cluster panel images in the run folder")
        return None

    W, H = 15.0, 13.6
    fig = new_figure(W, H)
    fig.text(0.04, 0.975, "What Gemma sees for each failure pattern", fontsize=27, fontweight="bold", color=INK,
             ha="left", va="top")
    fig.text(0.04, 0.945, "Cluster panels of the four largest clusters: three example windows (actual vs forecast) "
             "and the residuals of all member windows.", fontsize=14, color=INK2, ha="left", va="top")
    fig.text(0.04, 0.008, f"TimeSeriousDoctor  |  run {run.name}", fontsize=9, color=MUTED, ha="left", va="bottom")

    col_x = [0.04, 0.515]
    cw = 0.445
    img_h = cw * W * 0.75 / H          # panels are 4:3
    row_top = [0.895, 0.895 - img_h - 0.075]
    for i, c in enumerate(clusters):
        x, top = col_x[i % 2], row_top[i // 2]
        color = cluster_color(c.cluster_id)
        ax = fig.add_axes([x, top - img_h, cw, img_h])
        img = mpimg.imread(run.path / c.image_path)
        ax.imshow(img)
        pad_x, pad_y = 0.025 * img.shape[1], 0.025 * img.shape[0]   # white margin inside the colored frame
        ax.set_xlim(-pad_x, img.shape[1] + pad_x)
        ax.set_ylim(img.shape[0] + pad_y, -pad_y)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_color(color)
            sp.set_linewidth(3.5)
        fig.text(x + 0.004, top - img_h - 0.012, f"{c.cluster_id}", color=on_color(color), fontsize=13, fontweight="bold",
                 ha="center", va="top", bbox=dict(boxstyle="circle,pad=0.3", fc=color, ec="none"))
        ft = c.report.failure_type if c.report else None
        fig.text(x + 0.022, top - img_h - 0.011, cluster_name(c), fontsize=16, fontweight="bold", color=INK, ha="left",
                 va="top")
        detail = f"{c.size:,} windows from {c.n_series} series"
        if ft:
            detail += f"  |  diagnosed as {FAILURE_NAMES.get(ft, ft)}"
        fig.text(x + 0.022, top - img_h - 0.032, detail, fontsize=12, color=INK2, ha="left", va="top")
    return save(fig, out_dir, "05_cluster_gallery.png")


# ---------------------------------------------------------------- main

FIGURES = {
    1: fig_transformation,
    2: fig_latent_space,
    3: fig_pipeline_vs_gemma,
    4: fig_llm_plus_expert,
    5: fig_cluster_gallery,
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Explanatory figures from a real run (pitch deck, README).")
    ap.add_argument("--run", type=Path, required=True, help="run folder, e.g. runs/synth_features_v1")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "figures", help="output folder for the PNGs")
    ap.add_argument("--only", default="", help="comma-separated figure numbers to make (default: all)")
    ap.add_argument("--tsne-cache", type=Path, default=None, help=".npy file to cache the t-SNE coordinates in")
    args = ap.parse_args()

    set_style()
    ctx = load_context(args.run, args.tsne_cache)
    wanted = [int(x) for x in args.only.split(",") if x.strip()] or sorted(FIGURES)
    for n in wanted:
        FIGURES[n](ctx, args.out)


if __name__ == "__main__":
    main()
