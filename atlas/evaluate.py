"""Metrics on labeled (synthetic) runs.

Pure functions over a RunResult whose windows carry `true_label`, `gated`, `cluster`
and whose data rows carry `true_label`. Includes the pipeline-vs-Gemma-alone comparison
(SPEC section 7.11). Label mapping everywhere: healthy_noise -> healthy.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    f1_score,
    normalized_mutual_info_score,
    precision_recall_fscore_support,
)

from atlas.schema import CLUSTER_NOISE, RunResult

HEALTHY = "healthy"
HISTORY = "history"
OTHER = "other"
_MISSING = "__missing__"  # stands in for a None/absent baseline answer (never equals a truth class)


def _norm(label: str) -> str:
    return HEALTHY if label == "healthy_noise" else label


def _majority(labels) -> str:
    """Most common label; ties broken alphabetically so results are deterministic."""
    counts = Counter(labels)
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _cluster_type(run: RunResult, cluster_id: int) -> str:
    """Mapped failure_type of a cluster's Gemma report; `other` if there is no report."""
    for c in run.clusters:
        if c.cluster_id == cluster_id:
            return _norm(c.report.failure_type) if c.report is not None else OTHER
    return OTHER


# ---------------------------------------------------------------- series level

def series_truth(run: RunResult) -> dict[str, str]:
    """Injected failure type per series: the non-healthy, non-history forecast-row label, else healthy."""
    if "true_label" not in run.data.columns:
        raise ValueError("run.data has no true_label column (labeled runs only)")
    fc = run.data[run.data["is_forecast"].astype(bool)]
    truth = {sid: HEALTHY for sid in run.data["series_id"].unique()}
    for sid, labels in fc.groupby("series_id")["true_label"]:
        failures = [_norm(x) for x in labels if _norm(x) not in (HEALTHY, HISTORY)]
        if failures:
            truth[sid] = _majority(failures)
    return dict(sorted(truth.items()))


def series_predictions(run: RunResult) -> dict[str, str]:
    """Pipeline's predicted failure type per series (rule in SPEC 7.11)."""
    w = run.windows
    gated = w[w["gated"].astype(bool)]
    preds = {sid: HEALTHY for sid in run.data["series_id"].unique()}
    for sid, g in gated.groupby("series_id"):
        clustered = [int(c) for c in g["cluster"] if int(c) >= 0]
        if not clustered:
            preds[sid] = OTHER  # gated, but only noise windows
            continue
        counts = Counter(clustered)
        best = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        preds[sid] = _cluster_type(run, best)
    return dict(sorted(preds.items()))


# ---------------------------------------------------------------- run level

def _purity(true: np.ndarray, clusters: np.ndarray) -> float:
    total = 0
    for c in np.unique(clusters):
        total += Counter(true[clusters == c]).most_common(1)[0][1]
    return float(total / len(true))


def evaluate_run(run: RunResult) -> dict:
    """Gate, clustering, naming and series-level metrics for one labeled run."""
    w = run.windows
    if "true_label" not in w.columns:
        raise ValueError("run.windows has no true_label column (labeled runs only)")
    true = w["true_label"].map(_norm).to_numpy()
    gated = w["gated"].to_numpy().astype(bool)
    cluster = w["cluster"].to_numpy().astype(int)
    positive = true != HEALTHY

    p, r, f1, _ = precision_recall_fscore_support(
        positive.astype(int), gated.astype(int), average="binary", zero_division=0
    )

    # clustering quality on gated, truly failing windows that landed in a real cluster
    sel = gated & positive
    clustered = sel & (cluster >= 0)
    if clustered.any():
        ari = float(adjusted_rand_score(true[clustered], cluster[clustered]))
        nmi = float(normalized_mutual_info_score(true[clustered], cluster[clustered]))
        purity = _purity(true[clustered], cluster[clustered])
    else:
        ari = nmi = purity = 0.0
    noise_share = float((cluster[sel] == CLUSTER_NOISE).mean()) if sel.any() else 0.0

    confusion: dict[str, dict[str, int]] = {}
    for c in sorted(set(cluster[gated])):
        counts = Counter(true[gated & (cluster == c)])
        confusion[str(int(c))] = {k: int(v) for k, v in sorted(counts.items())}

    # naming accuracy: size-weighted over every window of every real cluster
    n_ok = n_all = 0
    for c in sorted(set(cluster[gated & (cluster >= 0)])):
        members = true[gated & (cluster == c)]
        n_all += len(members)
        if _cluster_type(run, int(c)) == _majority(members):
            n_ok += len(members)
    naming_accuracy = float(n_ok / n_all) if n_all else 0.0

    truth = series_truth(run)
    preds = series_predictions(run)
    series_accuracy = float(np.mean([preds.get(s) == t for s, t in truth.items()])) if truth else 0.0

    return {
        "run": run.name,
        "embedder": run.config.get("embedder", "unknown"),
        "gate": {"precision": float(p), "recall": float(r), "f1": float(f1)},
        "clustering": {
            "ari": ari, "nmi": nmi, "purity": purity,
            "noise_share": noise_share, "confusion": confusion,
        },
        "naming_accuracy": naming_accuracy,
        "series_accuracy": series_accuracy,
    }


# ---------------------------------------------------------------- pipeline vs Gemma alone

def stratified_sample(truth: dict[str, str], n: int, seed: int = 0) -> list[str]:
    """~n/num_classes series per class (a class with fewer series gives all of them); sorted ids."""
    by_class: dict[str, list[str]] = {}
    for sid in sorted(truth):
        by_class.setdefault(truth[sid], []).append(sid)
    if not by_class or n <= 0:
        return []
    classes = sorted(by_class)
    base, extra = divmod(n, len(classes))
    rng = np.random.default_rng(seed)
    picked: list[str] = []
    for i, cls in enumerate(classes):
        ids = by_class[cls]
        take = min(base + (1 if i < extra else 0), len(ids))
        idx = rng.choice(len(ids), size=take, replace=False)
        picked.extend(ids[j] for j in idx)
    return sorted(picked)


def _score(truth: dict[str, str], pred: dict[str, str | None], series_ids: list[str]) -> dict:
    y_true = [_norm(truth[s]) for s in series_ids]
    y_pred = [_MISSING if pred.get(s) is None else _norm(pred[s]) for s in series_ids]
    classes = sorted(set(y_true))
    correct = [a == b for a, b in zip(y_true, y_pred)]
    per_class = {
        c: float(np.mean([ok for t, ok in zip(y_true, correct) if t == c])) for c in classes
    }
    return {
        "accuracy": float(np.mean(correct)) if correct else 0.0,
        "macro_f1": float(f1_score(y_true, y_pred, labels=classes, average="macro", zero_division=0))
        if correct else 0.0,
        "per_class": per_class,
    }


def compare(
    truth: dict[str, str],
    pipeline_pred: dict[str, str | None],
    baseline_pred: dict[str, str | None],
    series_ids: list[str],
    pipeline_calls: int,
    n_series_total: int,
    n_clusters_without_report: int = 0,
) -> dict:
    """The eval.json `comparison` block: both methods scored on exactly `series_ids`.

    A missing or failed (None) Gemma-alone answer is scored wrong, the honest outcome for the naive
    solution, but counted in `gemma_alone.n_failed` so the table can say so. The pipeline side
    records `n_clusters_without_report`: those clusters are named `other`, i.e. wrong.
    """
    n_failed = sum(baseline_pred.get(s) is None for s in series_ids)
    return {
        "n_series": len(series_ids),
        "pipeline": {
            **_score(truth, pipeline_pred, series_ids),
            "gemma_calls": pipeline_calls,
            "n_clusters_without_report": n_clusters_without_report,
        },
        "gemma_alone": {
            **_score(truth, baseline_pred, series_ids),
            "gemma_calls": len(series_ids),
            "n_failed": int(n_failed),
        },
        "gemma_calls_full_dataset": {"pipeline": pipeline_calls, "gemma_alone": n_series_total},
    }


# ---------------------------------------------------------------- markdown

def _pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def _num(x: float | None) -> str:
    return "-" if x is None else f"{x:.2f}"


def markdown_table(evals: list[dict]) -> str:
    """Pipeline-vs-Gemma-alone table (first eval with a comparison), then one row per run."""
    parts: list[str] = []

    comp = next((e["comparison"] for e in evals if e.get("comparison")), None)
    if comp is not None:
        full = comp["gemma_calls_full_dataset"]
        rows = [
            ("Pipeline (ours)", comp["pipeline"], full["pipeline"]),
            ("Gemma alone", comp["gemma_alone"], full["gemma_alone"]),
        ]
        lines = [
            f"### Pipeline vs Gemma alone ({comp['n_series']} series, same sample for both)",
            "",
            "| method | accuracy | macro-F1 | Gemma calls (sample) | Gemma calls (full dataset) |",
            "|---|---|---|---|---|",
        ]
        for name, m, full_calls in rows:
            lines.append(
                f"| {name} | {_pct(m['accuracy'])} | {_num(m['macro_f1'])} "
                f"| {m['gemma_calls']} | {full_calls} |"
            )
        n_failed = comp["gemma_alone"].get("n_failed", 0)
        n_unnamed = comp["pipeline"].get("n_clusters_without_report", 0)
        notes = []
        if n_failed:
            notes.append(
                f"Note: {n_failed} of {comp['n_series']} Gemma-alone calls failed and are scored as wrong; "
                "re-run to retry them (cached answers cost nothing)."
            )
        if n_unnamed:
            notes.append(
                f"Note: {n_unnamed} pipeline cluster(s) have no Gemma report and are scored as `other` (wrong)."
            )
        for note in notes:
            lines += ["", note]
        parts.append("\n".join(lines))

    lines = [
        "### Clustering metrics per run",
        "",
        "| run | embedder | gate F1 | ARI | NMI | purity | noise share | naming acc | series acc |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for e in evals:
        cl = e.get("clustering", {})
        lines.append(
            f"| {e.get('run', '-')} | {e.get('embedder', '-')} "
            f"| {_num(e.get('gate', {}).get('f1'))} | {_num(cl.get('ari'))} | {_num(cl.get('nmi'))} "
            f"| {_num(cl.get('purity'))} | {_pct(cl.get('noise_share'))} "
            f"| {_pct(e.get('naming_accuracy'))} | {_pct(e.get('series_accuracy'))} |"
        )
    parts.append("\n".join(lines))
    return "\n\n".join(parts) + "\n"
