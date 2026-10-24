"""Ask the Doctor: grounded question answering over one run's measured evidence.

Code measures, Gemma explains: `build_context` assembles the facts from the run folder (numbers,
cluster reports, expert verdicts) and Gemma only interprets them. User-supplied strings (model
cards, series ids, expert notes) reach the prompt through `prompts.clean_text` inside `<data>`
blocks, like every other prompt.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import TYPE_CHECKING

from pydantic import BaseModel

from atlas import prompts
from atlas.features import FEATURE_DOCS
from atlas.prompts import clean_text, data_block
from atlas.schema import CLUSTER_NOISE, ClusterInfo, RunResult

if TYPE_CHECKING:
    from atlas.gemma import GemmaClient

MAX_CONTEXT_CHARS = 6500           # taxonomy is trimmed first so the facts above it stay whole
MAX_TAXONOMY_CHARS = 1000
MAX_CLUSTER_LINES = 12
MAX_SERIES_WINDOWS = 12
MAX_MODEL_LINES = 8
SUMMARY_CHARS = 160                # first sentence of the cluster summary in the overview
REPORT_ITEM_CHARS = 240
SERIES_FEATURES = ["abs_err", "bias", "best_lag", "var_ratio", "peak_excess", "sign_run"]
EXPERT_STATUSES = {"confirmed", "rejected", "renamed"}
_THOUGHT_LEAK = re.compile(r"thought\s*:", re.IGNORECASE)


class ChatAnswer(BaseModel):
    answer: str                  # <= 6 sentences, plain language
    evidence: list[str]          # 1-4 items, each quotes a provided number or report
    suggested_next_step: str
    confidence: float            # 0-1


# ---------------------------------------------------------------- small formatters

def _t(text: object, limit: int = prompts.MAX_FIELD_CHARS) -> str:
    """`clean_text` for the context, with double quotes turned into single quotes.

    Gemma copies names from the facts into JSON strings; a `"` in the facts makes it escape quotes,
    and with schema-constrained output it then degenerates into runs of `\\"'\\"'...` (seen in real calls).
    """
    return clean_text(str(text).replace('"', "'"), limit)


def _num(value, digits: int = 2) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if v != v else f"{v:.{digits}f}"


def _pct(value) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if v != v else f"{v:.0%}"


def _first_sentence(text: str, limit: int = SUMMARY_CHARS) -> str:
    flat = _t(text, 1000)
    first = flat.split(". ")[0]
    return _t(first if first.endswith(".") else first + ".", limit)


def _cluster_label(cid: int) -> str:
    return "unclustered windows" if cid == CLUSTER_NOISE else f"cluster {cid}"


def _cluster_ref(run: RunResult, cid: int) -> str:
    """'cluster 1 ("Gemma name")', or just the label for the unclustered bucket / a cluster without a report."""
    try:
        report = run.cluster(cid).report
    except KeyError:
        report = None
    if report is None:
        return _cluster_label(cid)
    return f"{_cluster_label(cid)} ({_t(report.name, 80)})"


def _expert_text(run: RunResult, cid: int) -> str:
    """One-line expert verdict for a cluster ('' if there is none)."""
    entry = (run.expert_feedback or {}).get(str(cid))
    if not isinstance(entry, dict) or entry.get("status") not in EXPERT_STATUSES:
        return ""
    text = f"expert {entry['status']}"
    if entry["status"] == "renamed" and entry.get("name"):
        text += f" as {_t(entry['name'], 80)}"
    note = _t(entry.get("note", ""))
    return text + (f"; expert note: {note}" if note else "")


# ---------------------------------------------------------------- context sections

def _headline(run: RunResult) -> str:
    w = run.windows
    n_clusters = sum(c.cluster_id >= 0 for c in run.clusters)
    n_unclustered = int((w["cluster"] == CLUSTER_NOISE).sum())
    window_len = run.config.get("window_len", 28)
    return (
        f"Run overview: {run.data['series_id'].nunique()} series, {len(w)} windows of {window_len} forecast "
        f"steps, {int(w['gated'].sum())} anomalous, {n_clusters} clusters, {n_unclustered} anomalous windows "
        f"left unclustered; embedder: {_t(run.config.get('embedder', 'unknown'), 40)}."
    )


def _eval_lines(ev: dict | None) -> list[str]:
    if not ev:
        return []
    lines = []
    comp = ev.get("comparison") or {}
    pipe, alone = comp.get("pipeline") or {}, comp.get("gemma_alone") or {}
    if pipe and alone:
        lines.append(
            f"Evaluation (synthetic series with known failures, {comp.get('n_series', 'n/a')}-series sample): "
            f"full pipeline accuracy {_pct(pipe.get('accuracy'))} (macro-F1 {_num(pipe.get('macro_f1'))}, "
            f"{pipe.get('gemma_calls', 'n/a')} Gemma calls) vs Gemma alone on raw plots "
            f"{_pct(alone.get('accuracy'))} (macro-F1 {_num(alone.get('macro_f1'))}, "
            f"{alone.get('gemma_calls', 'n/a')} Gemma calls)."
        )
    gate, clu = ev.get("gate") or {}, ev.get("clustering") or {}
    if gate:
        lines.append(
            f"Anomaly gate: precision {_num(gate.get('precision'))}, recall {_num(gate.get('recall'))}, "
            f"F1 {_num(gate.get('f1'))}."
        )
    if clu or ev.get("naming_accuracy") is not None:
        lines.append(
            f"Clustering vs true failure types: ARI {_num(clu.get('ari'))}, NMI {_num(clu.get('nmi'))}, "
            f"purity {_pct(clu.get('purity'))}, noise share {_pct(clu.get('noise_share'))}; "
            f"naming accuracy {_pct(ev.get('naming_accuracy'))}, series accuracy {_pct(ev.get('series_accuracy'))}."
        )
    return lines


def _cluster_overview(run: RunResult) -> str:
    lines = []
    for c in run.clusters[:MAX_CLUSTER_LINES]:
        label = _cluster_label(c.cluster_id)
        r = c.report
        if r is not None:
            line = (
                f"- {label}: {_t(r.name, 80)} [{r.failure_type}], {c.size} windows from {c.n_series} "
                f"series, confidence {_num(r.confidence)}. {_first_sentence(r.summary)}"
            )
        else:
            line = f"- {label}: {c.size} windows from {c.n_series} series, no Gemma explanation."
        expert = _expert_text(run, c.cluster_id)
        lines.append(f"{line} ({expert})" if expert else line)
    if len(run.clusters) > MAX_CLUSTER_LINES:
        lines.append(f"- (+{len(run.clusters) - MAX_CLUSTER_LINES} more clusters)")
    return "Clusters (anomalous windows grouped by error pattern):\n" + data_block(lines)


def _model_summary(run: RunResult) -> str:
    """Per forecasting model: how many of its windows are anomalous and in which clusters (code-measured)."""
    meta = run.series_meta[["series_id", "model_card"]].drop_duplicates("series_id")
    w = run.windows[["series_id", "gated", "cluster"]].merge(meta, on="series_id", how="left")
    w["model_card"] = w["model_card"].fillna("unknown")
    stats = []
    for card, g in w.groupby("model_card"):
        gated = g[g["gated"].astype(bool)]
        stats.append((len(gated) / len(g), str(card), g, gated))
    stats.sort(key=lambda s: (-s[0], s[1]))
    lines = []
    for share, card, g, gated in stats[:MAX_MODEL_LINES]:
        line = f"- {_t(card)}: {g['series_id'].nunique()} series, {len(gated)} of {len(g)} windows anomalous ({share:.0%})"
        if len(gated):
            top = Counter(int(c) for c in gated["cluster"]).most_common(2)
            line += "; main clusters: " + ", ".join(f"{_cluster_label(c)} ({n / len(gated):.0%})" for c, n in top)
        lines.append(line)
    if len(stats) > MAX_MODEL_LINES:
        lines.append(f"- (+{len(stats) - MAX_MODEL_LINES} more models)")
    return "Forecasting models, ranked by share of anomalous windows:\n" + data_block(lines)


def _report_block(run: RunResult, c: ClusterInfo) -> str:
    r = c.report
    if r is None:
        return "No Gemma report for this cluster."

    def item(text: str) -> str:
        return _t(text, REPORT_ITEM_CHARS)

    lines = [
        f"- name: {_t(r.name, 80)}; failure type: {r.failure_type}; confidence {_num(r.confidence)}",
        f"- summary: {item(r.summary)}",
    ]
    lines += [f"- evidence: {item(e)}" for e in r.evidence]
    lines += [f"- likely cause: {item(k.cause)} (why: {item(k.why)})" for k in r.likely_causes]
    lines += [f"- suggested fix: {item(f)}" for f in r.suggested_fixes]
    lines.append(f"- metric to monitor: {item(r.metric_to_monitor)}")
    expert = _expert_text(run, c.cluster_id)
    if expert:
        lines.append(f"- {expert}")
    return data_block(lines)


def _cluster_focus(run: RunResult, cluster_id: int) -> str:
    try:
        c = run.cluster(cluster_id)
    except KeyError:
        return f"Focus: cluster {cluster_id} does not exist in this run."
    parts = [
        f"Focus: {_cluster_label(cluster_id)}, {c.size} windows from {c.n_series} series. Full report:\n"
        + _report_block(run, c)
    ]
    if c.feature_median:
        names = [f for f in FEATURE_DOCS if f in c.feature_median] + [f for f in c.feature_median if f not in FEATURE_DOCS]
        rows = [
            f"- {_t(n, 40)}: {_num(c.feature_median[n])} vs {_num(c.feature_median_all_gated.get(n))}"
            for n in names
        ]
        parts.append("Median features, this cluster vs all anomalous windows (scaled units):\n" + "\n".join(rows))
    if c.model_card_dist:
        parts.append("Forecasting models in this cluster (share of windows):\n" + data_block(prompts._model_card_lines(c.model_card_dist)))
    return "\n".join(parts)


def _series_focus(run: RunResult, series_id: str) -> str:
    w = run.windows[run.windows["series_id"] == series_id].sort_values("start_idx")
    cards = run.series_meta.loc[run.series_meta["series_id"] == series_id, "model_card"]
    card = _t(cards.iloc[0]) if len(cards) else "unknown"
    parts = ["Focus: one series.\n" + data_block([f"- series_id: {_t(series_id, 80)}", f"- model card: {card}"])]
    if w.empty:
        parts.append("This series has no forecast windows in this run.")
        return "\n".join(parts)

    gated = w[w["gated"].astype(bool)]
    parts.append(f"It has {len(w)} forecast windows, {len(gated)} anomalous.")
    if gated.empty:
        return "\n".join(parts)

    err = "abs_err" if "abs_err" in gated.columns else "score_abs"
    shown = gated if len(gated) <= MAX_SERIES_WINDOWS else gated.nlargest(MAX_SERIES_WINDOWS, err).sort_values("start_idx")
    cols = [f for f in SERIES_FEATURES if f in shown.columns] or ["score_abs", "score_bias"]
    rows = []
    for _, row in shown.iterrows():
        cid = int(row["cluster"])
        feats = ", ".join(f"{f} {_num(row[f])}" for f in cols)
        rows.append(f'- {str(row["start_ts"])[:10]}: {_cluster_ref(run, cid)}; {feats}')
    if len(gated) > len(shown):
        rows.append(f"- (+{len(gated) - len(shown)} more anomalous windows not shown)")
    parts.append("Anomalous windows of this series (start of window; scaled units):\n" + data_block(rows))

    known = {c.cluster_id for c in run.clusters}
    counts = Counter(int(c) for c in gated["cluster"] if int(c) >= 0 and int(c) in known)
    if counts:
        cid, n = counts.most_common(1)[0]
        parts.append(
            f"Most of this series' anomalous windows ({n} of {len(gated)}) are in cluster {cid}. "
            "Its report:\n" + _report_block(run, run.cluster(cid))
        )
    return "\n".join(parts)


def short_taxonomy(tax: dict) -> str:
    """One line per failure type (name and definition) for the context; the full taxonomy is too long."""
    return "\n".join(f"- {name}: {_t(entry.get('definition', ''), 160)}" for name, entry in tax.items())


# ---------------------------------------------------------------- public API

def build_context(
    run: RunResult,
    cluster_id: int | None = None,
    series_id: str | None = None,
    taxonomy_text: str = "",
) -> str:
    """Plain-text facts about this run for one question (compact, about 6k chars at most).

    Always: run headline, evaluation numbers, one line per cluster. Optionally the focus cluster's
    report and feature medians, and/or the focus series' model card and anomalous windows with the
    report of the cluster holding most of them.
    """
    parts = [_headline(run), *_eval_lines(run.eval), _cluster_overview(run), _model_summary(run)]
    if cluster_id is not None:
        parts.append(_cluster_focus(run, cluster_id))
    if series_id is not None:
        parts.append(_series_focus(run, series_id))
    head = "\n\n".join(parts)
    if taxonomy_text.strip():
        room = min(MAX_TAXONOMY_CHARS, MAX_CONTEXT_CHARS - len(head) - 60)
        if room > 100:
            tax = taxonomy_text.strip()
            tax = tax if len(tax) <= room else tax[: room - 3] + "..."
            head += "\n\nFailure-type taxonomy (reference):\n" + tax
    return head


def answer_question(
    gc: GemmaClient,
    run: RunResult,
    question: str,
    history: list[tuple[str, str]] | None = None,
    cluster_id: int | None = None,
    series_id: str | None = None,
    taxonomy_text: str = "",
) -> ChatAnswer | None:
    """Gemma's grounded answer to `question`, or None if the call failed (generate never raises)."""
    context = build_context(run, cluster_id=cluster_id, series_id=series_id, taxonomy_text=taxonomy_text)
    user = prompts.chat_user_prompt(context, question, (history or [])[-3:])
    result = gc.generate(prompts.CHAT_SYSTEM, user, ChatAnswer, tag="chat")
    return _without_thought_leak(result) if result is not None else None


def _strip_leak(text: str) -> str:
    """Cut a leaked reasoning marker ("thought: ...") and the sentence it interrupted from a model-written field."""
    m = _THOUGHT_LEAK.search(text)
    if m is None:
        return text
    kept = text[: m.start()].rstrip()
    ends = [e.end() for e in re.finditer(r"[.!?](?:\s|$)", kept)]  # a decimal point is not a sentence end
    return kept[: ends[-1]].rstrip() if ends else kept


def _without_thought_leak(a: ChatAnswer) -> ChatAnswer:
    """Gemma sometimes leaks its reasoning channel into a JSON string (seen once in three real answers).

    A field that is nothing but leaked reasoning keeps its original text (there is nothing better to show);
    evidence items that are nothing but leaked reasoning are dropped.
    """
    return a.model_copy(update={
        "answer": _strip_leak(a.answer) or a.answer,
        "evidence": [cleaned for e in a.evidence if (cleaned := _strip_leak(e))],
        "suggested_next_step": _strip_leak(a.suggested_next_step) or a.suggested_next_step,
    })
