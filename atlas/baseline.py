"""Gemma-alone baseline on full-series images.

The naive solution the pipeline is compared against (SPEC.md section 7.11): one Gemma call per
series with the actual-vs-predicted chart and the model card, same system prompt, taxonomy and
model as the pipeline. No residuals, no features, no clustering.
"""

from __future__ import annotations

from atlas.plots import baseline_series_png
from atlas.prompts import baseline_user_prompt
from atlas.schema import GemmaBaselineAnswer, RunResult

PROGRESS_EVERY = 10


def _model_cards(run: RunResult) -> dict[str, str]:
    if "model_card" not in run.series_meta.columns:
        return {}
    return dict(zip(run.series_meta["series_id"].astype(str), run.series_meta["model_card"].astype(str)))


def run_baseline(run: RunResult, series_ids: list[str], gc, system: str) -> dict[str, GemmaBaselineAnswer | None]:
    """One Gemma call per series (chart + model card). A failed call is None, never an exception."""
    if run.path is None:
        raise ValueError("run.path is not set (baseline charts are written into the run folder)")
    image_dir = run.path / "baseline_images"
    cards = _model_cards(run)
    by_series = {sid: rows for sid, rows in run.data.groupby("series_id")}

    answers: dict[str, GemmaBaselineAnswer | None] = {}
    for i, sid in enumerate(series_ids, start=1):
        png = baseline_series_png(by_series[sid], path=image_dir / f"{sid}.png", title=sid)
        answers[sid] = gc.generate(
            system,
            baseline_user_prompt(cards.get(sid, "unknown")),
            GemmaBaselineAnswer,
            image=png,
            tag=f"baseline_{sid}",
        )
        if i % PROGRESS_EVERY == 0 or i == len(series_ids):
            print(f"baseline: {i}/{len(series_ids)} series")
    return answers
