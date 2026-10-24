"""Evaluate runs and print a comparison table.

Owner: C. Task: P1.3. Spec: SPEC.md §7.11.

    python scripts/run_eval.py --runs runs/synth_features [runs/synth_moment ...] [--n-baseline 40] [--no-baseline]

Writes eval.json into every run folder and baseline.json into the first one. The Gemma-alone
baseline does not depend on the embedder, so it is computed once (on the first run) and every
run's pipeline predictions are scored against it on the same sample of series.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # `python scripts/run_eval.py` puts scripts/ first, not the repo root
    sys.path.insert(0, str(REPO_ROOT))

from atlas.baseline import run_baseline  # noqa: E402
from atlas.config import GemmaConfig, get_api_key  # noqa: E402
from atlas.evaluate import (  # noqa: E402
    compare,
    evaluate_run,
    markdown_table,
    series_predictions,
    series_truth,
    stratified_sample,
)
from atlas.features import FEATURE_DOCS  # noqa: E402
from atlas.gemma import GemmaClient  # noqa: E402
from atlas.prompts import system_prompt  # noqa: E402
from atlas.schema import RunResult, load_run  # noqa: E402
from atlas.taxonomy import load_taxonomy, render_taxonomy  # noqa: E402


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _baseline(
    run: RunResult, n_baseline: int, taxonomy: Path, gemma_client
) -> tuple[list[str], dict[str, str | None]]:
    """Gemma alone on a stratified sample of the run's series; saves baseline.json in the run folder."""
    truth = series_truth(run)
    sample = stratified_sample(truth, n_baseline)
    gc = gemma_client
    if gc is None:
        if not get_api_key():
            raise SystemExit("no API key (set GEMMA4_API_KEY in .env), or pass --no-baseline")
        cfg = GemmaConfig.model_validate(run.config.get("gemma", {}))
        gc = GemmaClient(cfg, raw_dir=run.path / "gemma_raw")
    system = system_prompt(render_taxonomy(load_taxonomy(taxonomy)), FEATURE_DOCS)

    print(f"baseline: Gemma alone on {len(sample)} of {len(truth)} series ({run.name})")
    answers = run_baseline(run, sample, gc, system)
    pred = {
        sid: None if a is None else ("healthy" if a.failure_type == "healthy_noise" else a.failure_type)
        for sid, a in answers.items()
    }
    _write_json(run.path / "baseline.json", {
        "truth": {sid: truth[sid] for sid in sample},
        "answers": {sid: None if a is None else a.model_dump(mode="json") for sid, a in answers.items()},
    })
    n_failed = sum(a is None for a in answers.values())
    if n_failed:
        print(f"baseline: {n_failed} of {len(sample)} Gemma calls failed (counted as wrong)")
    return sample, pred


def main(argv: list[str] | None = None, gemma_client=None) -> list[dict]:
    parser = argparse.ArgumentParser(description="Evaluate runs and compare the pipeline with Gemma alone.")
    parser.add_argument("--runs", type=Path, nargs="+", required=True, help="run folders, e.g. runs/synth_features")
    parser.add_argument("--n-baseline", type=int, default=40, help="series sampled for the Gemma-alone baseline")
    parser.add_argument("--no-baseline", action="store_true", help="skip the Gemma-alone comparison")
    parser.add_argument("--taxonomy", type=Path, default=REPO_ROOT / "taxonomy.yaml")
    args = parser.parse_args(argv)

    runs = [load_run(p) for p in args.runs]
    sample: list[str] = []
    baseline_pred: dict[str, str | None] = {}
    if not args.no_baseline:
        sample, baseline_pred = _baseline(runs[0], args.n_baseline, args.taxonomy, gemma_client)

    evals = []
    for run in runs:
        ev = evaluate_run(run)
        if not args.no_baseline:
            pipeline_calls = sum(1 for c in run.clusters if c.cluster_id >= 0) + 1
            truth = series_truth(run)
            ev["comparison"] = compare(
                truth, series_predictions(run), baseline_pred, sample, pipeline_calls, len(truth)
            )
        _write_json(run.path / "eval.json", ev)
        print(f"wrote {run.path / 'eval.json'}")
        evals.append(ev)

    print()
    print(markdown_table(evals))
    return evals


if __name__ == "__main__":
    main()
