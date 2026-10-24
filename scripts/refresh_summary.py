"""Refresh the executive summary (summary.json) of an existing run, touching nothing else.

    python scripts/refresh_summary.py --run runs/synth_features

Tries the Gemma summary call once more (cached client, raw replies into the run's gemma_raw/);
if it still fails, writes the summary composed in code from the cluster reports.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:  # `python scripts/refresh_summary.py` puts scripts/ first, not the repo root
    sys.path.insert(0, str(REPO_ROOT))

from atlas.config import GemmaConfig  # noqa: E402
from atlas.gemma import GemmaClient, summarize_atlas  # noqa: E402
from atlas.pipeline import fallback_summary  # noqa: E402
from atlas.schema import AtlasSummary, load_run  # noqa: E402


def main(argv: list[str] | None = None, gemma_client=None) -> AtlasSummary | None:
    ap = argparse.ArgumentParser(description="Refresh summary.json of a run (no other file is changed).")
    ap.add_argument("--run", type=Path, required=True, help="run folder, e.g. runs/synth_features")
    args = ap.parse_args(argv)

    run = load_run(args.run)
    gc = gemma_client
    if gc is None:
        gc = GemmaClient(GemmaConfig.model_validate(run.config.get("gemma", {})), raw_dir=run.path / "gemma_raw")

    summary = summarize_atlas(gc, run.clusters)
    source = "Gemma"
    if summary is None:
        summary = fallback_summary(run.clusters, run.windows)
        source = "composed from the cluster reports"
    if summary is None:
        print(f"{run.name}: no cluster has a report, so there is no summary to write")
        return None

    (run.path / "summary.json").write_text(json.dumps(summary.model_dump(mode="json"), indent=2), encoding="utf-8")
    print(f"{run.name}: wrote summary.json ({source}): {summary.headline}")
    return summary


if __name__ == "__main__":
    main()
