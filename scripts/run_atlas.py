"""CLI for run_atlas().

Owner: A. Task: P0.14. Spec: SPEC.md §7.10.

    python scripts/run_atlas.py --data data/synthetic --embedder features --run-name synth_features [--no-gemma]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:  # `python scripts/run_atlas.py` puts scripts/ first, not the repo root
    sys.path.insert(0, str(REPO_ROOT))

from atlas.config import load_config  # noqa: E402
from atlas.pipeline import run_atlas  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Run the TimeSeriousDoctor atlas pipeline on a dataset folder.")
    ap.add_argument("--data", type=Path, required=True, help="folder with data.parquet + series_meta.parquet")
    ap.add_argument("--run-name", required=True, help="run folder name under runs_dir")
    ap.add_argument("--embedder", choices=["features", "moment", "both"], help="override config embedder")
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "config.default.yaml")
    ap.add_argument("--gate-mode", choices=["threshold", "quantile"], help="override config gate.mode")
    ap.add_argument("--no-gemma", action="store_true", help="skip the Gemma reports and summary")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.embedder:
        cfg = cfg.model_copy(update={"embedder": args.embedder})
    if args.gate_mode:
        cfg = cfg.model_copy(update={"gate": cfg.gate.model_copy(update={"mode": args.gate_mode})})
    run_atlas(args.data, cfg, args.run_name, use_gemma=not args.no_gemma)


if __name__ == "__main__":
    main()
