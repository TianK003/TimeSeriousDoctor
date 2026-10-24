"""Prepare the M5 dataset.

Reads the Kaggle M5 files, aggregates items over stores, runs statsforecast cross-validation and writes
`data.parquet`, `series_meta.parquet` and `calendar.parquet`:

    python scripts/prepare_m5.py --raw data/m5_raw --n-items 1000 --history-days 730 --out data/m5

Runtime guard: 50- and 100-item probes are timed first; if the extrapolated forecasting time exceeds
`--max-minutes` (default 5), `--n-items` drops to 300 (`--no-guard` disables the check).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from anywhere without installing atlas

from atlas.data.m5 import (  # noqa: E402
    aggregate_item_total,
    load_m5_raw,
    make_forecasts,
    save_calendar,
    to_contract,
)
from atlas.schema import Dataset, save_dataset  # noqa: E402

FALLBACK_ITEMS = 300
PROBE_ITEMS = 50


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=Path("data/m5_raw"))
    ap.add_argument("--n-items", type=int, default=1000)
    ap.add_argument("--history-days", type=int, default=730)
    ap.add_argument("--out", type=Path, default=Path("data/m5"))
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--max-minutes", type=float, default=5.0)
    ap.add_argument("--no-guard", action="store_true", help="skip the timing probes")
    args = ap.parse_args()

    t0 = time.time()
    sales, calendar = load_m5_raw(args.raw)
    print(f"load_m5_raw: {len(sales)} item x store rows, {calendar.shape[0]} calendar days ({time.time() - t0:.1f}s)")

    t1 = time.time()
    n_items = args.n_items
    df, item_dept = aggregate_item_total(sales, calendar, n_items=n_items, history_days=args.history_days)
    print(
        f"aggregate_item_total: {df['unique_id'].nunique()} items, {len(df)} rows, "
        f"{df['ds'].min().date()} .. {df['ds'].max().date()} ({time.time() - t1:.1f}s)"
    )

    if not args.no_guard and n_items > 2 * PROBE_ITEMS:
        # two probes (50 and 100 items) so the fixed start-up cost (workers, numba) is not extrapolated
        ids = df["unique_id"].unique()
        probe_s = []
        for k in (PROBE_ITEMS, 2 * PROBE_ITEMS):
            tp = time.time()
            make_forecasts(df[df["unique_id"].isin(ids[:k])], n_jobs=args.n_jobs)
            probe_s.append(time.time() - tp)
        per_item = max(probe_s[1] - probe_s[0], 0.0) / PROBE_ITEMS
        est_min = (probe_s[1] + per_item * (n_items - 2 * PROBE_ITEMS)) / 60
        print(
            f"guard: probes {probe_s[0]:.1f}s ({PROBE_ITEMS} items) and {probe_s[1]:.1f}s ({2 * PROBE_ITEMS} items) "
            f"-> ~{est_min:.1f} min for {n_items} items"
        )
        if est_min > args.max_minutes:
            n_items = FALLBACK_ITEMS
            print(f"guard: over {args.max_minutes:g} min, falling back to --n-items {n_items}")
            df, item_dept = aggregate_item_total(sales, calendar, n_items=n_items, history_days=args.history_days)

    t2 = time.time()
    cv = make_forecasts(df, n_jobs=args.n_jobs)
    data, meta = to_contract(df, cv, item_dept)
    save_dataset(Dataset(data=data, series_meta=meta), args.out)
    save_calendar(calendar, args.out)

    n_hist = int((~data["is_forecast"]).groupby(data["series_id"]).sum().iloc[0])
    n_fc = int(data["is_forecast"].groupby(data["series_id"]).sum().iloc[0])
    print(
        f"wrote {args.out}: {df['unique_id'].nunique()} items -> {data['series_id'].nunique()} series "
        f"({n_hist} history + {n_fc} forecast rows each), {len(data)} rows, groups {sorted(meta['group'].unique())}"
    )
    print(f"timings: forecasting+contract {time.time() - t2:.1f}s, total {time.time() - t0:.1f}s")


if __name__ == "__main__":  # required: statsforecast n_jobs=-1 spawns worker processes on Windows
    main()
