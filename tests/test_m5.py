"""M5 prep on a tiny fake wide sales file (no network, no Kaggle data)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from atlas.data.m5 import (
    aggregate_item_total,
    load_m5_raw,
    make_forecasts,
    save_calendar,
    to_contract,
)
from atlas.schema import Dataset, load_dataset, save_dataset

N_DAYS = 120
ITEMS = {"FOODS_1_001": "FOODS_1", "HOBBIES_1_002": "HOBBIES_1", "HOUSEHOLD_1_003": "HOUSEHOLD_1"}
STORES = ["CA_1", "TX_1"]


def write_fake_raw(raw_dir: Path) -> None:
    rng = np.random.default_rng(0)
    rows = []
    for i, (item, dept) in enumerate(ITEMS.items()):
        for store in STORES:
            base = 5 + 4 * i
            week = 1 + 0.5 * np.sin(2 * np.pi * np.arange(N_DAYS) / 7)
            sales = rng.poisson(base * week)
            rows.append(
                {"id": f"{item}_{store}_evaluation", "item_id": item, "dept_id": dept,
                 "cat_id": dept.split("_")[0], "store_id": store, "state_id": store[:2],
                 **{f"d_{d + 1}": int(sales[d]) for d in range(N_DAYS)}}
            )
    pd.DataFrame(rows).to_csv(raw_dir / "sales_train_evaluation.csv", index=False)

    dates = pd.date_range("2011-01-29", periods=N_DAYS, freq="D")
    cal = pd.DataFrame(
        {"date": dates.strftime("%Y-%m-%d"), "wm_yr_wk": 11101, "weekday": dates.day_name(),
         "d": [f"d_{d + 1}" for d in range(N_DAYS)],
         "event_name_1": [None] * N_DAYS, "event_type_1": [None] * N_DAYS}
    )
    cal.loc[10, ["event_name_1", "event_type_1"]] = ["SuperBowl", "Sporting"]
    cal.to_csv(raw_dir / "calendar.csv", index=False)


def test_m5_pipeline_to_contract(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    write_fake_raw(raw)

    sales, calendar = load_m5_raw(raw)
    assert {"item_id", "dept_id", "d_1", f"d_{N_DAYS}"} <= set(sales.columns)

    df, item_dept = aggregate_item_total(sales, calendar, n_items=3, history_days=N_DAYS)
    assert set(df.columns) == {"unique_id", "ds", "y"}
    assert df["unique_id"].nunique() == 3 and len(df) == 3 * N_DAYS
    # item total = sum over the two stores
    first = sales[sales["item_id"] == "FOODS_1_001"][["d_1"]].sum().iloc[0]
    got = df[(df["unique_id"] == "FOODS_1_001") & (df["ds"] == calendar["date"].iloc[0])]["y"].iloc[0]
    assert got == first
    assert item_dept["FOODS_1_001"] == "FOODS_1"
    # top-n keeps the best seller and last history_days days
    top1, _ = aggregate_item_total(sales, calendar, n_items=1, history_days=60)
    assert top1["unique_id"].unique().tolist() == ["HOUSEHOLD_1_003"] and len(top1) == 60

    cv = make_forecasts(df, h=7, step_size=7, n_windows=2, n_jobs=1)
    assert {"unique_id", "ds", "cutoff", "y", "SeasonalNaive", "AutoETS"} <= set(cv.columns)

    data, meta = to_contract(df, cv, item_dept)
    assert data["series_id"].nunique() == 6 and len(meta) == 6
    assert set(meta["series_id"]) == {f"{i}|{m}" for i in ITEMS for m in ("SeasonalNaive", "AutoETS")}
    assert meta.loc[meta["series_id"] == "FOODS_1_001|AutoETS", "group"].iloc[0] == "FOODS_1"
    assert meta["model_card"].iloc[0].endswith("rolling 28-day forecasts, refit every 28 days")
    assert (data.loc[data["is_forecast"], "predicted"] >= 0).all()

    out = tmp_path / "m5"
    save_dataset(Dataset(data=data, series_meta=meta), out)
    ds = load_dataset(out, min_history=56, window_len=7)
    assert ds.data["series_id"].nunique() == 6  # all series kept

    for _, s in ds.data.groupby("series_id"):
        assert s["is_forecast"].sum() == 14
        hist = s[~s["is_forecast"]]
        assert len(hist) == N_DAYS - 14
        assert hist["predicted"].isna().all()
        assert s[s["is_forecast"]]["predicted"].notna().all()
        assert s["ts"].is_monotonic_increasing

    path = save_calendar(calendar, out)
    cal = pd.read_parquet(path)
    assert list(cal.columns) == ["ts", "weekday", "event_name_1", "event_type_1"]
    assert len(cal) == N_DAYS and cal["event_name_1"].notna().sum() == 1
