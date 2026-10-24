"""M5 aggregation + statsforecast CV -> data contract.

Real Walmart data (SPEC section 7.2): sum each item over all stores, run two statsforecast models with
rolling-origin cross-validation, and write one series per (item, model) in the SPEC section 5 contract.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

ID_COLUMNS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
MODELS = ["SeasonalNaive", "AutoETS"]
CALENDAR_COLUMNS = ["ts", "weekday", "event_name_1", "event_type_1"]


def _day_columns(sales: pd.DataFrame) -> list[str]:
    """`d_1..d_N` columns in numeric order."""
    days = [c for c in sales.columns if c.startswith("d_")]
    return sorted(days, key=lambda c: int(c[2:]))


def load_m5_raw(raw_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read `sales_train_evaluation.csv` (wide, d_1..d_1941) and `calendar.csv` from a Kaggle-layout folder."""
    raw_dir = Path(raw_dir)
    sales_path, calendar_path = raw_dir / "sales_train_evaluation.csv", raw_dir / "calendar.csv"
    for p in (sales_path, calendar_path):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found (expected the Kaggle M5 files unzipped in {raw_dir})")
    header = pd.read_csv(sales_path, nrows=0).columns
    dtypes = {c: np.float32 for c in header if c.startswith("d_")}
    sales = pd.read_csv(sales_path, dtype=dtypes)
    calendar = pd.read_csv(calendar_path)
    calendar["date"] = pd.to_datetime(calendar["date"])
    return sales, calendar


def aggregate_item_total(
    sales: pd.DataFrame, calendar: pd.DataFrame, n_items: int = 1000, history_days: int = 730
) -> tuple[pd.DataFrame, pd.Series]:
    """Sum each item over all stores, keep the last `history_days` days and the `n_items` best sellers.

    Items are ranked by total sales inside the kept window (the period we forecast). Returns
    `(df, item_dept)`: `df` is long format with `unique_id` (item_id), `ds` (date), `y` (float);
    `item_dept` maps item_id -> dept_id (used for the `group` column).
    """
    days = _day_columns(sales)[-history_days:]
    by_item = sales.groupby("item_id", sort=True)[days].sum(numeric_only=True).astype(np.float64)
    top = by_item.sum(axis=1).sort_values(ascending=False, kind="stable").head(n_items).index
    by_item = by_item.loc[sorted(top)]

    dates = calendar.set_index("d")["date"].loc[days].to_numpy()
    df = pd.DataFrame(
        {
            "unique_id": np.repeat(by_item.index.to_numpy(), len(days)),
            "ds": np.tile(dates, len(by_item)),
            "y": by_item.to_numpy().ravel(),
        }
    )
    item_dept = sales.drop_duplicates("item_id").set_index("item_id")["dept_id"].loc[by_item.index]
    return df, item_dept


def make_forecasts(
    df: pd.DataFrame, h: int = 28, step_size: int = 28, n_windows: int = 6, n_jobs: int = -1
) -> pd.DataFrame:
    """Rolling-origin cross-validation of SeasonalNaive and AutoETS (weekly season, daily data)."""
    from statsforecast import StatsForecast
    from statsforecast.models import AutoETS, SeasonalNaive

    sf = StatsForecast(
        models=[SeasonalNaive(season_length=7), AutoETS(season_length=7)], freq="D", n_jobs=n_jobs
    )
    t0 = time.time()
    cv = sf.cross_validation(df=df, h=h, step_size=step_size, n_windows=n_windows)
    print(
        f"make_forecasts: {df['unique_id'].nunique()} items x {len(MODELS)} models x {n_windows} windows "
        f"in {time.time() - t0:.1f}s"
    )
    return cv.reset_index() if "unique_id" not in cv.columns else cv


def to_contract(df: pd.DataFrame, cv: pd.DataFrame, item_dept: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert CV output to the data contract: one series per (item, model).

    History rows (up to and including the first cutoff) have `predicted = NaN`, `is_forecast = False`;
    forecast rows carry the model's CV prediction (clipped at 0), `is_forecast = True`.
    """
    models = [c for c in cv.columns if c not in ("unique_id", "ds", "cutoff", "y")]
    first_cutoff = cv["cutoff"].min()
    # if windows overlap (step_size < h) keep the forecast from the most recent refit
    cv = cv.sort_values(["unique_id", "ds", "cutoff"]).drop_duplicates(["unique_id", "ds"], keep="last")
    history = df[df["ds"] <= first_cutoff]

    frames = []
    for model in models:
        hist = pd.DataFrame(
            {
                "series_id": history["unique_id"].astype(str) + "|" + model,
                "ts": history["ds"].to_numpy(),
                "actual": history["y"].astype(float).to_numpy(),
                "predicted": np.nan,
                "is_forecast": False,
            }
        )
        fc = pd.DataFrame(
            {
                "series_id": cv["unique_id"].astype(str) + "|" + model,
                "ts": cv["ds"].to_numpy(),
                "actual": cv["y"].astype(float).to_numpy(),
                "predicted": np.clip(cv[model].astype(float).to_numpy(), 0.0, None),
                "is_forecast": True,
            }
        )
        frames += [hist, fc]
    data = pd.concat(frames, ignore_index=True).sort_values(["series_id", "ts"]).reset_index(drop=True)

    series_ids = data["series_id"].drop_duplicates()
    items = series_ids.str.rsplit("|", n=1).str[0]
    model_names = series_ids.str.rsplit("|", n=1).str[1]
    meta = pd.DataFrame(
        {
            "series_id": series_ids.to_numpy(),
            "model_card": [f"{m}, rolling 28-day forecasts, refit every 28 days" for m in model_names],
            "group": items.map(item_dept).astype(str).to_numpy(),
        }
    )
    return data, meta


def save_calendar(calendar: pd.DataFrame, out_dir: Path) -> Path:
    """Write `out_dir/calendar.parquet` with `ts, weekday, event_name_1, event_type_1` (for calendar context)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cal = pd.DataFrame(
        {
            "ts": pd.to_datetime(calendar["date"]),
            "weekday": calendar["weekday"].astype(str),
            "event_name_1": calendar["event_name_1"].astype(object),
            "event_type_1": calendar["event_type_1"].astype(object),
        }
    )[CALENDAR_COLUMNS]
    path = out_dir / "calendar.parquet"
    cal.to_parquet(path, index=False)
    return path
