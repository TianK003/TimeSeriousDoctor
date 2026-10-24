"""Write the synthetic dataset + preview.png.

Owner: C. Task: P0.4. Spec: SPEC.md §7.1.

    python scripts/make_synthetic.py --n 400 --out data/synthetic [--seed 0]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from anywhere without installing atlas

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from atlas.data.synthetic import LABELS, make_synthetic  # noqa: E402
from atlas.schema import Dataset, save_dataset  # noqa: E402

CONTEXT_DAYS = 60  # history days drawn before the forecast region in the preview


def series_class(data: pd.DataFrame) -> pd.Series:
    """Per-series true type: the failure label if the series has one, else healthy."""
    fc = data[data["is_forecast"]]
    failing = fc[~fc["true_label"].isin(["healthy", "history"])].groupby("series_id")["true_label"].first()
    types = pd.Series("healthy", index=data["series_id"].unique(), name="true_type")
    types.loc[failing.index] = failing
    return types


def write_preview(data: pd.DataFrame, meta: pd.DataFrame, path: Path) -> None:
    classes = series_class(data)
    cards = meta.set_index("series_id")["model_card"]
    fig, axes = plt.subplots(len(LABELS), 1, figsize=(14, 12), dpi=100)
    for ax, label in zip(axes, LABELS):
        sid = classes.index[classes == label][0]
        s = data[data["series_id"] == sid].reset_index(drop=True)
        first_fc = int(s["is_forecast"].idxmax())
        s = s.iloc[max(first_fc - CONTEXT_DAYS, 0):]
        ax.plot(s["ts"], s["actual"], color="black", lw=1.2, label="actual")
        ax.plot(s["ts"], s["predicted"], color="red", lw=1.2, label="predicted")
        fc = s[s["is_forecast"]]
        ax.axvspan(fc["ts"].iloc[0], fc["ts"].iloc[-1], color="tab:blue", alpha=0.08, lw=0)
        seg = s[s["true_label"] == label] if label != "healthy" else s.iloc[0:0]
        if len(seg):
            ax.axvspan(seg["ts"].iloc[0], seg["ts"].iloc[-1], color="gold", alpha=0.35, lw=0, label="failure segment")
        ax.set_title(f"{label}  ({sid}, {cards[sid]})", loc="left", fontsize=11)
        ax.margins(x=0.005)
        ax.legend(loc="upper left", ncol=3, fontsize=8, frameon=False)
    fig.suptitle("Synthetic examples: blue = forecast region, yellow = injected failure segment", fontsize=12)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=400, help="number of series")
    ap.add_argument("--out", type=Path, default=Path("data/synthetic"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data, meta = make_synthetic(n_series=args.n, seed=args.seed)
    save_dataset(Dataset(data=data, series_meta=meta), args.out)
    write_preview(data, meta, args.out / "preview.png")

    print(f"wrote {len(meta)} series ({len(data)} rows) to {args.out}")
    print("series per class:")
    for label, count in series_class(data).value_counts().reindex(LABELS).items():
        print(f"  {label:18s} {count}")
    print("forecast rows per label:")
    for label, count in data.loc[data["is_forecast"], "true_label"].value_counts().items():
        print(f"  {label:18s} {count}")


if __name__ == "__main__":
    main()
