"""Shared data contract: input dataset, run folder, and Gemma JSON models.

Engine, UI and Gemma code all agree on the names and types defined here
(SPEC sections 5, 6 and 7.9).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel

DATA_REQUIRED_COLUMNS = ["series_id", "ts", "actual", "predicted", "is_forecast"]  # optional: "true_label"
META_REQUIRED_COLUMNS = ["series_id", "model_card"]  # optional: "group"
WINDOW_COLUMNS = [
    "window_id", "series_id", "start_idx", "end_idx", "start_ts", "end_ts",
    "score_abs", "score_bias", "gated", "emb_row", "cluster", "x2d", "y2d",
]  # plus "true_label" on labeled runs, plus every feature column from atlas.features.FEATURES
CLUSTER_NOISE = -1  # gated but unclustered ("look manually")
CLUSTER_NOT_GATED = -2


# ---------------------------------------------------------------- Gemma JSON models

FailureType = Literal[
    "delayed_response", "missed_peak", "unadapted_shift",
    "over_smoothing", "healthy_noise", "other",
]


class Cause(BaseModel):
    cause: str
    why: str


class GemmaClusterReport(BaseModel):
    failure_type: FailureType
    name: str                    # short human name, <= 6 words
    summary: str                 # <= 2 sentences, plain language
    evidence: list[str]          # 2-4 items, each cites a provided feature value or visible pattern
    likely_causes: list[Cause]   # 1-3, most likely first
    suggested_fixes: list[str]   # 1-3 concrete experiments
    metric_to_monitor: str
    confidence: float            # 0-1


class GemmaBaselineAnswer(BaseModel):
    failure_type: FailureType
    confidence: float
    explanation: str


class AtlasSummary(BaseModel):
    headline: str
    summary: str                 # 3-5 sentences
    actions: list[str]           # top 3 recommended actions


# ---------------------------------------------------------------- run folder models

class ClusterInfo(BaseModel):
    cluster_id: int                 # 0..n-1 ordered by size desc; -1 = unclustered bucket
    size: int                       # number of windows
    n_series: int
    medoid_window_ids: list[str]    # up to 3
    feature_median: dict[str, float]
    feature_median_all_gated: dict[str, float]
    model_card_dist: dict[str, float]   # model_card -> share of windows
    context: dict[str, float] | None = None      # P2 calendar context
    image_path: str | None = None   # relative to the run folder, e.g. "images/cluster_0.png"
    report: GemmaClusterReport | None = None
    expert: dict | None = None                    # P2: {status, name, note}


# ---------------------------------------------------------------- input dataset

@dataclass
class Dataset:
    data: pd.DataFrame
    series_meta: pd.DataFrame


def _check_columns(df: pd.DataFrame, required: list[str], what: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{what} is missing required columns: {missing}")


def load_dataset(path: Path, min_history: int = 56, window_len: int = 28) -> Dataset:
    """Read, validate and clean a dataset folder (data.parquet + series_meta.parquet)."""
    path = Path(path)
    data = pd.read_parquet(path / "data.parquet")
    meta = pd.read_parquet(path / "series_meta.parquet")
    _check_columns(data, DATA_REQUIRED_COLUMNS, "data.parquet")
    _check_columns(meta, META_REQUIRED_COLUMNS, "series_meta.parquet")

    data["series_id"] = data["series_id"].astype(str)
    meta["series_id"] = meta["series_id"].astype(str)
    data["actual"] = data["actual"].astype(float)
    data["predicted"] = data["predicted"].astype(float)
    data["is_forecast"] = data["is_forecast"].astype(bool)
    data = data.sort_values(["series_id", "ts"]).reset_index(drop=True)

    n_forecast = data.groupby("series_id")["is_forecast"].sum()
    n_history = data.groupby("series_id")["is_forecast"].count() - n_forecast
    short_history = n_history < min_history
    short_forecast = (n_forecast < window_len) & ~short_history  # counted once, history first
    keep = ~(short_history | short_forecast)
    kept_ids = set(keep.index[keep])

    n_total = len(keep)
    print(
        f"load_dataset: kept {len(kept_ids)} of {n_total} series "
        f"(dropped {n_total - len(kept_ids)}: short history {int(short_history.sum())}, "
        f"short forecast {int(short_forecast.sum())})"
    )

    data = data[data["series_id"].isin(kept_ids)].reset_index(drop=True)
    meta = meta[meta["series_id"].isin(kept_ids)].reset_index(drop=True)
    return Dataset(data=data, series_meta=meta)


def save_dataset(ds: Dataset, path: Path) -> None:
    """Write a Dataset as data.parquet + series_meta.parquet."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    ds.data.to_parquet(path / "data.parquet", index=False)
    ds.series_meta.to_parquet(path / "series_meta.parquet", index=False)


# ---------------------------------------------------------------- run bundle

@dataclass
class RunResult:
    name: str
    config: dict                      # fully resolved Config.model_dump(mode="json")
    data: pd.DataFrame
    series_meta: pd.DataFrame
    windows: pd.DataFrame
    embeddings: np.ndarray            # float32 [n_gated, d]; row i = window with emb_row == i
    clusters: list[ClusterInfo]
    summary: AtlasSummary | None = None
    eval: dict | None = None
    expert_feedback: dict | None = None
    path: Path | None = None          # run folder; set by save_run/load_run

    def cluster(self, cluster_id: int) -> ClusterInfo:
        for c in self.clusters:
            if c.cluster_id == cluster_id:
                return c
        raise KeyError(cluster_id)


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_run(run: RunResult, path: Path) -> None:
    """Persist a RunResult as a run folder. Images are written by the pipeline, not here."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / "images").mkdir(exist_ok=True)
    (path / "gemma_raw").mkdir(exist_ok=True)

    _write_json(path / "config.json", run.config)
    run.data.to_parquet(path / "data.parquet", index=False)
    run.series_meta.to_parquet(path / "series_meta.parquet", index=False)
    run.windows.to_parquet(path / "windows.parquet", index=False)
    np.save(path / "embeddings.npy", np.asarray(run.embeddings, dtype=np.float32))
    _write_json(path / "clusters.json", [c.model_dump(mode="json") for c in run.clusters])
    if run.summary is not None:
        _write_json(path / "summary.json", run.summary.model_dump(mode="json"))
    if run.eval is not None:
        _write_json(path / "eval.json", run.eval)
    if run.expert_feedback is not None:
        _write_json(path / "expert_feedback.json", run.expert_feedback)
    run.path = path


def load_run(path: Path) -> RunResult:
    """Load a run folder written by save_run."""
    path = Path(path)
    emb_path = path / "embeddings.npy"
    embeddings = np.load(emb_path) if emb_path.exists() else np.zeros((0, 0), dtype=np.float32)

    def optional_json(name: str):
        p = path / name
        return _read_json(p) if p.exists() else None

    summary = optional_json("summary.json")
    return RunResult(
        name=path.name,
        config=_read_json(path / "config.json"),
        data=pd.read_parquet(path / "data.parquet"),
        series_meta=pd.read_parquet(path / "series_meta.parquet"),
        windows=pd.read_parquet(path / "windows.parquet"),
        embeddings=embeddings,
        clusters=[ClusterInfo.model_validate(c) for c in _read_json(path / "clusters.json")],
        summary=AtlasSummary.model_validate(summary) if summary is not None else None,
        eval=optional_json("eval.json"),
        expert_feedback=optional_json("expert_feedback.json"),
        path=path,
    )


def list_runs(runs_dir: Path) -> list[Path]:
    """Sorted run folders (those containing clusters.json) under runs_dir."""
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        return []
    return sorted(p for p in runs_dir.iterdir() if p.is_dir() and (p / "clusters.json").exists())
