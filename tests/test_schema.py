import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from atlas.schema import (
    AtlasSummary,
    Cause,
    ClusterInfo,
    Dataset,
    GemmaClusterReport,
    RunResult,
    list_runs,
    load_dataset,
    load_run,
    save_dataset,
    save_run,
)


def make_series(series_id: str, n_history: int, n_forecast: int) -> pd.DataFrame:
    n = n_history + n_forecast
    actual = np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "series_id": series_id,
            "ts": pd.date_range("2024-01-01", periods=n, freq="D"),
            "actual": actual,
            "predicted": np.where(np.arange(n) >= n_history, actual + 1.0, np.nan),
            "is_forecast": np.arange(n) >= n_history,
        }
    )


def make_dataset(specs: dict[str, tuple[int, int]]) -> Dataset:
    data = pd.concat([make_series(sid, h, f) for sid, (h, f) in specs.items()], ignore_index=True)
    meta = pd.DataFrame({"series_id": list(specs), "model_card": "AutoETS, weekly season"})
    return Dataset(data=data, series_meta=meta)


def make_report() -> GemmaClusterReport:
    return GemmaClusterReport(
        failure_type="delayed_response",
        name="Lagging after level shift",
        summary="Forecast reacts a few days late.",
        evidence=["lag_best = 3", "bias_run_len high"],
        likely_causes=[Cause(cause="Slow smoothing", why="Alpha is too small")],
        suggested_fixes=["Refit with a higher alpha"],
        metric_to_monitor="lag_best",
        confidence=0.8,
    )


def make_run() -> RunResult:
    ds = make_dataset({"a": (60, 30), "b": (60, 30)})
    windows = pd.DataFrame(
        {
            "window_id": ["a#0", "a#2", "b#0"],
            "series_id": ["a", "a", "b"],
            "start_idx": [0, 2, 0],
            "end_idx": [28, 30, 28],
            "start_ts": pd.to_datetime(["2024-03-01", "2024-03-03", "2024-03-01"]),
            "end_ts": pd.to_datetime(["2024-03-29", "2024-03-31", "2024-03-29"]),
            "score_abs": [1.5, 0.4, 2.0],
            "score_bias": [0.9, 0.1, 1.2],
            "gated": [True, False, True],
            "emb_row": [0, -1, 1],
            "cluster": [0, -2, -1],
            "x2d": [0.1, np.nan, 0.7],
            "y2d": [0.2, np.nan, 0.8],
        }
    )
    embeddings = np.arange(12, dtype=np.float32).reshape(3, 4)
    clusters = [
        ClusterInfo(
            cluster_id=0,
            size=1,
            n_series=1,
            medoid_window_ids=["a#0"],
            feature_median={"lag_best": 3.0},
            feature_median_all_gated={"lag_best": 1.0},
            model_card_dist={"AutoETS, weekly season": 1.0},
            image_path="images/cluster_0.png",
            report=make_report(),
        ),
        ClusterInfo(
            cluster_id=-1,
            size=1,
            n_series=1,
            medoid_window_ids=["b#0"],
            feature_median={"lag_best": 0.0},
            feature_median_all_gated={"lag_best": 1.0},
            model_card_dist={"AutoETS, weekly season": 1.0},
            report=None,
        ),
    ]
    summary = AtlasSummary(headline="One lagging cluster", summary="Details.", actions=["Refit"])
    return RunResult(
        name="ignored",
        config={"window_len": 28},
        data=ds.data,
        series_meta=ds.series_meta,
        windows=windows,
        embeddings=embeddings,
        clusters=clusters,
        summary=summary,
    )


def test_save_load_run_roundtrip(tmp_path):
    run = make_run()
    folder = tmp_path / "runs" / "demo"
    save_run(run, folder)
    assert run.path == folder
    assert (folder / "images").is_dir()
    assert (folder / "gemma_raw").is_dir()
    assert not (folder / "eval.json").exists()
    assert not (folder / "expert_feedback.json").exists()

    loaded = load_run(folder)
    assert loaded.name == "demo"
    assert loaded.path == folder
    assert loaded.config == run.config
    pd.testing.assert_frame_equal(loaded.data, run.data)
    pd.testing.assert_frame_equal(loaded.series_meta, run.series_meta)
    pd.testing.assert_frame_equal(loaded.windows, run.windows)
    assert loaded.embeddings.dtype == np.float32
    np.testing.assert_array_equal(loaded.embeddings, run.embeddings)
    assert loaded.clusters == run.clusters
    assert loaded.clusters[0].report == make_report()
    assert loaded.clusters[1].report is None
    assert loaded.summary == run.summary
    assert loaded.eval is None
    assert loaded.expert_feedback is None


def test_run_cluster_lookup():
    run = make_run()
    assert run.cluster(0).size == 1
    assert run.cluster(-1).cluster_id == -1
    with pytest.raises(KeyError):
        run.cluster(7)


def test_load_run_without_optional_files(tmp_path):
    run = make_run()
    run.summary = None
    folder = tmp_path / "demo"
    save_run(run, folder)
    (folder / "embeddings.npy").unlink()
    loaded = load_run(folder)
    assert loaded.summary is None
    assert loaded.embeddings.shape == (0, 0)
    assert loaded.embeddings.dtype == np.float32


def test_optional_files_written_when_set(tmp_path):
    run = make_run()
    run.eval = {"purity": 0.9}
    run.expert_feedback = {"0": {"status": "confirmed"}}
    folder = tmp_path / "demo"
    save_run(run, folder)
    loaded = load_run(folder)
    assert loaded.eval == {"purity": 0.9}
    assert loaded.expert_feedback == {"0": {"status": "confirmed"}}


def test_list_runs(tmp_path):
    runs_dir = tmp_path / "runs"
    assert list_runs(runs_dir) == []
    save_run(make_run(), runs_dir / "b_run")
    save_run(make_run(), runs_dir / "a_run")
    (runs_dir / "not_a_run").mkdir()
    assert list_runs(runs_dir) == [runs_dir / "a_run", runs_dir / "b_run"]


def test_load_dataset_drops_short_series(tmp_path, capsys):
    ds = make_dataset({"good": (60, 30), "short_hist": (10, 30), "short_fc": (60, 5)})
    save_dataset(ds, tmp_path / "ds")
    loaded = load_dataset(tmp_path / "ds", min_history=56, window_len=28)
    assert set(loaded.data["series_id"]) == {"good"}
    assert list(loaded.series_meta["series_id"]) == ["good"]
    out = capsys.readouterr().out
    assert "kept 1 of 3 series" in out
    assert "short history 1" in out
    assert "short forecast 1" in out


def test_load_dataset_casts_and_sorts(tmp_path):
    ds = make_dataset({"b": (60, 30), "a": (60, 30)})
    ds.data = ds.data.sample(frac=1.0, random_state=0).reset_index(drop=True)
    ds.data["is_forecast"] = ds.data["is_forecast"].astype(int)
    save_dataset(ds, tmp_path / "ds")
    loaded = load_dataset(tmp_path / "ds")
    assert loaded.data["is_forecast"].dtype == bool
    assert loaded.data["actual"].dtype == float
    assert list(loaded.data["series_id"].unique()) == ["a", "b"]
    for _, g in loaded.data.groupby("series_id"):
        assert g["ts"].is_monotonic_increasing


def test_load_dataset_missing_column_raises(tmp_path):
    ds = make_dataset({"good": (60, 30)})
    ds.data = ds.data.drop(columns=["actual"])
    save_dataset(ds, tmp_path / "ds")
    with pytest.raises(ValueError, match="actual"):
        load_dataset(tmp_path / "ds")


def test_load_dataset_missing_meta_column_raises(tmp_path):
    ds = make_dataset({"good": (60, 30)})
    ds.series_meta = ds.series_meta.drop(columns=["model_card"])
    save_dataset(ds, tmp_path / "ds")
    with pytest.raises(ValueError, match="model_card"):
        load_dataset(tmp_path / "ds")


def test_report_rejects_unknown_failure_type():
    payload = make_report().model_dump()
    payload["failure_type"] = "bogus"
    with pytest.raises(ValidationError):
        GemmaClusterReport.model_validate(payload)
