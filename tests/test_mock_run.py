import json
import re
import shutil
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from atlas.schema import CLUSTER_NOISE, CLUSTER_NOT_GATED, list_runs, load_run
from scripts.make_mock_run import FEATURES, main

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture(scope="module")
def mock_runs(tmp_path_factory) -> Path:
    """A runs dir holding one mock run named 'mock'."""
    runs = tmp_path_factory.mktemp("runs")
    main(["--out", str(runs / "mock")])
    return runs


def run_app(runs: Path, monkeypatch, **state) -> AppTest:
    monkeypatch.setenv("ATLAS_RUNS_DIR", str(runs))
    at = AppTest.from_file(APP, default_timeout=60)
    for k, v in state.items():
        at.session_state[k] = v
    return at.run()


def all_text(at: AppTest) -> str:
    parts = [m.value for m in at.markdown] + [m.value for m in at.caption]
    parts += [w.value for w in at.warning] + [i.value for i in at.info] + [s.value for s in at.success]
    return "\n".join(parts)


# ------------------------------------------------------------------ mock run

def test_mock_run_is_loadable(mock_runs):
    assert [p.name for p in list_runs(mock_runs)] == ["mock"]
    run = load_run(mock_runs / "mock")

    assert run.data["series_id"].nunique() == 20
    per_series = run.data.groupby("series_id")["is_forecast"].agg(["sum", "count"])
    assert (per_series["sum"] == 84).all() and (per_series["count"] == 144).all()
    assert len(run.series_meta) == 20

    w = run.windows
    for col in ["window_id", "series_id", "start_idx", "end_idx", "start_ts", "end_ts", "true_label",
                "score_abs", "score_bias", "gated", "emb_row", "cluster", "x2d", "y2d", *FEATURES]:
        assert col in w.columns
    assert len(w) == 20 * 5
    assert set(w["cluster"]) == {0, 1, 2, CLUSTER_NOISE, CLUSTER_NOT_GATED}
    assert (w.loc[~w["gated"], "cluster"] == CLUSTER_NOT_GATED).all()
    assert w.loc[~w["gated"], ["x2d", "y2d"]].isna().all().all()
    assert (w.loc[~w["gated"], "emb_row"] == -1).all()

    assert run.embeddings.dtype == np.float32
    assert run.embeddings.shape[0] == int(w["gated"].sum())
    assert sorted(w.loc[w["gated"], "emb_row"]) == list(range(run.embeddings.shape[0]))

    ids = [c.cluster_id for c in run.clusters]
    assert ids == [0, 1, 2, CLUSTER_NOISE]
    sizes = [c.size for c in run.clusters if c.cluster_id >= 0]
    assert sizes == sorted(sizes, reverse=True)
    for c in run.clusters:
        assert c.size == int((w["cluster"] == c.cluster_id).sum())
        assert set(c.feature_median) == set(FEATURES)
    for c in run.clusters[:3]:
        assert c.report is not None and c.image_path == f"images/cluster_{c.cluster_id}.png"
        assert (run.path / c.image_path).stat().st_size > 0
    assert run.clusters[-1].report is None
    assert {c.report.failure_type for c in run.clusters[:3]} == {
        "delayed_response", "unadapted_shift", "over_smoothing"}
    assert run.summary is not None and len(run.summary.actions) == 3
    assert run.config["embedder"] == "features"


def test_mock_run_eval_has_comparison(mock_runs):
    ev = load_run(mock_runs / "mock").eval
    assert ev is not None
    assert set(ev) >= {"run", "embedder", "gate", "clustering", "naming_accuracy", "series_accuracy", "comparison"}
    assert set(ev["clustering"]) >= {"ari", "nmi", "purity", "noise_share", "confusion"}
    comp = ev["comparison"]
    assert comp["n_series"] == 40
    for side in ("pipeline", "gemma_alone"):
        assert set(comp[side]) == {"accuracy", "macro_f1", "per_class", "gemma_calls"}
    assert set(comp["gemma_calls_full_dataset"]) == {"pipeline", "gemma_alone"}


def test_mock_run_is_deterministic(mock_runs, tmp_path):
    main(["--out", str(tmp_path / "again")])
    assert load_run(mock_runs / "mock").windows.equals(load_run(tmp_path / "again").windows)


# ----------------------------------------------------------------------- app

def test_app_renders_mock_run(mock_runs, monkeypatch):
    at = run_app(mock_runs, monkeypatch)
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Atlas", "Cluster", "Series", "Evaluation"]
    assert [m.label for m in at.metric][:5] == ["Series", "Windows", "Anomalous", "Clusters", "Unclustered"]
    assert [m.value for m in at.metric][:5] == ["20", "100", "44", "3", "4"]
    assert {b.key for b in at.button} >= {"open_0", "open_1", "open_2", "open_-1"}
    assert "not guarantees" in all_text(at)
    assert at.sidebar.selectbox[0].value == "mock"
    assert len(at.dataframe) == 3  # feature table + member series of the default cluster + series windows


def test_open_button_selects_cluster(mock_runs, monkeypatch):
    at = run_app(mock_runs, monkeypatch)
    at.button(key="open_2").click().run()
    assert not at.exception
    assert at.session_state["cluster_id"] == 2
    assert at.selectbox(key="cluster_id").value == 2
    assert "Cluster 2 selected" in all_text(at)


def test_app_without_runs_shows_instructions(tmp_path, monkeypatch):
    at = run_app(tmp_path / "does_not_exist", monkeypatch)
    assert not at.exception
    sidebar_text = "\n".join(
        [w.value for w in at.sidebar.warning] + [m.value for m in at.sidebar.markdown])
    assert "No runs found" in sidebar_text and "scripts/make_mock_run.py" in sidebar_text
    assert len(at.tabs) == 0


def test_app_survives_corrupt_run(tmp_path, monkeypatch):
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "clusters.json").write_text("[]", encoding="utf-8")  # everything else missing
    at = run_app(tmp_path, monkeypatch)
    assert not at.exception
    assert any("Could not load run 'broken'" in e.value for e in at.error)


def test_cluster_without_report_shows_warning_and_missing_image(mock_runs, tmp_path, monkeypatch):
    run_dir = tmp_path / "mock"
    shutil.copytree(mock_runs / "mock", run_dir)
    clusters = json.loads((run_dir / "clusters.json").read_text(encoding="utf-8"))
    clusters[1]["report"] = None
    (run_dir / "clusters.json").write_text(json.dumps(clusters), encoding="utf-8")
    (run_dir / "images" / "cluster_1.png").unlink()

    at = run_app(tmp_path, monkeypatch, cluster_id=1)
    assert not at.exception
    warnings = [w.value for w in at.warning]
    assert any("Explanation unavailable" in w for w in warnings)
    assert any("Cluster image not found" in w for w in warnings)


def test_unclustered_bucket_has_manual_review_note(mock_runs, monkeypatch):
    at = run_app(mock_runs, monkeypatch, cluster_id=CLUSTER_NOISE)
    assert not at.exception
    assert any("review them manually" in i.value for i in at.info)


def test_gemma_text_is_escaped(mock_runs, tmp_path, monkeypatch):
    run_dir = tmp_path / "mock"
    shutil.copytree(mock_runs / "mock", run_dir)
    clusters = json.loads((run_dir / "clusters.json").read_text(encoding="utf-8"))
    clusters[0]["report"]["summary"] = "<script>alert(1)</script> costs $5 and $6 [x](http://evil.example)"
    clusters[0]["report"]["evidence"][0] = "<img src=x onerror=alert(1)>"
    (run_dir / "clusters.json").write_text(json.dumps(clusters), encoding="utf-8")

    at = run_app(tmp_path, monkeypatch, cluster_id=0)
    assert not at.exception
    text = all_text(at)
    assert not re.search(r"(?<!\\)<(script|img)", text)  # no unescaped tag reaches st.markdown
    assert r"\<script\>" in text and r"\$5" in text and r"\[x\]" in text
