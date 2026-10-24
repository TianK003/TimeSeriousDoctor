"""AppTest coverage for the Series tab, Evaluation tab and Atlas scatter (SPEC §7.12)."""

import base64
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from scripts.make_mock_run import main

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture(scope="module")
def mock_runs(tmp_path_factory) -> Path:
    runs = tmp_path_factory.mktemp("runs")
    main(["--out", str(runs / "mock")])
    return runs


def run_app(runs: Path, monkeypatch, **state) -> AppTest:
    monkeypatch.setenv("ATLAS_RUNS_DIR", str(runs))
    at = AppTest.from_file(APP, default_timeout=60)
    for k, v in state.items():
        at.session_state[k] = v
    return at.run()


def copy_run(mock_runs: Path, tmp_path: Path) -> Path:
    shutil.copytree(mock_runs / "mock", tmp_path / "mock")
    return tmp_path / "mock"


def figures(at: AppTest) -> list[dict]:
    return [json.loads(pc.proto.spec) for pc in at.get("plotly_chart")]


def arr(a) -> np.ndarray:
    """Plotly JSON stores numpy arrays as {dtype, bdata[, shape]}; plain lists pass through."""
    if isinstance(a, dict):
        out = np.frombuffer(base64.b64decode(a["bdata"]), dtype=np.dtype(a["dtype"]))
        return out.reshape([int(n) for n in a["shape"].split(",")]) if "shape" in a else out
    return np.asarray(a)


def metrics(at: AppTest) -> dict[str, tuple[str, str]]:
    return {m.label: (m.value, m.delta) for m in at.metric}


# -------------------------------------------------------------------- all tabs

def test_every_tab_renders(mock_runs, monkeypatch):
    at = run_app(mock_runs, monkeypatch)
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Atlas", "Cluster", "Series", "Evaluation"]
    # atlas scatter, model mix, series figure, comparison bars, confusion heatmap
    assert len(figures(at)) == 5
    assert not [i.value for i in at.info if "Coming soon" in i.value]


def test_atlas_scatter_below_cards(mock_runs, monkeypatch):
    at = run_app(mock_runs, monkeypatch)
    scatter = figures(at)[0]
    names = [t["name"] for t in scatter["data"]]
    assert set(names) == {"0", "1", "2", "-1"}  # one trace per cluster, no un-gated windows
    assert sum(len(arr(t["x"])) for t in scatter["data"]) == 44  # all anomalous windows


# ---------------------------------------------------------------------- Series

def test_series_selector_sorted_by_anomalous_windows(mock_runs, monkeypatch):
    at = run_app(mock_runs, monkeypatch)
    sel = at.selectbox(key="series_id")
    assert len(sel.options) == 20
    counts = [int(o.split("(")[1].split()[0]) for o in sel.options]
    assert counts == sorted(counts, reverse=True)
    assert sel.options[0].endswith(f"({counts[0]} anomalous)")
    assert sel.value == sel.options[0].split(" ")[0]  # defaults to the most anomalous series


def test_series_tab_shows_figure_and_window_table(mock_runs, monkeypatch):
    at = run_app(mock_runs, monkeypatch, series_id="S01")
    assert not at.exception
    assert at.selectbox(key="series_id").value == "S01"
    fig = figures(at)[2]
    assert fig["layout"]["title"]["text"] == "Series S01"
    table = at.dataframe[-1].value  # last dataframe on the page = series windows
    assert list(table.columns) == ["window_id", "start_ts", "cluster", "cluster name", "abs_err", "bias"]
    assert len(table) == 5 and set(table["window_id"].str.split("#").str[0]) == {"S01"}
    names = dict(zip(table["cluster"], table["cluster name"]))
    assert all(n == "not anomalous" for c, n in names.items() if c == -2)
    assert any(c >= 0 and n != "not anomalous" for c, n in names.items())


def test_show_in_series_tab_button(mock_runs, monkeypatch):
    at = run_app(mock_runs, monkeypatch)
    sid = "S07"
    at.selectbox(key="member_series_0").select(sid)
    at.button(key="show_series_0").click().run()
    assert not at.exception
    assert at.session_state["series_id"] == sid
    assert at.selectbox(key="series_id").value == sid
    assert figures(at)[2]["layout"]["title"]["text"] == f"Series {sid}"
    assert f"Series {sid} selected" in "\n".join(s.value for s in at.success)


def test_unknown_series_in_session_falls_back(mock_runs, monkeypatch):
    at = run_app(mock_runs, monkeypatch, series_id="not-a-series")
    assert not at.exception
    sel = at.selectbox(key="series_id")
    assert sel.value == sel.options[0].split(" ")[0]


# ------------------------------------------------------------------ Evaluation

def test_evaluation_headline(mock_runs, monkeypatch):
    at = run_app(mock_runs, monkeypatch)
    m = metrics(at)
    assert m["TimeSeriousDoctor accuracy"] == ("80%", "+35 points vs Gemma alone")
    assert m["Gemma alone accuracy"][0] == "45%"
    assert m["Gemma calls on the 40-series sample"][0] == "7 vs 40"
    assert m["Gemma calls on the full dataset"][0] == "7 vs 400"
    for label in ["Precision", "Recall", "F1", "ARI", "NMI", "Purity", "Noise share",
                  "Naming accuracy", "Series accuracy"]:
        assert label in m
    assert m["Purity"][0] == "80%" and m["ARI"][0] == "0.70"


def test_evaluation_charts(mock_runs, monkeypatch):
    bars, heat = figures(run_app(mock_runs, monkeypatch))[3:5]
    assert [t["name"] for t in bars["data"]] == ["TimeSeriousDoctor", "Gemma alone"]
    assert bars["data"][0]["x"][0] == "overall" and len(bars["data"][0]["x"]) == 5
    assert bars["data"][0]["y"][0] == pytest.approx(80) and bars["data"][1]["y"][0] == pytest.approx(45)
    assert heat["data"][0]["type"] == "heatmap"
    assert len(heat["data"][0]["y"]) == 4 and heat["data"][0]["y"][-1] == "Unclustered windows"
    assert arr(heat["data"][0]["z"]).sum() == 100  # 40+3+28+19+2+6+2


def test_evaluation_without_eval_json_shows_info(mock_runs, tmp_path, monkeypatch):
    run_dir = copy_run(mock_runs, tmp_path)
    (run_dir / "eval.json").unlink()
    at = run_app(tmp_path, monkeypatch)
    assert not at.exception
    assert any("No eval.json for this run" in i.value and "scripts/run_eval.py" in i.value for i in at.info)
    assert "TimeSeriousDoctor accuracy" not in metrics(at)
    assert len(figures(at)) == 3  # scatter, model mix, series figure


def test_evaluation_without_comparison(mock_runs, tmp_path, monkeypatch):
    run_dir = copy_run(mock_runs, tmp_path)
    ev = json.loads((run_dir / "eval.json").read_text(encoding="utf-8"))
    del ev["comparison"]
    (run_dir / "eval.json").write_text(json.dumps(ev), encoding="utf-8")
    at = run_app(tmp_path, monkeypatch)
    assert not at.exception
    m = metrics(at)
    assert "TimeSeriousDoctor accuracy" not in m and "Precision" in m
    assert len(figures(at)) == 4  # no comparison bars, heatmap still there


def test_evaluation_tolerates_missing_numbers(mock_runs, tmp_path, monkeypatch):
    run_dir = copy_run(mock_runs, tmp_path)
    ev = json.loads((run_dir / "eval.json").read_text(encoding="utf-8"))
    ev["gate"] = {}
    ev["naming_accuracy"] = None
    ev["comparison"]["gemma_alone"]["per_class"].pop("healthy")
    (run_dir / "eval.json").write_text(json.dumps(ev), encoding="utf-8")
    at = run_app(tmp_path, monkeypatch)
    assert not at.exception
    m = metrics(at)
    assert m["Precision"][0] == "n/a" and m["Naming accuracy"][0] == "n/a"
