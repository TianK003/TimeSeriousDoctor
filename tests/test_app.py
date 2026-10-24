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


# ------------------------------------------------------------- expert review (P2.2)

def markdown_text(at: AppTest) -> str:
    return "\n".join(m.value for m in at.markdown)


def feedback(run_dir: Path) -> dict:
    return json.loads((run_dir / "expert_feedback.json").read_text(encoding="utf-8"))


def test_expert_box_defaults(mock_runs, monkeypatch):
    at = run_app(mock_runs, monkeypatch, cluster_id=0)
    assert not at.exception
    radio = at.radio(key="expert_verdict_0")
    assert radio.options == ["Confirm", "Reject", "Rename"] and radio.value == "Confirm"
    assert not [t for t in at.text_input if t.key == "expert_name_0"]  # name input only for Rename
    types = at.selectbox(key="expert_type_0")
    assert {"delayed_response", "missed_peak", "healthy_noise", "other"} <= set(types.options)
    assert types.value == "delayed_response"  # Gemma's failure type is the default
    assert "Expert review" in markdown_text(at)
    assert not at.button(key="expert_save_0").disabled
    assert "Expert-confirmed" not in markdown_text(at)  # nothing saved yet


def test_no_expert_box_for_unclustered_bucket(mock_runs, monkeypatch):
    at = run_app(mock_runs, monkeypatch, cluster_id=-1)
    assert not at.exception
    assert not [r for r in at.radio if r.key and r.key.startswith("expert_verdict")]


def test_save_confirm_writes_feedback_and_shows_badges(mock_runs, tmp_path, monkeypatch):
    run_dir = copy_run(mock_runs, tmp_path)
    at = run_app(tmp_path, monkeypatch, cluster_id=0)
    at.text_area(key="expert_note_0").set_value("Matches what we see on the promo series.")
    at.button(key="expert_save_0").click().run()
    assert not at.exception

    entry = feedback(run_dir)["0"]
    assert entry["status"] == "confirmed" and entry["failure_type"] == "delayed_response"
    assert entry["name"] == entry["gemma_name"] == "Forecast lags actuals by days"
    assert entry["note"] == "Matches what we see on the promo series."
    assert entry["summary"].startswith("The forecast follows the actuals") and entry["feature_median"]["best_lag"] > 0
    assert any("Saved" in s.value for s in at.success)
    # Cluster-tab header badge plus the Atlas card badge
    assert markdown_text(at).count("✔ Expert-confirmed") == 2


def test_save_rename_changes_card_title_and_header(mock_runs, tmp_path, monkeypatch):
    run_dir = copy_run(mock_runs, tmp_path)
    at = run_app(tmp_path, monkeypatch, cluster_id=1)
    at.radio(key="expert_verdict_1").set_value("Rename").run()
    assert at.button(key="expert_save_1").disabled  # a rename needs a name
    at.text_input(key="expert_name_1").set_value("Stale baseline after refit").run()
    at.selectbox(key="expert_type_1").select("missed_peak")
    at.button(key="expert_save_1").click().run()
    assert not at.exception

    entry = feedback(run_dir)["1"]
    assert (entry["status"], entry["name"], entry["failure_type"]) == ("renamed", "Stale baseline after refit", "missed_peak")
    assert entry["gemma_name"] == "Stuck at the old level"
    text = markdown_text(at).replace("\\", "")  # drop md_escape backslashes
    caption = "\n".join(c.value for c in at.caption).replace("\\", "")
    assert "#### Stale baseline after refit" in text  # Atlas card title is the expert name
    assert "Gemma: Stuck at the old level" in caption  # Gemma name as a small caption under it
    assert "Gemma's name: Stuck at the old level" in caption  # and in the Cluster tab
    assert "✎ Renamed by expert: Stale baseline after refit" in text
    assert [h.value for h in at.header if "Stale baseline" in h.value]
    assert "1: Stale baseline after refit" in at.selectbox(key="cluster_id").options


def test_save_reject_keeps_gemma_diagnosis_and_updates_existing_file(mock_runs, tmp_path, monkeypatch):
    run_dir = copy_run(mock_runs, tmp_path)
    other = {"status": "confirmed", "name": "Kept", "failure_type": "other", "note": "", "gemma_name": "g",
             "summary": "", "feature_median": {}}
    (run_dir / "expert_feedback.json").write_text(json.dumps({"2": other}), encoding="utf-8")
    at = run_app(tmp_path, monkeypatch, cluster_id=0)
    assert "✔ Expert-confirmed" in markdown_text(at)  # existing verdict for cluster 2 shows on its card
    at.radio(key="expert_verdict_0").set_value("Reject").run()
    assert not [s for s in at.selectbox if s.key == "expert_type_0"]  # no type choice for a rejection
    at.button(key="expert_save_0").click().run()
    assert not at.exception

    saved = feedback(run_dir)
    assert saved["2"] == other  # untouched
    assert saved["0"]["status"] == "rejected" and saved["0"]["failure_type"] == "delayed_response"
    assert "✖ Rejected by expert" in markdown_text(at)


def test_saved_verdict_is_prefilled_when_reopening(mock_runs, tmp_path, monkeypatch):
    run_dir = copy_run(mock_runs, tmp_path)
    saved = {"status": "renamed", "name": "Promo lag", "failure_type": "missed_peak", "note": "see wiki",
             "gemma_name": "x", "summary": "", "feature_median": {}}
    (run_dir / "expert_feedback.json").write_text(json.dumps({"0": saved}), encoding="utf-8")
    at = run_app(tmp_path, monkeypatch, cluster_id=0)
    assert at.radio(key="expert_verdict_0").value == "Rename"
    assert at.text_input(key="expert_name_0").value == "Promo lag"
    assert at.selectbox(key="expert_type_0").value == "missed_peak"
    assert at.text_area(key="expert_note_0").value == "see wiki"
