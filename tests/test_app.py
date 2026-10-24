"""AppTest coverage for the app tabs, expert review, latent-space map, dark mode and run selection (SPEC §7.12)."""

import base64
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from atlas.chat import ChatAnswer
from atlas.prompts import CHAT_SYSTEM
from atlas.schema import load_run
from scripts.make_mock_run import main

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.fixture(autouse=True)
def no_api_key(monkeypatch):
    """No test sees a real key (a .env above the repo would be found) or reaches the network, unless it installs a fake."""
    monkeypatch.setattr("atlas.config.get_api_key", lambda: None)
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()  # the chat client is cached across AppTest runs


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
    assert [t.label for t in at.tabs] == ["Atlas", "Latent space", "Cluster", "Series", "Evaluation", "Ask the Doctor"]
    # figure order: atlas scatter, latent map, model mix, series figure, series latent map, bars, heatmap
    assert len(figures(at)) == 7
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
    fig = figures(at)[3]
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
    assert figures(at)[3]["layout"]["title"]["text"] == f"Series {sid}"
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


def test_evaluation_shows_failed_call_counts_only_when_there_are_some(mock_runs, tmp_path, monkeypatch):
    assert not [c for c in run_app(mock_runs, monkeypatch).caption if "failed" in c.value or "no Gemma report" in c.value]

    run_dir = copy_run(mock_runs, tmp_path)
    ev = json.loads((run_dir / "eval.json").read_text(encoding="utf-8"))
    ev["comparison"]["gemma_alone"]["n_failed"] = 3
    ev["comparison"]["pipeline"]["n_clusters_without_report"] = 1
    (run_dir / "eval.json").write_text(json.dumps(ev), encoding="utf-8")
    captions = [c.value for c in run_app(tmp_path, monkeypatch).caption]
    assert any("3 of 40 Gemma-alone calls failed" in c and "scored as wrong" in c for c in captions)
    assert any("1 pipeline cluster(s) have no Gemma report" in c for c in captions)


def test_evaluation_charts(mock_runs, monkeypatch):
    bars, heat = figures(run_app(mock_runs, monkeypatch))[5:7]
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
    assert len(figures(at)) == 5  # scatter, latent map, model mix, series figure, series latent map


def test_evaluation_without_comparison(mock_runs, tmp_path, monkeypatch):
    run_dir = copy_run(mock_runs, tmp_path)
    ev = json.loads((run_dir / "eval.json").read_text(encoding="utf-8"))
    del ev["comparison"]
    (run_dir / "eval.json").write_text(json.dumps(ev), encoding="utf-8")
    at = run_app(tmp_path, monkeypatch)
    assert not at.exception
    m = metrics(at)
    assert "TimeSeriousDoctor accuracy" not in m and "Precision" in m
    assert len(figures(at)) == 6  # no comparison bars, heatmap still there


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


# ------------------------------------------------- latent space, theme, loader cache, default run

def trace_names(fig: dict) -> list[str]:
    return [t["name"] for t in fig["data"]]


def test_latent_tab_renders_with_modes_and_controls(mock_runs, monkeypatch):
    at = run_app(mock_runs, monkeypatch)
    assert not at.exception
    assert at.radio(key="latent_mode").options == ["All windows (feature space)", "Anomalous windows (cluster embedding)"]
    assert at.selectbox(key="latent_color").options == ["Cluster", "True label (synthetic data)", "Series"]
    highlight = at.selectbox(key="latent_highlight")
    assert highlight.options[0] == "(none)" and len(highlight.options) == 21
    latent = figures(at)[1]
    names = trace_names(latent)
    assert "normal (not anomalous)" in names and "0: Forecast lags actuals by days" in names
    assert sum(len(t["customdata"]) for t in latent["data"]) == 100  # every window; customdata = window_id
    assert any("Click a point on the map" in i.value for i in at.info)


def test_latent_modes_color_and_highlight(mock_runs, monkeypatch):
    at = run_app(mock_runs, monkeypatch)
    at.radio(key="latent_mode").set_value("Anomalous windows (cluster embedding)").run()
    assert not at.exception
    anomalous = figures(at)[1]
    assert "normal (not anomalous)" not in trace_names(anomalous)
    assert sum(len(t["customdata"]) for t in anomalous["data"]) == 44  # only the gated windows

    at.selectbox(key="latent_color").select("true_label").run()
    assert not at.exception
    assert "delayed_response" in trace_names(figures(at)[1])

    at.selectbox(key="latent_highlight").select("S01").run()
    assert not at.exception
    assert trace_names(figures(at)[1])[-1] == "selected series"


def test_series_tab_has_small_latent_map_highlighting_the_series(mock_runs, monkeypatch):
    at = run_app(mock_runs, monkeypatch, series_id="S05")
    small = figures(at)[4]
    assert trace_names(small)[-1] == "selected series"
    assert small["layout"]["showlegend"] is False and small["layout"]["height"] == 340


def test_clicking_a_point_shows_window_and_cluster_explanation(mock_runs, monkeypatch):
    run = load_run(mock_runs / "mock")
    wid = run.windows.loc[run.windows["cluster"] == 0, "window_id"].iloc[0]
    sid = wid.split("#")[0]
    selection = {"selection": {"points": [{"customdata": wid, "point_index": 0}], "point_indices": [0],
                               "box": [], "lasso": []}}
    at = run_app(mock_runs, monkeypatch, latent_plot=selection)
    assert not at.exception
    assert len(figures(at)) == 8  # + the window figure beside the map
    assert any(fig["layout"]["title"]["text"].startswith(wid) for fig in figures(at))
    assert not any("Click a point on the map" in i.value for i in at.info)
    assert "delayed response" in markdown_text(at)  # the cluster's failure type badge
    assert "The forecast follows the actuals with a delay" in markdown_text(at)  # Gemma's summary
    at.button(key="latent_show_series").click().run()
    assert not at.exception and at.session_state["series_id"] == sid
    at.button(key="latent_open_cluster").click().run()
    assert at.session_state["cluster_id"] == 0


def test_unknown_selection_is_ignored(mock_runs, monkeypatch):
    selection = {"selection": {"points": [{"customdata": "nope#1"}], "point_indices": [], "box": [], "lasso": []}}
    at = run_app(mock_runs, monkeypatch, latent_plot=selection)
    assert not at.exception
    assert len(figures(at)) == 7


def test_dark_mode_toggle_switches_templates_and_injects_css(mock_runs, monkeypatch):
    light = run_app(mock_runs, monkeypatch)
    assert light.sidebar.toggle(key="dark_mode").label == "🌙 Dark mode"
    assert not any("<style>" in m.value for m in light.markdown)
    assert all(pc.proto.theme == "" for pc in light.get("plotly_chart"))  # theme=None keeps our plotly template

    at = run_app(mock_runs, monkeypatch)
    at.sidebar.toggle(key="dark_mode").set_value(True).run()
    assert not at.exception
    assert at.sidebar.toggle(key="dark_mode").label == "☀️ Light mode"
    assert any("<style>" in m.value for m in at.markdown)
    assert len(figures(at)) == 7

    def backgrounds(a: AppTest) -> set[str]:
        return {f["layout"]["template"]["layout"]["paper_bgcolor"] for f in figures(a)}

    assert len(backgrounds(light)) == 1 and len(backgrounds(at)) == 1  # one template for every figure
    assert backgrounds(light) != backgrounds(at)


def test_loader_cache_picks_up_files_written_after_first_load(mock_runs, tmp_path, monkeypatch):
    run_dir = copy_run(mock_runs, tmp_path)
    (run_dir / "eval.json").unlink()
    at = run_app(tmp_path, monkeypatch)
    assert any("No eval.json" in i.value for i in at.info) and "TimeSeriousDoctor accuracy" not in metrics(at)

    shutil.copy(mock_runs / "mock" / "eval.json", run_dir / "eval.json")
    verdict = {"status": "confirmed", "name": "x", "failure_type": "other", "note": "", "gemma_name": "x",
               "summary": "", "feature_median": {}}
    (run_dir / "expert_feedback.json").write_text(json.dumps({"0": verdict}), encoding="utf-8")
    newest = max(p.stat().st_mtime for p in run_dir.iterdir() if p.is_file())
    for name in ("eval.json", "expert_feedback.json"):
        os.utime(run_dir / name, (newest + 10, newest + 10))  # clusters.json keeps its old mtime
    at.run()
    assert not at.exception
    assert metrics(at)["TimeSeriousDoctor accuracy"][0] == "80%"
    assert "✔ Expert-confirmed" in markdown_text(at)


def test_tsne_is_not_recomputed_by_expert_saves_or_eval_writes(mock_runs, tmp_path, monkeypatch):
    import atlas.cluster

    run_dir = copy_run(mock_runs, tmp_path)
    calls = []
    real = atlas.cluster.project_tsne
    monkeypatch.setattr(atlas.cluster, "project_tsne", lambda *a, **kw: calls.append(1) or real(*a, **kw))

    at = run_app(tmp_path, monkeypatch, cluster_id=0)
    assert not at.exception
    n_first = len(calls)
    assert n_first >= 1

    at.button(key="expert_save_0").click().run()  # writes expert_feedback.json into the run folder
    (run_dir / "eval.json").write_text((run_dir / "eval.json").read_text(encoding="utf-8"), encoding="utf-8")
    at.run()
    assert not at.exception and (run_dir / "expert_feedback.json").exists()
    assert len(calls) == n_first  # same windows.parquet / embeddings.npy: cached map is reused

    newer = time.time() + 100
    os.utime(run_dir / "windows.parquet", (newer, newer))  # the t-SNE inputs changed: recompute
    at.run()
    assert len(calls) > n_first


def make_runs(tmp_path: Path, mock_runs: Path, ages: dict[str, float]) -> Path:
    """A runs dir with copies of the mock run; every file of run `name` gets mtime = now - ages[name] seconds."""
    runs = tmp_path / "runs"
    for name, age in ages.items():
        shutil.copytree(mock_runs / "mock", runs / name)
        t = time.time() - age
        for p in (runs / name).iterdir():
            if p.is_file():
                os.utime(p, (t, t))
    return runs


def test_default_run_prefers_newest_non_mock_run(mock_runs, tmp_path, monkeypatch):
    runs = make_runs(tmp_path, mock_runs, {"mock": 0, "old_run": 5000, "new_run": 1000})  # mock is the newest
    at = run_app(runs, monkeypatch)
    assert not at.exception
    sel = at.sidebar.selectbox[0]
    assert sel.value == "new_run"
    assert sel.options == ["mock (demo data)", "new_run", "old_run"]  # mock stays selectable, labeled

    sel.select("mock").run()  # an explicit choice sticks across reruns
    assert at.sidebar.selectbox[0].value == "mock" and not at.exception


def test_default_run_is_mock_when_it_is_the_only_one(mock_runs, tmp_path, monkeypatch):
    at = run_app(make_runs(tmp_path, mock_runs, {"mock": 0}), monkeypatch)
    assert at.sidebar.selectbox[0].value == "mock" and at.sidebar.selectbox[0].options == ["mock (demo data)"]


# ------------------------------------------------------------------ Ask the Doctor

class FakeGemma:
    """Stands in for atlas.gemma.GemmaClient: records every call, returns `result` (a ChatAnswer, or None = failed call)."""

    calls: list[dict] = []
    result: ChatAnswer | None = None

    def __init__(self, cfg, api_key=None, raw_dir=None, **kwargs):
        self.cfg, self.raw_dir = cfg, raw_dir

    def generate(self, system, user, schema, image=None, tag="call"):
        FakeGemma.calls.append({"system": system, "user": user, "schema": schema, "tag": tag, "raw_dir": self.raw_dir, "model": self.cfg.model})
        return FakeGemma.result


@pytest.fixture()
def fake_gemma(monkeypatch):
    import atlas.gemma

    FakeGemma.calls = []
    FakeGemma.result = ChatAnswer(
        answer="The forecast trails the actuals by about three steps.",
        evidence=["best_lag 3.00 vs 0.00 for all anomalous windows", "lag_gain 0.57"],
        suggested_next_step="Add leading indicators and compare MASE on these windows.",
        confidence=0.82,
    )
    monkeypatch.setattr("atlas.config.get_api_key", lambda: "test-key")
    monkeypatch.setattr(atlas.gemma, "GemmaClient", FakeGemma)
    return FakeGemma


def chat_text(at: AppTest) -> str:
    """Captions and markdown of every chat message, with the md_escape backslashes dropped."""
    return "\n".join(m.value for msg in at.chat_message for m in [*msg.caption, *msg.markdown]).replace("\\", "")


def test_ask_the_doctor_without_key_shows_info_and_disables_input(mock_runs, monkeypatch):
    at = run_app(mock_runs, monkeypatch)
    assert not at.exception
    assert any(i.value.startswith("Ask the Doctor needs a Gemma API key (GEMMA4_API_KEY in .env)") for i in at.info)
    assert at.chat_input[0].disabled
    assert [at.button(key=f"chat_ex_{i}").disabled for i in range(3)] == [True, True, True]
    assert at.radio(key="chat_scope").options == ["Whole run", "A cluster", "A series"]
    assert len(at.chat_message) == 0


def test_ask_the_doctor_renders_without_calling_gemma(mock_runs, monkeypatch, fake_gemma):
    at = run_app(mock_runs, monkeypatch)
    assert not at.exception
    assert not any("needs a Gemma API key" in i.value for i in at.info)
    assert not at.chat_input[0].disabled
    assert [at.button(key=f"chat_ex_{i}").label for i in range(3)] == [
        "Why is the model failing here?", "What should we fix first?", "Which forecasting models fail most, and how?",
    ]
    assert fake_gemma.calls == [] and len(at.chat_message) == 0


def test_example_question_click_renders_a_grounded_answer(mock_runs, monkeypatch, fake_gemma):
    at = run_app(mock_runs, monkeypatch)
    at.button(key="chat_ex_1").click().run()
    assert not at.exception
    (call,) = fake_gemma.calls
    assert call["system"] == CHAT_SYSTEM and call["tag"] == "chat" and call["raw_dir"] is None
    assert call["model"] == "gemma-4-31b-it"  # the dense fallback model: the default 26B one hit 504s on chat prompts
    assert "Question: What should we fix first?" in call["user"]
    assert "Run overview: 20 series" in call["user"] and "Forecast lags actuals by days" in call["user"]
    assert "Focus:" not in call["user"]  # whole-run scope

    text = chat_text(at)
    assert "What should we fix first?" in text and "Scope: whole run" in text
    assert "The forecast trails the actuals by about three steps." in text
    assert "- best_lag 3.00 vs 0.00 for all anomalous windows" in text and "- lag_gain 0.57" in text
    assert "**Suggested next step:** Add leading indicators and compare MASE on these windows." in text
    assert "Confidence: 82%" in text
    assert [m.name for m in at.chat_message] == ["user", "assistant"]


def test_typed_question_uses_series_scope_and_remembers_history(mock_runs, monkeypatch, fake_gemma):
    at = run_app(mock_runs, monkeypatch)
    at.radio(key="chat_scope").set_value("A series").run()
    sel = at.selectbox(key="chat_series")
    assert sel.value == at.session_state["series_id"]  # defaults to the series open in the Series tab
    sel.select("S01").run()
    at.chat_input[0].set_value("Why is the model failing on this series?").run()
    assert not at.exception
    first = fake_gemma.calls[0]["user"]
    assert "series_id: S01" in first and "Question: Why is the model failing on this series?" in first
    assert "Earlier in this conversation" not in first

    at.chat_input[0].set_value("And the next step?").run()
    second = fake_gemma.calls[1]["user"]
    assert "Q: Why is the model failing on this series?" in second
    assert "A: The forecast trails the actuals by about three steps." in second
    assert [m.name for m in at.chat_message] == ["user", "assistant", "user", "assistant"]
    assert "Scope: series S01" in chat_text(at)


def test_cluster_scope_puts_the_cluster_report_in_the_context(mock_runs, monkeypatch, fake_gemma):
    at = run_app(mock_runs, monkeypatch, cluster_id=1)
    at.radio(key="chat_scope").set_value("A cluster").run()
    sel = at.selectbox(key="chat_cluster")
    assert sel.value == 1 and sel.options[1] == "1: Stuck at the old level"  # follows the Cluster tab
    at.button(key="chat_ex_0").click().run()
    user = fake_gemma.calls[0]["user"]
    assert "Focus: cluster 1" in user and "series_id" not in user
    assert "Scope: cluster 1" in chat_text(at)


def test_failed_gemma_call_shows_a_warning_and_is_not_kept_as_history(mock_runs, monkeypatch, fake_gemma):
    fake_gemma.result = None
    at = run_app(mock_runs, monkeypatch)
    at.button(key="chat_ex_0").click().run()
    assert not at.exception
    assert any("Gemma didn't answer (API busy) — try again in a moment." in w.value for w in at.warning)

    fake_gemma.result = ChatAnswer(answer="OK now.", evidence=["x 1.00"], suggested_next_step="Go.", confidence=0.5)
    at.chat_input[0].set_value("Try again").run()
    assert "Earlier in this conversation" not in fake_gemma.calls[1]["user"]  # the failed turn has no answer to quote
    assert "OK now." in chat_text(at)


def test_clear_conversation_and_per_run_history(mock_runs, monkeypatch, fake_gemma):
    at = run_app(mock_runs, monkeypatch)
    assert not [b for b in at.button if b.key == "chat_clear"]
    at.button(key="chat_ex_2").click().run()
    assert len(at.chat_message) == 2 and at.session_state["chat_log"]["mock"][0]["q"].startswith("Which forecasting models")
    at.run()  # the conversation survives a plain rerun
    assert len(at.chat_message) == 2
    at.button(key="chat_clear").click().run()
    assert not at.exception and len(at.chat_message) == 0


def test_answer_text_is_escaped_as_plain_text(mock_runs, monkeypatch, fake_gemma):
    fake_gemma.result = ChatAnswer(
        answer="# Big <b>heading</b> $x$", evidence=["[link](http://x.test)"], suggested_next_step="`rm -rf`", confidence=1.7
    )
    at = run_app(mock_runs, monkeypatch)
    at.button(key="chat_ex_0").click().run()
    assert not at.exception
    raw = "\n".join(m.value for msg in at.chat_message for m in msg.markdown)
    assert "\\# Big \\<b\\>heading\\</b\\> \\$x\\$" in raw and "\\[link\\]\\(http" in raw
    assert "Confidence: 100%" in "\n".join(c.value for msg in at.chat_message for c in msg.caption)  # clamped
