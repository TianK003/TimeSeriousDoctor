import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from atlas.baseline import run_baseline
from atlas.schema import ClusterInfo, GemmaBaselineAnswer, GemmaClusterReport, RunResult, load_run, save_run
from scripts.run_eval import main

CLASSES = ["delayed_response", "missed_peak", "unadapted_shift"]
TAXONOMY = Path(__file__).resolve().parent.parent / "taxonomy.yaml"
PNG_MAGIC = b"\x89PNG"
SERIES = [f"s{i}" for i in range(8)]  # s0-s5: two per failure class, s6-s7: healthy
TRUTH = {sid: (CLASSES[i // 2] if i < 6 else "healthy") for i, sid in enumerate(SERIES)}


def answer(failure_type: str) -> GemmaBaselineAnswer:
    return GemmaBaselineAnswer(failure_type=failure_type, confidence=0.7, explanation="looks like it")


class FakeClient:
    """Stands in for GemmaClient: `replies` maps series id -> answer (default: healthy_noise)."""

    def __init__(self, replies: dict[str, GemmaBaselineAnswer | None] | None = None):
        self.replies = replies or {}
        self.calls: list[dict] = []

    def generate(self, system, user, schema, image=None, tag="call"):
        self.calls.append({"system": system, "user": user, "schema": schema, "image": image, "tag": tag})
        return self.replies.get(tag.removeprefix("baseline_"), answer("healthy_noise"))


def report(failure_type: str) -> GemmaClusterReport:
    return GemmaClusterReport(
        failure_type=failure_type, name="n", summary="s", evidence=["e", "f"],
        likely_causes=[], suggested_fixes=["x"], metric_to_monitor="m", confidence=0.9,
    )


def make_run(tmp_path: Path, name: str = "unit") -> RunResult:
    """8 series x 6 steps (last 4 are forecast rows) + a perfect pipeline: cluster i names class i."""
    data_rows, win_rows = [], []
    for sid, label in TRUTH.items():
        for step in range(6):
            forecast = step >= 2
            data_rows.append({
                "series_id": sid, "ts": pd.Timestamp("2024-01-01") + pd.Timedelta(days=step),
                "actual": float(step + 1), "predicted": float(step), "is_forecast": forecast,
                "true_label": label if forecast and step >= 4 else ("healthy" if forecast else "history"),
            })
        for k in range(2):
            failing = label != "healthy"
            win_rows.append({
                "window_id": f"{sid}#{k}", "series_id": sid, "true_label": label,
                "gated": failing, "cluster": CLASSES.index(label) if failing else -2,
            })
    clusters = [
        ClusterInfo(
            cluster_id=i, size=4, n_series=2, medoid_window_ids=[], feature_median={},
            feature_median_all_gated={}, model_card_dist={}, report=report(c),
        )
        for i, c in enumerate(CLASSES)
    ]
    run = RunResult(
        name=name, config={"embedder": "features"}, data=pd.DataFrame(data_rows),
        series_meta=pd.DataFrame({"series_id": SERIES, "model_card": "AutoETS"}),
        windows=pd.DataFrame(win_rows), embeddings=np.zeros((12, 2), dtype=np.float32), clusters=clusters,
    )
    save_run(run, tmp_path / name)
    return run


# ------------------------------------------------------------------ run_baseline

def test_run_baseline_answers_every_series_and_writes_pngs(tmp_path):
    run = make_run(tmp_path)
    gc = FakeClient()
    answers = run_baseline(run, SERIES, gc, "SYSTEM")

    assert list(answers) == SERIES
    assert all(isinstance(a, GemmaBaselineAnswer) for a in answers.values())
    assert [c["tag"] for c in gc.calls] == [f"baseline_{s}" for s in SERIES]
    for call in gc.calls:
        assert call["system"] == "SYSTEM"
        assert call["schema"] is GemmaBaselineAnswer
        assert call["image"].startswith(PNG_MAGIC)
        assert "AutoETS" in call["user"]
        assert "delayed_response" not in call["user"]  # the truth must not leak into the prompt
    for sid in SERIES:
        png = run.path / "baseline_images" / f"{sid}.png"
        assert png.read_bytes().startswith(PNG_MAGIC)


def test_run_baseline_passes_none_through(tmp_path):
    run = make_run(tmp_path)
    gc = FakeClient({"s1": None})
    answers = run_baseline(run, ["s0", "s1"], gc, "SYSTEM")
    assert answers["s1"] is None
    assert answers["s0"] is not None


def test_run_baseline_prints_progress(tmp_path, capsys):
    run = make_run(tmp_path)
    run_baseline(run, SERIES, FakeClient(), "SYSTEM")
    assert "baseline: 8/8 series" in capsys.readouterr().out


# ------------------------------------------------------------------ run_eval.main

def test_main_writes_eval_with_both_methods(tmp_path, capsys):
    run = make_run(tmp_path)
    # Gemma alone: right on s0 (delayed_response), wrong on s2, healthy_noise on the healthy s6/s7
    # (counts as healthy), a failed call on s4, and everything else healthy_noise (wrong).
    fake = FakeClient({"s0": answer("delayed_response"), "s2": answer("delayed_response"), "s4": None})

    evals = main(["--runs", str(run.path), "--taxonomy", str(TAXONOMY)], gemma_client=fake)

    eval_json = json.loads((run.path / "eval.json").read_text(encoding="utf-8"))
    assert eval_json == evals[0]
    comp = eval_json["comparison"]
    assert comp["n_series"] == 8
    assert comp["pipeline"]["accuracy"] == 1.0
    assert comp["pipeline"]["gemma_calls"] == 4            # 3 clusters + 1 summary call
    # correct: s0, s6, s7 (healthy_noise -> healthy)
    assert comp["gemma_alone"]["accuracy"] == pytest.approx(3 / 8)
    assert comp["gemma_alone"]["gemma_calls"] == 8
    assert comp["gemma_alone"]["n_failed"] == 1             # the failed s4 call, visible in eval.json
    assert comp["pipeline"]["n_clusters_without_report"] == 0
    assert set(comp["gemma_alone"]["per_class"]) == set(CLASSES) | {"healthy"}
    assert comp["gemma_alone"]["per_class"]["healthy"] == 1.0
    assert comp["gemma_alone"]["per_class"]["delayed_response"] == 0.5
    assert comp["gemma_calls_full_dataset"] == {"pipeline": 4, "gemma_alone": 8}
    assert len(fake.calls) == 8

    baseline = json.loads((run.path / "baseline.json").read_text(encoding="utf-8"))
    assert baseline["truth"] == TRUTH
    assert baseline["answers"]["s4"] is None
    assert baseline["answers"]["s0"]["failure_type"] == "delayed_response"

    out = capsys.readouterr().out
    assert "Pipeline vs Gemma alone" in out and "Gemma alone" in out
    assert "WARNING: baseline: 1 of 8 Gemma-alone calls failed" in out and "Re-run" in out
    assert "Note: 1 of 8 Gemma-alone calls failed" in out  # footnote under the comparison table


def test_main_counts_clusters_without_report(tmp_path, capsys):
    run = make_run(tmp_path)
    run.cluster(0).report = None  # as if Gemma failed on cluster 0
    save_run(run, run.path)

    evals = main(["--runs", str(run.path), "--taxonomy", str(TAXONOMY)], gemma_client=FakeClient())

    assert evals[0]["comparison"]["pipeline"]["n_clusters_without_report"] == 1
    out = capsys.readouterr().out
    assert "WARNING: unit: 1 of 3 clusters have no Gemma report" in out
    assert "Note: 1 pipeline cluster(s) have no Gemma report" in out
    assert "Gemma-alone calls failed" not in out  # the baseline itself was fine


def test_main_samples_n_baseline_series(tmp_path):
    run = make_run(tmp_path)
    fake = FakeClient()
    main(["--runs", str(run.path), "--n-baseline", "4", "--taxonomy", str(TAXONOMY)], gemma_client=fake)
    assert len(fake.calls) == 4
    eval_json = json.loads((run.path / "eval.json").read_text(encoding="utf-8"))
    assert eval_json["comparison"]["n_series"] == 4
    assert eval_json["comparison"]["gemma_calls_full_dataset"]["gemma_alone"] == 8


def test_main_no_baseline_skips_gemma(tmp_path):
    run = make_run(tmp_path)
    fake = FakeClient()
    main(["--runs", str(run.path), "--no-baseline"], gemma_client=fake)
    assert fake.calls == []
    eval_json = json.loads((run.path / "eval.json").read_text(encoding="utf-8"))
    assert "comparison" not in eval_json
    assert eval_json["series_accuracy"] == 1.0
    assert not (run.path / "baseline.json").exists()


def test_main_computes_the_baseline_once_for_several_runs(tmp_path):
    first = make_run(tmp_path, "run_a")
    shutil.copytree(first.path, tmp_path / "run_b")
    fake = FakeClient()

    main(["--runs", str(first.path), str(tmp_path / "run_b"), "--taxonomy", str(TAXONOMY)], gemma_client=fake)

    assert len(fake.calls) == 8
    assert (tmp_path / "run_a" / "baseline.json").exists()
    assert not (tmp_path / "run_b" / "baseline.json").exists()
    for name in ("run_a", "run_b"):
        comp = load_run(tmp_path / name).eval["comparison"]
        assert comp["n_series"] == 8
        assert comp["pipeline"]["accuracy"] == 1.0
        assert comp["gemma_alone"]["accuracy"] == pytest.approx(2 / 8)  # only the healthy series


def test_main_without_api_key_stops_before_any_work(tmp_path, monkeypatch):
    run = make_run(tmp_path)
    monkeypatch.setattr("scripts.run_eval.get_api_key", lambda: None)
    with pytest.raises(SystemExit, match="no API key"):
        main(["--runs", str(run.path), "--taxonomy", str(TAXONOMY)])
    assert not (run.path / "eval.json").exists()
