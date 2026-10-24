import copy
import json

import numpy as np
import pandas as pd
import pytest

from atlas.evaluate import (
    compare,
    evaluate_run,
    markdown_table,
    series_predictions,
    series_truth,
    stratified_sample,
)
from atlas.schema import ClusterInfo, GemmaClusterReport, RunResult

CLASSES = ["delayed_response", "missed_peak", "unadapted_shift"]


def report(failure_type: str) -> GemmaClusterReport:
    return GemmaClusterReport(
        failure_type=failure_type, name="n", summary="s", evidence=["e", "f"],
        likely_causes=[], suggested_fixes=["x"], metric_to_monitor="m", confidence=0.9,
    )


def cluster_info(cid: int, failure_type: str | None) -> ClusterInfo:
    return ClusterInfo(
        cluster_id=cid, size=4, n_series=2, medoid_window_ids=[], feature_median={},
        feature_median_all_gated={}, model_card_dist={},
        report=report(failure_type) if failure_type else None,
    )


def make_run() -> RunResult:
    """8 series x 2 windows: 2 series per failure class (cluster i = class i) + 2 healthy series."""
    labels = {f"s{i}": CLASSES[i // 2] for i in range(6)} | {"s6": "healthy", "s7": "healthy"}
    data_rows, win_rows = [], []
    for sid, label in labels.items():
        for step in range(6):
            forecast = step >= 2
            data_rows.append({
                "series_id": sid, "ts": pd.Timestamp("2024-01-01") + pd.Timedelta(days=step),
                "is_forecast": forecast,
                "true_label": label if forecast and step >= 4 else ("healthy" if forecast else "history"),
            })
        for k in range(2):
            failing = label != "healthy"
            win_rows.append({
                "window_id": f"{sid}#{k}", "series_id": sid, "true_label": label,
                "gated": failing, "cluster": CLASSES.index(label) if failing else -2,
            })
    return RunResult(
        name="unit", config={"embedder": "features"},
        data=pd.DataFrame(data_rows), series_meta=pd.DataFrame({"series_id": list(labels)}),
        windows=pd.DataFrame(win_rows), embeddings=np.zeros((12, 2), dtype=np.float32),
        clusters=[cluster_info(i, c) for i, c in enumerate(CLASSES)],
    )


def test_series_truth_ignores_history_and_healthy_rows():
    truth = series_truth(make_run())
    assert truth == {
        "s0": "delayed_response", "s1": "delayed_response",
        "s2": "missed_peak", "s3": "missed_peak",
        "s4": "unadapted_shift", "s5": "unadapted_shift",
        "s6": "healthy", "s7": "healthy",
    }


def test_series_truth_maps_healthy_noise():
    run = make_run()
    run.data["true_label"] = run.data["true_label"].replace("healthy", "healthy_noise")
    assert series_truth(run)["s6"] == "healthy"


def test_perfect_run():
    ev = evaluate_run(make_run())
    assert ev["run"] == "unit" and ev["embedder"] == "features"
    assert ev["gate"] == {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    cl = ev["clustering"]
    assert cl["ari"] == pytest.approx(1.0)
    assert cl["nmi"] == pytest.approx(1.0)
    assert cl["purity"] == pytest.approx(1.0)
    assert cl["noise_share"] == 0.0
    assert cl["confusion"] == {
        "0": {"delayed_response": 4}, "1": {"missed_peak": 4}, "2": {"unadapted_shift": 4},
    }
    assert ev["naming_accuracy"] == 1.0
    assert ev["series_accuracy"] == 1.0
    assert json.loads(json.dumps(ev)) == ev  # eval.json-serializable


def test_swapped_report_lowers_naming_and_series_accuracy():
    run = make_run()
    run.clusters = [cluster_info(0, "missed_peak"), cluster_info(1, "delayed_response"),
                    cluster_info(2, "unadapted_shift")]
    ev = evaluate_run(run)
    assert ev["naming_accuracy"] == pytest.approx(4 / 12)  # only cluster 2 is named right
    assert ev["series_accuracy"] == pytest.approx(4 / 8)   # s4, s5 and the healthy pair
    assert ev["clustering"]["ari"] == pytest.approx(1.0)   # clustering itself is unaffected


def test_missing_report_counts_as_wrong_and_series_become_other():
    run = make_run()
    run.clusters[2] = cluster_info(2, None)
    assert evaluate_run(run)["naming_accuracy"] == pytest.approx(8 / 12)
    preds = series_predictions(run)
    assert preds["s4"] == preds["s5"] == "other"
    assert preds["s0"] == "delayed_response"


def test_series_predictions_rules():
    run = make_run()
    w = run.windows
    # s0: one window noise, one in cluster 1 -> cluster 1 wins (noise never wins)
    w.loc[w["window_id"] == "s0#0", "cluster"] = -1
    w.loc[w["window_id"] == "s0#1", "cluster"] = 1
    # s2: only noise windows -> other
    w.loc[w["series_id"] == "s2", "cluster"] = -1
    # s6: gated nowhere -> healthy; s7 lands in a cluster named healthy_noise -> healthy
    w.loc[w["series_id"] == "s7", ["gated", "cluster"]] = [True, 2]
    run.clusters[2] = cluster_info(2, "healthy_noise")
    preds = series_predictions(run)
    assert preds["s0"] == "missed_peak"
    assert preds["s2"] == "other"
    assert preds["s6"] == "healthy"
    assert preds["s7"] == "healthy"
    assert preds["s4"] == "healthy"  # cluster 2 is now "healthy_noise" -> mapped to healthy


def test_gate_metrics_on_known_confusion():
    run = make_run()
    w = run.windows
    # 3 failing windows missed by the gate (FN), 2 healthy windows gated as noise (FP); TP = 9
    w.loc[w["window_id"].isin(["s0#0", "s2#0", "s4#0"]), ["gated", "cluster"]] = [False, -2]
    w.loc[w["window_id"].isin(["s6#0", "s7#0"]), ["gated", "cluster"]] = [True, -1]
    ev = evaluate_run(run)
    assert ev["gate"]["precision"] == pytest.approx(9 / 11)
    assert ev["gate"]["recall"] == pytest.approx(9 / 12)
    assert ev["gate"]["f1"] == pytest.approx(18 / 23)
    # clustering metrics only look at gated, non-healthy, clustered windows; healthy noise is in confusion
    assert ev["clustering"]["noise_share"] == 0.0
    assert ev["clustering"]["confusion"]["-1"] == {"healthy": 2}
    assert ev["clustering"]["purity"] == pytest.approx(1.0)


def test_noise_share_counts_gated_failing_windows_in_noise():
    run = make_run()
    run.windows.loc[run.windows["window_id"].isin(["s0#0", "s1#0", "s2#0"]), "cluster"] = -1
    ev = evaluate_run(run)
    assert ev["clustering"]["noise_share"] == pytest.approx(3 / 12)
    assert ev["clustering"]["confusion"]["-1"] == {"delayed_response": 2, "missed_peak": 1}


def test_no_gated_failures_gives_zeros():
    run = make_run()
    run.windows["gated"] = False
    run.windows["cluster"] = -2
    ev = evaluate_run(run)
    assert ev["gate"]["recall"] == 0.0 and ev["gate"]["f1"] == 0.0
    assert ev["clustering"]["ari"] == 0.0 and ev["naming_accuracy"] == 0.0
    assert ev["clustering"]["confusion"] == {}


def test_stratified_sample_balanced_and_deterministic():
    truth = {f"s{i:02d}": CLASSES[i % 3] for i in range(60)} | {f"h{i:02d}": "healthy" for i in range(20)}
    a = stratified_sample(truth, 16, seed=0)
    assert a == stratified_sample(truth, 16, seed=0)
    assert a == sorted(a) and len(set(a)) == 16
    counts = pd.Series([truth[s] for s in a]).value_counts()
    assert set(counts) == {4} and len(counts) == 4
    assert a != stratified_sample(truth, 16, seed=1)


def test_stratified_sample_small_class_and_remainder():
    truth = {"a": "healthy", "b": "missed_peak", "c": "missed_peak", "d": "missed_peak", "e": "missed_peak"}
    got = stratified_sample(truth, 6, seed=0)
    assert "a" in got and len(got) == 4  # healthy has only 1 series (quota 3), missed_peak gives 3
    assert stratified_sample({}, 5) == []


def test_compare_hand_checked():
    truth = {"a": "healthy", "b": "healthy", "c": "delayed_response", "d": "delayed_response",
             "e": "missed_peak", "f": "missed_peak"}
    pipeline = {"a": "healthy", "b": "healthy", "c": "delayed_response", "d": "delayed_response",
                "e": "missed_peak", "f": "healthy_noise"}
    baseline = {"a": "healthy_noise", "b": None, "c": "missed_peak", "d": "delayed_response",
                "e": "delayed_response", "f": "missed_peak"}
    ids = sorted(truth)
    out = compare(truth, pipeline, baseline, ids, pipeline_calls=4, n_series_total=60)

    assert out["n_series"] == 6
    p = out["pipeline"]
    assert p["accuracy"] == pytest.approx(5 / 6)
    assert p["per_class"] == {"delayed_response": 1.0, "healthy": 1.0, "missed_peak": 0.5}
    assert p["macro_f1"] == pytest.approx((0.8 + 1.0 + 2 / 3) / 3)
    assert p["gemma_calls"] == 4

    g = out["gemma_alone"]
    assert g["accuracy"] == pytest.approx(0.5)
    assert g["per_class"] == {"delayed_response": 0.5, "healthy": 0.5, "missed_peak": 0.5}
    assert g["macro_f1"] == pytest.approx((2 / 3 + 0.5 + 0.5) / 3)
    assert g["gemma_calls"] == 6

    assert out["gemma_calls_full_dataset"] == {"pipeline": 4, "gemma_alone": 60}


def test_compare_scores_only_the_sample_and_missing_counts_wrong():
    truth = {"a": "healthy", "b": "missed_peak", "c": "missed_peak"}
    out = compare(truth, {"a": "healthy", "b": "missed_peak"}, {}, ["a", "b"], 2, 3)
    assert out["n_series"] == 2
    assert out["pipeline"]["accuracy"] == 1.0
    assert out["gemma_alone"]["accuracy"] == 0.0 and out["gemma_alone"]["macro_f1"] == 0.0
    assert out["gemma_alone"]["gemma_calls"] == 2


def test_compare_records_failed_baseline_calls_and_unnamed_clusters():
    truth = {"a": "healthy", "b": "missed_peak", "c": "missed_peak"}
    baseline = {"a": "healthy", "b": None}  # b failed, c has no entry at all
    out = compare(truth, {"a": "healthy", "b": "missed_peak", "c": "other"}, baseline, ["a", "b", "c"], 3, 10,
                  n_clusters_without_report=1)
    assert out["gemma_alone"]["n_failed"] == 2
    assert out["gemma_alone"]["accuracy"] == pytest.approx(1 / 3)  # failures still score wrong
    assert out["pipeline"]["n_clusters_without_report"] == 1
    clean = compare(truth, {}, {"a": "healthy", "b": "missed_peak", "c": "missed_peak"}, ["a", "b", "c"], 3, 10)
    assert clean["gemma_alone"]["n_failed"] == 0 and clean["pipeline"]["n_clusters_without_report"] == 0


def test_markdown_table_footnotes_failures_only_when_there_are_any():
    ev = evaluate_run(make_run())
    truth = {"a": "healthy", "b": "missed_peak"}
    bad = copy.deepcopy(ev)
    bad["comparison"] = compare(truth, {"a": "healthy", "b": "other"}, {"a": "healthy", "b": None},
                                ["a", "b"], 5, 40, n_clusters_without_report=2)
    md = markdown_table([bad])
    assert "Note: 1 of 2 Gemma-alone calls failed and are scored as wrong" in md
    assert "Note: 2 pipeline cluster(s) have no Gemma report" in md
    assert md.index("| Gemma alone |") < md.index("Note: 1 of 2") < md.index("### Clustering metrics")

    good = copy.deepcopy(ev)
    good["comparison"] = compare(truth, {"a": "healthy", "b": "missed_peak"},
                                 {"a": "healthy", "b": "missed_peak"}, ["a", "b"], 5, 40)
    assert "Note:" not in markdown_table([good])
    old = copy.deepcopy(ev)  # an eval.json written before these fields existed
    old["comparison"] = good["comparison"]
    del old["comparison"]["gemma_alone"]["n_failed"], old["comparison"]["pipeline"]["n_clusters_without_report"]
    assert "Note:" not in markdown_table([old])


def test_markdown_table_has_both_tables():
    ev = evaluate_run(make_run())
    truth = {"a": "healthy", "b": "missed_peak"}
    with_comp = copy.deepcopy(ev)
    with_comp["comparison"] = compare(truth, {"a": "healthy", "b": "missed_peak"},
                                      {"a": "healthy", "b": None}, ["a", "b"], 5, 40)
    other = copy.deepcopy(ev) | {"run": "unit_moment", "embedder": "moment"}
    md = markdown_table([with_comp, other])

    assert "Pipeline vs Gemma alone" in md
    assert "| method | accuracy | macro-F1 | Gemma calls (sample) | Gemma calls (full dataset) |" in md
    assert "| Pipeline (ours) | 100.0% | 1.00 | 5 | 5 |" in md
    assert "| Gemma alone | 50.0% | 0.50 | 2 | 40 |" in md
    assert "| run | embedder | gate F1 | ARI | NMI | purity | noise share | naming acc | series acc |" in md
    assert "| unit | features | 1.00 | 1.00 | 1.00 | 1.00 | 0.0% | 100.0% | 100.0% |" in md
    assert "| unit_moment | moment |" in md
    assert md.index("Pipeline vs Gemma alone") < md.index("| run | embedder")


def test_markdown_table_without_comparison_skips_first_table():
    md = markdown_table([evaluate_run(make_run())])
    assert "Pipeline vs Gemma alone" not in md
    assert "| run | embedder" in md
