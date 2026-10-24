import json
from pathlib import Path

import numpy as np
import pytest

from atlas.config import Config
from atlas.data.synthetic import make_synthetic
from atlas.features import FEATURES
from atlas.pipeline import FALLBACK_NOTE, fallback_summary, run_atlas
from atlas.prompts import EXPERT_HEADER
from atlas.schema import (
    CLUSTER_NOISE,
    CLUSTER_NOT_GATED,
    AtlasSummary,
    Cause,
    ClusterInfo,
    Dataset,
    GemmaClusterReport,
    load_run,
    save_dataset,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


class FakeGemma:
    """Stands in for GemmaClient: canned replies by schema, records the calls."""

    def __init__(self):
        self.calls: list[tuple[str, type, bool]] = []

    def generate(self, system, user, schema, image=None, tag=""):
        self.calls.append((tag, schema, image is not None))
        if schema is GemmaClusterReport:
            return GemmaClusterReport(
                failure_type="delayed_response",
                name=f"canned {tag}",
                summary="Forecast trails the actual.",
                evidence=["best_lag is positive", "lag_gain is large"],
                likely_causes=[Cause(cause="lagged features", why="model reacts late")],
                suggested_fixes=["add a lead indicator"],
                metric_to_monitor="lag_gain",
                confidence=0.8,
            )
        assert schema is AtlasSummary
        return AtlasSummary(headline="Headline", summary="Summary.", actions=["a", "b", "c"])


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("synthetic")
    data, meta = make_synthetic(n_series=40, history_len=120, forecast_len=84, seed=0)
    save_dataset(Dataset(data, meta), path)
    return path


def make_cfg(tmp_path: Path) -> Config:
    cfg = Config(runs_dir=tmp_path / "runs")
    return cfg.model_copy(update={"cluster": cfg.cluster.model_copy(update={"min_cluster_size": 5})})


def test_run_atlas_with_gemma(data_dir, tmp_path, capsys):
    fake = FakeGemma()
    run = run_atlas(data_dir, make_cfg(tmp_path), "t", use_gemma=True, gemma_client=fake, taxonomy_path=REPO_ROOT / "taxonomy.yaml")
    out = capsys.readouterr().out
    for stage in ("load", "residuals", "windows", "gate", "features", "embed", "cluster", "summarize", "panels", "gemma", "save"):
        assert f"stage: {stage} …" in out

    loaded = load_run(tmp_path / "runs" / "t")
    assert loaded.name == "t"
    assert loaded.config["data_dir"] == str(data_dir)
    assert loaded.summary is not None and loaded.summary.headline == "Headline"
    assert len(loaded.clusters) == len(run.clusters)

    w = loaded.windows
    cols = list(w.columns)
    assert cols.index("score_max") == cols.index("score_bias") + 1
    assert all(f in cols for f in FEATURES)
    assert cols[-4:] == ["emb_row", "cluster", "x2d", "y2d"]
    gated = w[w["gated"]]
    assert 0 < len(gated) < len(w)
    assert loaded.embeddings.shape == (len(gated), len(FEATURES))
    assert loaded.embeddings.dtype == np.float32
    assert sorted(gated["emb_row"]) == list(range(len(gated)))
    assert (w.loc[~w["gated"], "emb_row"] == -1).all()
    assert (w.loc[~w["gated"], "cluster"] == CLUSTER_NOT_GATED).all()
    assert (gated["cluster"] >= CLUSTER_NOISE).all()
    assert w.loc[~w["gated"], ["x2d", "y2d"]].isna().all().all()
    assert gated[["x2d", "y2d"]].notna().all().all()

    sizes = w[w["gated"]]["cluster"].value_counts()
    for c in loaded.clusters:
        assert c.size == sizes[c.cluster_id]
        assert (loaded.path / c.image_path).is_file()
        if c.cluster_id >= 0:
            assert c.report is not None and c.report.name == f"canned cluster_{c.cluster_id}"
        else:
            assert c.report is None and c.image_path == "images/cluster_noise.png"
    assert any(c.cluster_id >= 0 for c in loaded.clusters)
    n_clusters = sum(c.cluster_id >= 0 for c in loaded.clusters)
    assert [s for _, s, _ in fake.calls].count(GemmaClusterReport) == n_clusters
    assert all(has_image for _, s, has_image in fake.calls if s is GemmaClusterReport)
    assert f"Gemma reports {n_clusters}/{n_clusters}" in out


def test_run_atlas_without_gemma(data_dir, tmp_path, capsys):
    cfg = make_cfg(tmp_path)
    run_atlas(data_dir, cfg, "nog", use_gemma=False)
    assert "stage: gemma" not in capsys.readouterr().out
    loaded = load_run(tmp_path / "runs" / "nog")
    assert loaded.summary is None
    assert not (tmp_path / "runs" / "nog" / "summary.json").exists()
    assert loaded.clusters and all(c.report is None for c in loaded.clusters)
    assert all((loaded.path / c.image_path).is_file() for c in loaded.clusters)


def test_moment_embedder_not_available(data_dir, tmp_path):
    cfg = make_cfg(tmp_path).model_copy(update={"embedder": "moment"})
    with pytest.raises(NotImplementedError, match="--embedder features"):
        run_atlas(data_dir, cfg, "m", use_gemma=False)


# ---------------------------------------------------------------- expert feedback (P2.2)

class RecordingGemma(FakeGemma):
    def __init__(self):
        super().__init__()
        self.cluster_users: list[str] = []

    def generate(self, system, user, schema, image=None, tag=""):
        if schema is GemmaClusterReport:
            self.cluster_users.append(user)
        return super().generate(system, user, schema, image, tag)


def entry(status: str, name: str) -> dict:
    return {
        "status": status, "name": name, "failure_type": "delayed_response", "note": f"note for {name}",
        "gemma_name": f"gemma {name}", "summary": "s", "feature_median": {"best_lag": 3.0},
    }


def test_run_atlas_passes_expert_examples_into_every_cluster_prompt(data_dir, tmp_path):
    prev = tmp_path / "prev"
    prev.mkdir()
    feedback = {
        "0": entry("rejected", "RejectedOne"), "1": entry("renamed", "RenamedOne"),
        "2": entry("confirmed", "ConfirmedOne"), "3": entry("confirmed", "ConfirmedTwo"),
        "4": {"status": "bogus", "name": "Ignored"},
    }
    (prev / "expert_feedback.json").write_text(json.dumps(feedback), encoding="utf-8")

    fake = RecordingGemma()
    run_atlas(data_dir, make_cfg(tmp_path), "t", gemma_client=fake, taxonomy_path=REPO_ROOT / "taxonomy.yaml",
              expert_from=prev)
    assert fake.cluster_users
    for user in fake.cluster_users:
        assert EXPERT_HEADER in user
        assert "ConfirmedOne" in user and "ConfirmedTwo" in user and "RenamedOne" in user  # confirmed first, max 3
        assert "RejectedOne" not in user and "Ignored" not in user
    assert not (tmp_path / "runs" / "t" / "expert_feedback.json").exists()  # nothing else is copied


def test_run_atlas_expert_from_without_feedback_file_still_runs(data_dir, tmp_path, capsys):
    empty = tmp_path / "empty_prev"
    empty.mkdir()
    fake = RecordingGemma()
    run_atlas(data_dir, make_cfg(tmp_path), "t", gemma_client=fake, taxonomy_path=REPO_ROOT / "taxonomy.yaml",
              expert_from=empty)
    assert "no expert_feedback.json" in capsys.readouterr().out
    assert fake.cluster_users and all(EXPERT_HEADER not in u for u in fake.cluster_users)


def test_cli_accepts_expert_from(monkeypatch):
    import scripts.run_atlas as cli

    seen = {}
    monkeypatch.setattr(cli, "run_atlas", lambda *a, **kw: seen.update(kw))
    cli.main(["--data", "d", "--run-name", "r", "--expert-from", "runs/prev"])
    assert seen["expert_from"] == Path("runs/prev")


# ---------------------------------------------------------------- fallback executive summary

class NoSummaryGemma(FakeGemma):
    """Cluster reports work, the executive summary call fails (returns None), as the real API often did."""

    def generate(self, system, user, schema, image=None, tag=""):
        if schema is AtlasSummary:
            self.calls.append((tag, schema, image is not None))
            return None
        return super().generate(system, user, schema, image, tag)


def make_report(name: str, fixes: list[str]) -> GemmaClusterReport:
    return GemmaClusterReport(
        failure_type="missed_peak", name=name, summary="s", evidence=["e1", "e2"],
        likely_causes=[Cause(cause="c", why="w")], suggested_fixes=fixes, metric_to_monitor="m", confidence=0.5,
    )


def make_info(cluster_id: int, size: int, n_series: int, report=None) -> ClusterInfo:
    return ClusterInfo(
        cluster_id=cluster_id, size=size, n_series=n_series, medoid_window_ids=[], feature_median={},
        feature_median_all_gated={}, model_card_dist={}, report=report,
    )


def test_fallback_summary_lists_the_three_largest_clusters():
    import pandas as pd

    clusters = [
        make_info(0, 50, 8, make_report("Big one", ["fix A1", "fix A2"])),
        make_info(1, 40, 6, make_report("Medium", ["fix B1"])),
        make_info(2, 30, 5, make_report("Small", ["fix C1"])),
        make_info(3, 20, 4, make_report("Smallest", ["fix D1"])),
        make_info(4, 10, 3),                                    # no report: never listed
        make_info(CLUSTER_NOISE, 99, 12),                       # noise bucket: not a pattern
    ]
    windows = pd.DataFrame({"series_id": ["a", "a", "b", "c", "d"], "cluster": [0, 1, 2, CLUSTER_NOISE, CLUSTER_NOT_GATED]})
    s = fallback_summary(clusters, windows)
    assert s.headline == "5 recurring failure patterns across 2 series"  # series in real clusters: a, b
    assert "Big one (50 windows, 8 series)" in s.summary
    assert "Medium (40 windows, 6 series)" in s.summary and "Small (30 windows, 5 series)" in s.summary
    assert "Smallest" not in s.summary and "No report" not in s.summary
    assert s.summary.index("Big one") < s.summary.index("Medium") < s.summary.index("Small (")
    assert s.summary.endswith(FALLBACK_NOTE) and FALLBACK_NOTE == "(summary composed from the cluster reports)"
    assert s.actions == ["fix A1", "fix B1", "fix C1"]


def test_fallback_summary_handles_single_cluster_and_none():
    import pandas as pd

    windows = pd.DataFrame({"series_id": ["a"], "cluster": [0]})
    one = fallback_summary([make_info(0, 5, 1, make_report("Only", []))], windows)
    assert one.headline == "1 recurring failure pattern across 1 series" and one.actions == []
    assert fallback_summary([make_info(0, 5, 1), make_info(CLUSTER_NOISE, 3, 2)], windows) is None


def test_run_atlas_composes_a_summary_when_gemma_returns_none(data_dir, tmp_path, capsys):
    fake = NoSummaryGemma()
    run_atlas(data_dir, make_cfg(tmp_path), "t", gemma_client=fake, taxonomy_path=REPO_ROOT / "taxonomy.yaml")
    assert "composed one from the cluster reports" in capsys.readouterr().out
    loaded = load_run(tmp_path / "runs" / "t")
    n = sum(c.cluster_id >= 0 for c in loaded.clusters)
    assert loaded.summary is not None and loaded.summary.headline.startswith(f"{n} recurring failure pattern")
    assert loaded.summary.summary.endswith(FALLBACK_NOTE)
    assert loaded.summary.actions == ["add a lead indicator"] * min(n, 3)


def test_run_atlas_keeps_gemmas_summary_when_it_succeeds(data_dir, tmp_path):
    run_atlas(data_dir, make_cfg(tmp_path), "t", gemma_client=FakeGemma(), taxonomy_path=REPO_ROOT / "taxonomy.yaml")
    assert load_run(tmp_path / "runs" / "t").summary.headline == "Headline"


# ---------------------------------------------------------------- scripts/refresh_summary.py

def snapshot(run_dir: Path) -> dict[str, tuple[bytes, int]]:
    return {
        p.name: (p.read_bytes(), p.stat().st_mtime_ns)
        for p in run_dir.iterdir() if p.is_file() and p.name != "summary.json"
    }


@pytest.fixture()
def run_without_summary(data_dir, tmp_path) -> Path:
    run_atlas(data_dir, make_cfg(tmp_path), "t", gemma_client=FakeGemma(), taxonomy_path=REPO_ROOT / "taxonomy.yaml")
    run_dir = tmp_path / "runs" / "t"
    (run_dir / "summary.json").unlink()
    return run_dir


def test_refresh_summary_writes_gemmas_summary_and_touches_nothing_else(run_without_summary):
    import scripts.refresh_summary as cli

    before = snapshot(run_without_summary)
    out = cli.main(["--run", str(run_without_summary)], gemma_client=FakeGemma())
    assert out is not None and out.headline == "Headline"
    assert load_run(run_without_summary).summary.headline == "Headline"
    assert snapshot(run_without_summary) == before


def test_refresh_summary_falls_back_to_the_composed_summary(run_without_summary, capsys):
    import scripts.refresh_summary as cli

    before = snapshot(run_without_summary)
    cli.main(["--run", str(run_without_summary)], gemma_client=NoSummaryGemma())
    summary = load_run(run_without_summary).summary
    assert summary.summary.endswith(FALLBACK_NOTE) and "recurring failure pattern" in summary.headline
    assert "composed from the cluster reports" in capsys.readouterr().out
    assert snapshot(run_without_summary) == before


def test_refresh_summary_without_reports_writes_nothing(data_dir, tmp_path, capsys):
    import scripts.refresh_summary as cli

    run_atlas(data_dir, make_cfg(tmp_path), "nog", use_gemma=False)
    run_dir = tmp_path / "runs" / "nog"
    assert cli.main(["--run", str(run_dir)], gemma_client=NoSummaryGemma()) is None
    assert not (run_dir / "summary.json").exists()
    assert "no cluster has a report" in capsys.readouterr().out
