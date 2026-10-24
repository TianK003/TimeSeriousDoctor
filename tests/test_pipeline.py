from pathlib import Path

import numpy as np
import pytest

from atlas.config import Config
from atlas.data.synthetic import make_synthetic
from atlas.features import FEATURES
from atlas.pipeline import run_atlas
from atlas.schema import (
    CLUSTER_NOISE,
    CLUSTER_NOT_GATED,
    AtlasSummary,
    Cause,
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
