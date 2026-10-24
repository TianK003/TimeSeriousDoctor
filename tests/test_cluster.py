import numpy as np
import pandas as pd

from atlas import cluster
from atlas.cluster import cluster_windows, embed_features, project_2d, summarize_clusters
from atlas.schema import CLUSTER_NOISE

FEATS = ["f1", "f2", "f3"]


def _blobs(sizes=(120, 80, 40), dim=4, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0] * dim, [12.0] * dim, [-12.0, 12.0] * (dim // 2)])
    X = np.vstack([rng.normal(c, 0.5, size=(n, dim)) for c, n in zip(centers, sizes)])
    truth = np.repeat(np.arange(len(sizes)), sizes)
    return X, truth


def test_three_blobs_give_three_clusters_biggest_first():
    X, truth = _blobs()
    labels = cluster_windows(X, min_cluster_size=20)
    assert sorted(set(labels.tolist())) == [0, 1, 2]
    assert [int((labels == c).sum()) for c in range(3)] == [120, 80, 40]
    # label 0 is the biggest blob, and clusters are pure
    for c in range(3):
        assert len(set(truth[labels == c].tolist())) == 1
    assert set(truth[labels == 0].tolist()) == {0}


def test_high_dim_input_is_pca_reduced_and_still_clusters():
    X, truth = _blobs(dim=24)
    labels = cluster_windows(X, min_cluster_size=20)
    assert _n(labels) == 3
    assert set(truth[labels == 0].tolist()) == {0}


def _n(labels):
    return len(set(labels[labels >= 0].tolist()))


def test_uniform_noise_falls_back_to_kmeans_with_at_least_three_clusters():
    X = np.random.default_rng(1).uniform(size=(300, 4))
    labels = cluster_windows(X, min_cluster_size=20)
    assert _n(labels) >= 3
    assert not (labels == CLUSTER_NOISE).any()  # KMeans assigns every point
    sizes = [int((labels == c).sum()) for c in range(_n(labels))]
    assert sizes == sorted(sizes, reverse=True)


def test_too_many_clusters_is_capped():
    rng = np.random.default_rng(2)
    centers = rng.uniform(-60, 60, size=(12, 3))
    X = np.vstack([rng.normal(c, 0.3, size=(30, 3)) for c in centers])
    labels = cluster_windows(X, min_cluster_size=10, max_clusters=5)
    assert 2 <= _n(labels) <= 5


def test_tiny_input_does_not_crash():
    assert cluster_windows(np.zeros((2, 3))).tolist() == [0, 0]


def _windows(labels, n_series=6):
    n = len(labels)
    rng = np.random.default_rng(3)
    df = pd.DataFrame({
        "window_id": [f"s{i % n_series}#{i}" for i in range(n)],
        "series_id": [f"s{i % n_series}" for i in range(n)],
        "f1": rng.normal(size=n),
        "f2": rng.normal(size=n),
        "f3": rng.normal(size=n),
    })
    df.loc[0, "f2"] = np.nan
    meta = pd.DataFrame({
        "series_id": [f"s{i}" for i in range(n_series)],
        "model_card": ["A", "A", "A", "B", "B", "C"],
    })
    return df, meta


def test_summaries_sizes_medoids_and_model_card_dist():
    labels = np.array([0] * 20 + [1] * 12 + [CLUSTER_NOISE] * 4)
    windows, meta = _windows(labels)
    X = np.random.default_rng(4).normal(size=(len(labels), 3)).astype(np.float32)
    infos = summarize_clusters(windows, labels, X, meta, FEATS)

    assert [c.cluster_id for c in infos] == [0, 1, CLUSTER_NOISE]
    assert [c.size for c in infos] == [20, 12, 4]
    for c in infos:
        members = set(windows.loc[labels == c.cluster_id, "window_id"])
        assert 1 <= len(c.medoid_window_ids) <= 3
        assert set(c.medoid_window_ids) <= members
        assert abs(sum(c.model_card_dist.values()) - 1.0) < 1e-9
        shares = list(c.model_card_dist.values())
        assert shares == sorted(shares, reverse=True)
        assert list(c.feature_median) == FEATS
        assert list(c.feature_median_all_gated) == FEATS
        assert all(type(v) is float and np.isfinite(v) for v in c.feature_median.values())
        assert c.image_path is None and c.report is None
    c0 = infos[0]
    assert c0.n_series == windows.loc[labels == 0, "series_id"].nunique()
    assert np.isclose(c0.feature_median["f1"], windows.loc[labels == 0, "f1"].median())
    assert np.isclose(c0.feature_median_all_gated["f1"], windows["f1"].median())
    # NaN-safe: f2 median ignores the NaN in row 0 (cluster 0)
    assert np.isclose(c0.feature_median["f2"], windows.loc[labels == 0, "f2"].dropna().median())


def test_summaries_without_noise_have_no_noise_bucket():
    labels = np.array([0] * 10 + [1] * 8)
    windows, meta = _windows(labels)
    X = np.random.default_rng(5).normal(size=(len(labels), 3))
    infos = summarize_clusters(windows, labels, X, meta, FEATS)
    assert [c.cluster_id for c in infos] == [0, 1]
    assert sum(c.size for c in infos) == len(labels)


def test_medoid_is_closest_to_centroid():
    X = np.array([[0.0, 0], [0.1, 0], [-0.1, 0], [5, 5], [5.1, 5], [5, 5.1]])
    labels = np.array([0, 0, 0, 1, 1, 1])
    windows = pd.DataFrame({
        "window_id": list("abcdef"), "series_id": ["s0"] * 6,
        "f1": np.arange(6.0), "f2": 0.0, "f3": 0.0,
    })
    meta = pd.DataFrame({"series_id": ["s0"], "model_card": ["A"]})
    infos = summarize_clusters(windows, labels, X, meta, FEATS)
    assert infos[0].medoid_window_ids[0] == "a"


def test_embed_features_shape_dtype_and_nan_handling():
    df = pd.DataFrame({
        "f1": [1.0, 2.0, 3.0, 4.0, np.nan],
        "f2": [10.0, np.inf, -np.inf, 0.0, 5.0],
        "f3": [7.0] * 5,  # constant column
    })
    E = embed_features(df, FEATS)
    assert E.shape == (5, 3)
    assert E.dtype == np.float32
    assert np.isfinite(E).all()
    assert E[4, 0] == 0.0 and E[1, 1] == 0.0 and E[2, 1] == 0.0
    assert np.allclose(E[:, 2], 0.0)


def test_project_2d_shape():
    X = np.random.default_rng(6).normal(size=(50, 8)).astype(np.float32)
    P = project_2d(X)
    assert P.shape == (50, 2) and np.isfinite(P).all()
    assert project_2d(X[:2]).shape == (2, 2)
    assert (project_2d(X[:2]) == 0).all()
    assert (project_2d(X[:, :1]) == 0).all()


def test_hdbscan_always_too_many_clusters_uses_kmeans_with_max_clusters(monkeypatch):
    import atlas.cluster as cl

    class FakeHDBSCAN:
        def __init__(self, min_cluster_size, copy=True):
            pass

        def fit_predict(self, Z):
            return np.arange(len(Z)) % 10  # 10 clusters, always > max_clusters

    monkeypatch.setattr(cl, "HDBSCAN", FakeHDBSCAN)
    X = np.random.default_rng(7).normal(size=(100, 3))
    labels = cl.cluster_windows(X, min_cluster_size=5, max_clusters=4)
    assert _n(labels) == 4


def test_embed_features_winsorizes_heavy_tails():
    rng = np.random.default_rng(6)
    x = rng.normal(size=300)
    x[:3] = [1e6, -5e5, 3e4]  # a few outliers hundreds of IQRs away
    df = pd.DataFrame({"f1": x, "f2": rng.normal(size=300), "f3": rng.exponential(size=300)})
    E = embed_features(df, FEATS)
    assert np.abs(E).max() <= 5.0 + 1e-6
    assert E[0, 0] == 5.0 and E[1, 0] == -5.0       # clipped, not dropped
    assert E[3:, 0].std() > 0.5                    # the bulk keeps its spread (not squashed by the outliers)


def _share(labels):
    return np.bincount(labels[labels >= 0]).max() / len(labels)


def test_kmeans_silhouette_skips_a_giant_cluster_solution(monkeypatch):
    rng = np.random.default_rng(7)
    giant = rng.normal(0, 1, size=(600, 2))
    far = [rng.normal(c, 0.2, size=(15, 2)) for c in ([40.0, 0.0], [0.0, 40.0], [-40.0, -40.0])]
    X = np.vstack([giant, *far])

    labels = cluster_windows(X, min_cluster_size=20)
    assert _n(labels) >= 4 and _share(labels) <= 0.5

    # without the cap the best silhouette is the useless 93 % cluster plus the three tiny ones
    monkeypatch.setattr(cluster, "MAX_LARGEST_SHARE", 1.0)
    assert _share(cluster_windows(X, min_cluster_size=20)) > 0.9


def test_hdbscan_result_with_a_giant_cluster_falls_back_to_kmeans(monkeypatch):
    rng = np.random.default_rng(8)
    X = np.vstack([rng.normal(0, 1, size=(400, 3)), rng.normal(15, 0.3, size=(30, 3))])

    labels = cluster_windows(X, min_cluster_size=20)
    assert _n(labels) >= 3 and _share(labels) <= 0.5
    assert not (labels == CLUSTER_NOISE).any()

    monkeypatch.setattr(cluster, "MAX_LARGEST_SHARE", 1.0)  # HDBSCAN's 93 % + 7 % answer is accepted again
    assert _share(cluster_windows(X, min_cluster_size=20)) > 0.9
