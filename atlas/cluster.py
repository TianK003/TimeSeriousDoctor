"""Clustering, medoids, cluster summaries, 2D projection.

Owner: A. Task: P0.9, P1.8. Spec: SPEC.md §7.7.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler

from atlas.schema import CLUSTER_NOISE, ClusterInfo

PCA_DIMS = 16
MAX_NOISE_SHARE = 0.6
KMEANS_K_RANGE = (3, 8)
HDBSCAN_RETRIES = 3


def embed_features(windows: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Feature columns -> RobustScaler -> NaN/inf replaced by 0. float32 [n, len(features)]."""
    vals = windows[features].to_numpy(dtype=np.float64)
    vals = np.where(np.isfinite(vals), vals, np.nan)  # RobustScaler rejects inf, tolerates NaN
    scaled = RobustScaler().fit_transform(vals)
    return np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _relabel_by_size(labels: np.ndarray) -> np.ndarray:
    """Noise stays -1; other clusters become 0..n-1 ordered by size descending."""
    ids, counts = np.unique(labels[labels >= 0], return_counts=True)
    order = ids[np.argsort(-counts, kind="stable")]
    out = np.full(labels.shape, CLUSTER_NOISE, dtype=int)
    for new_id, old_id in enumerate(order):
        out[labels == old_id] = new_id
    return out


def _n_clusters(labels: np.ndarray) -> int:
    return len(set(labels[labels >= 0].tolist()))


def _kmeans_silhouette(Z: np.ndarray, max_clusters: int, seed: int) -> np.ndarray:
    n = len(Z)
    hi = min(KMEANS_K_RANGE[1], max_clusters, n - 1)
    lo = min(KMEANS_K_RANGE[0], hi)
    best_labels, best_score = None, -np.inf
    for k in range(lo, hi + 1):
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(Z)
        if len(set(labels.tolist())) < 2:
            continue
        score = silhouette_score(Z, labels, sample_size=min(n, 3000), random_state=seed)
        if score > best_score:
            best_labels, best_score = labels, score
    if best_labels is None:  # degenerate data (e.g. all identical rows)
        best_labels = KMeans(n_clusters=hi, n_init=10, random_state=seed).fit_predict(Z)
    return best_labels


def cluster_windows(
    X: np.ndarray, min_cluster_size: int = 20, max_clusters: int = 8, seed: int = 0
) -> np.ndarray:
    """Cluster rows of X. int labels, -1 = noise, clusters relabeled 0..n-1 by size descending."""
    X = np.asarray(X, dtype=np.float64)
    n = len(X)
    if n < 4:  # too few points to cluster meaningfully
        print(f"cluster_windows: only {n} points, single cluster")
        return np.zeros(n, dtype=int)

    Z = X
    if X.shape[1] > PCA_DIMS:
        Z = PCA(n_components=PCA_DIMS, random_state=seed).fit_transform(X)

    mcs = max(2, min(int(min_cluster_size), n - 1))
    method = "hdbscan"
    fallback: str | None = None  # None | "silhouette" | "max_k"
    for attempt in range(HDBSCAN_RETRIES + 1):
        labels = HDBSCAN(min_cluster_size=mcs, copy=True).fit_predict(Z)
        k = _n_clusters(labels)
        noise_share = float(np.mean(labels == CLUSTER_NOISE))
        if k < 2 or noise_share > MAX_NOISE_SHARE:
            fallback = "silhouette"
            method = f"kmeans-silhouette (hdbscan gave {k} clusters, {noise_share:.0%} noise)"
            break
        if k <= max_clusters:
            method = f"hdbscan (min_cluster_size={mcs})"
            break
        if attempt == HDBSCAN_RETRIES:
            fallback = "max_k"
            method = f"kmeans k={max_clusters} (hdbscan still gave {k} clusters > {max_clusters})"
            break
        mcs = min(max(mcs + 1, int(math.ceil(mcs * 1.5))), n - 1)

    if fallback == "silhouette":
        labels = _kmeans_silhouette(Z, max_clusters, seed)
    elif fallback == "max_k":
        labels = KMeans(n_clusters=min(max_clusters, n - 1), n_init=10, random_state=seed).fit_predict(Z)

    labels = _relabel_by_size(labels)
    sizes = [int((labels == c).sum()) for c in range(_n_clusters(labels))]
    n_noise = int((labels == CLUSTER_NOISE).sum())
    print(f"cluster_windows: {method}; cluster sizes {sizes}, noise {n_noise}")
    return labels


def _finite_median(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    return float(np.median(values)) if len(values) else 0.0


def _medians(df: pd.DataFrame, features: list[str]) -> dict[str, float]:
    return {f: _finite_median(df[f].to_numpy(dtype=np.float64)) for f in features}


def summarize_clusters(
    windows: pd.DataFrame,
    labels: np.ndarray,
    X: np.ndarray,
    series_meta: pd.DataFrame,
    features: list[str],
) -> list[ClusterInfo]:
    """One ClusterInfo per cluster 0..n-1, plus a -1 bucket last if any noise.

    `windows` are the gated windows, row-aligned with `labels` and `X`.
    """
    windows = windows.reset_index(drop=True)
    labels = np.asarray(labels)
    X = np.asarray(X, dtype=np.float64)
    window_ids = windows["window_id"].to_numpy()
    card_of = series_meta.set_index("series_id")["model_card"].to_dict()
    cards = windows["series_id"].map(card_of).fillna("unknown")
    all_median = _medians(windows, features)

    cluster_ids = list(range(_n_clusters(labels)))
    if (labels == CLUSTER_NOISE).any():
        cluster_ids.append(CLUSTER_NOISE)

    infos: list[ClusterInfo] = []
    for cid in cluster_ids:
        idx = np.flatnonzero(labels == cid)
        members = windows.iloc[idx]
        Xc = X[idx]
        dist = np.linalg.norm(Xc - Xc.mean(axis=0), axis=1)
        medoids = [str(window_ids[idx[i]]) for i in np.argsort(dist, kind="stable")[:3]]
        card_share = cards.iloc[idx].value_counts(normalize=True).sort_values(ascending=False)
        infos.append(
            ClusterInfo(
                cluster_id=cid,
                size=len(idx),
                n_series=int(members["series_id"].nunique()),
                medoid_window_ids=medoids,
                feature_median=_medians(members, features),
                feature_median_all_gated=dict(all_median),
                model_card_dist={str(k): float(v) for k, v in card_share.items()},
                image_path=None,
                report=None,
            )
        )
    return infos


def project_2d(X: np.ndarray) -> np.ndarray:
    """PCA(2) projection [n, 2]; zeros if n < 3 or fewer than 2 dims."""
    X = np.asarray(X)
    n = len(X)
    if n < 3 or X.ndim < 2 or X.shape[1] < 2:
        return np.zeros((n, 2), dtype=np.float32)
    return PCA(n_components=2, random_state=0).fit_transform(X).astype(np.float32)


def project_tsne(
    X: np.ndarray, max_points: int = 4000, seed: int = 0, keep: np.ndarray | None = None
) -> np.ndarray:
    """t-SNE projection [n, 2] float32 for the latent-space map; NaN rows = points not projected.

    If n > max_points, every row where `keep` is True (e.g. gated windows) is projected plus a seeded
    random sample of the remaining rows, up to max_points in total. n < 5 -> zeros.
    """
    X = np.asarray(X, dtype=np.float64)
    n = len(X)
    if n < 5:
        return np.zeros((n, 2), dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    chosen = np.arange(n)
    if n > max_points:
        keep_mask = np.zeros(n, dtype=bool) if keep is None else np.asarray(keep, dtype=bool)
        kept = np.flatnonzero(keep_mask)
        rest = np.flatnonzero(~keep_mask)
        n_extra = min(len(rest), max(0, max_points - len(kept)))
        extra = np.random.default_rng(seed).choice(rest, size=n_extra, replace=False)
        chosen = np.sort(np.concatenate([kept, extra]))

    m = len(chosen)
    out = np.full((n, 2), np.nan, dtype=np.float32)
    if m < 5:
        out[chosen] = 0.0
        return out
    t0 = time.perf_counter()
    emb = TSNE(
        n_components=2,
        perplexity=min(30.0, (m - 1) / 3),
        init="pca" if X.shape[1] >= 2 else "random",
        random_state=seed,
    ).fit_transform(X[chosen])
    out[chosen] = emb.astype(np.float32)
    print(f"project_tsne: {m}/{n} points, {time.perf_counter() - t0:.1f}s")
    return out
