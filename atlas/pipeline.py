"""run_atlas(): end-to-end pipeline.

Owner: A. Task: P0.14. Spec: SPEC.md §7.10.
"""

from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from atlas.cluster import cluster_windows, embed_features, project_2d, summarize_clusters
from atlas.config import Config
from atlas.features import FEATURE_DOCS, FEATURES, compute_features
from atlas.gate import add_scores, gate_windows
from atlas.gemma import GemmaClient, explain_cluster, summarize_atlas
from atlas.plots import cluster_panel_png
from atlas.prompts import MAX_EXPERT_EXAMPLES, system_prompt
from atlas.residuals import compute_scaled_residuals, make_windows, window_matrix
from atlas.schema import (
    CLUSTER_NOISE,
    CLUSTER_NOT_GATED,
    RunResult,
    load_dataset,
    save_run,
)
from atlas.taxonomy import load_taxonomy, render_taxonomy


@contextmanager
def _stage(name: str):
    t0 = time.perf_counter()
    yield
    print(f"stage: {name} … {time.perf_counter() - t0:.1f}s")


EXPERT_STATUS_ORDER = {"confirmed": 0, "renamed": 1, "rejected": 2}


def load_expert_examples(expert_from: Path) -> list[dict]:
    """Reviewed diagnoses from a previous run's expert_feedback.json: max 3, confirmed first.

    Entries with an unknown status are ignored; a missing file gives no examples (with a note).
    """
    path = Path(expert_from) / "expert_feedback.json"
    if not path.is_file():
        print(f"expert feedback: no {path.name} in {expert_from}, running without expert examples")
        return []
    feedback = json.loads(path.read_text(encoding="utf-8"))
    entries = [e for e in feedback.values() if isinstance(e, dict) and e.get("status") in EXPERT_STATUS_ORDER]
    entries.sort(key=lambda e: EXPERT_STATUS_ORDER[e["status"]])  # stable: file order within a status
    return entries[:MAX_EXPERT_EXAMPLES]


def run_atlas(
    data_dir: Path,
    cfg: Config,
    run_name: str,
    use_gemma: bool = True,
    gemma_client=None,
    taxonomy_path: Path = Path("taxonomy.yaml"),
    expert_from: Path | None = None,
) -> RunResult:
    """Load -> residuals -> windows -> gate -> features -> embed -> cluster -> panels -> Gemma -> save.

    The run folder is written to `cfg.runs_dir / run_name`. `gemma_client` replaces the real
    GemmaClient (tests); with `use_gemma=False` the clusters carry no report and there is no summary.
    `expert_from` is an earlier run folder whose expert_feedback.json supplies reviewed diagnoses
    (max 3) as few-shot examples in every cluster prompt (SPEC P2.2); nothing else is copied.
    """
    if cfg.embedder != "features":
        raise NotImplementedError("MOMENT embedder not available yet; use --embedder features")
    data_dir = Path(data_dir)
    run_dir = Path(cfg.runs_dir) / run_name

    with _stage("load"):
        ds = load_dataset(data_dir, cfg.min_history, cfg.window_len)
    with _stage("residuals"):
        res = compute_scaled_residuals(ds, cfg.season)
    with _stage("windows"):
        windows = make_windows(res, cfg.window_len, cfg.stride)
    with _stage("gate"):
        windows = add_scores(windows, res)
        windows["gated"] = gate_windows(windows, cfg.gate)
    with _stage("features"):
        r = window_matrix(res, windows, "r_s")
        a = window_matrix(res, windows, "a_s")
        p = window_matrix(res, windows, "p_s")
        a_ctx = window_matrix(res, windows, "a_s", before=cfg.max_lag, after=cfg.max_lag)
        windows[FEATURES] = compute_features(r, a, p, a_ctx, cfg.max_lag).to_numpy()

    gated_mask = windows["gated"].to_numpy(dtype=bool)
    n_windows, n_gated = len(windows), int(gated_mask.sum())

    with _stage("embed"):
        if n_gated:
            X = embed_features(windows[gated_mask], FEATURES)
        else:
            X = np.zeros((0, len(FEATURES)), dtype=np.float32)
    with _stage("cluster"):
        labels = cluster_windows(X, cfg.cluster.min_cluster_size, cfg.cluster.max_clusters, cfg.cluster.seed)
        emb_row = np.full(n_windows, -1, dtype=int)
        emb_row[gated_mask] = np.arange(n_gated)
        cluster = np.full(n_windows, CLUSTER_NOT_GATED, dtype=int)
        cluster[gated_mask] = labels
        xy = np.full((n_windows, 2), np.nan, dtype=np.float32)
        xy[gated_mask] = project_2d(X)
        windows["emb_row"] = emb_row
        windows["cluster"] = cluster
        windows["x2d"] = xy[:, 0]
        windows["y2d"] = xy[:, 1]
    with _stage("summarize"):
        clusters = summarize_clusters(windows[gated_mask], labels, X, ds.series_meta, FEATURES)

    with _stage("panels"):
        image_dir = run_dir / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        for c in clusters:
            rel = f"images/cluster_{'noise' if c.cluster_id == CLUSTER_NOISE else c.cluster_id}.png"
            cluster_panel_png(c, windows, res, run_dir / rel)
            c.image_path = rel

    summary = None
    n_reports = 0
    if use_gemma:
        with _stage("gemma"):
            gc = gemma_client or GemmaClient(cfg.gemma, raw_dir=run_dir / "gemma_raw")
            system = system_prompt(render_taxonomy(load_taxonomy(taxonomy_path)), FEATURE_DOCS)
            expert_examples = load_expert_examples(expert_from) if expert_from is not None else None
            if expert_examples:
                print(f"expert feedback: {len(expert_examples)} reviewed diagnoses from {expert_from} in every prompt")
            for c in clusters:
                if c.cluster_id < 0:
                    continue
                image = (run_dir / c.image_path).read_bytes()
                c.report = explain_cluster(gc, c, image, n_gated, n_windows, system, FEATURE_DOCS, expert_examples)
            n_reports = sum(c.report is not None for c in clusters)
            summary = summarize_atlas(gc, clusters)

    run = RunResult(
        name=run_name,
        config={**cfg.model_dump(mode="json"), "data_dir": str(data_dir)},
        data=ds.data,
        series_meta=ds.series_meta,
        windows=windows,
        embeddings=X.astype(np.float32),
        clusters=clusters,
        summary=summary,
    )
    with _stage("save"):
        save_run(run, run_dir)

    n_explainable = sum(c.cluster_id >= 0 for c in clusters)
    gemma_note = f"{n_reports}/{n_explainable}" if use_gemma else "skipped"
    print(
        f"run {run_name}: {n_windows} windows, {n_gated} gated, "
        f"{n_explainable} clusters, Gemma reports {gemma_note} -> {run_dir}"
    )
    return run
