"""Pipeline configuration: pydantic models, YAML loading, API key lookup."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel


class GateConfig(BaseModel):
    mode: Literal["threshold", "quantile"] = "threshold"
    t_abs: float = 1.0
    t_bias: float = 0.75
    q: float = 0.25


class ClusterConfig(BaseModel):
    min_cluster_size: int = 20
    max_clusters: int = 8
    seed: int = 0


class MomentConfig(BaseModel):
    model_size: str = "small"
    batch_size: int = 64
    device: str = "cpu"


class GemmaConfig(BaseModel):
    model: str = "gemma-4-26b-a4b-it"
    fallback_model: str = "gemma-4-31b-it"
    rpm: int = 10
    temperature: float = 0.2
    cache_dir: Path = Path("cache/gemma")


class Config(BaseModel):
    season: int = 7
    window_len: int = 28
    stride: int = 14
    min_history: int = 56
    max_lag: int = 7
    embedder: Literal["features", "moment", "both"] = "features"
    n_baseline: int = 40
    cache_dir: Path = Path("cache")
    runs_dir: Path = Path("runs")
    gate: GateConfig = GateConfig()
    cluster: ClusterConfig = ClusterConfig()
    moment: MomentConfig = MomentConfig()
    gemma: GemmaConfig = GemmaConfig()


def load_config(path: Path | None = None) -> Config:
    """Load a Config from a YAML file; None gives the defaults."""
    if path is None:
        return Config()
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return Config.model_validate(raw)


def get_api_key() -> str | None:
    """Return the Gemma/Gemini API key from the environment (or a local .env), else None."""
    load_dotenv()
    return os.environ.get("GEMMA4_API_KEY") or os.environ.get("GEMINI_API_KEY")
