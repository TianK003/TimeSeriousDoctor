from pathlib import Path

from atlas import config
from atlas.config import Config, get_api_key, load_config

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_defaults():
    cfg = Config()
    assert cfg.gate.t_abs == 1.0
    assert cfg.gemma.model == "gemma-4-26b-a4b-it"
    assert cfg.window_len == 28
    assert cfg.embedder == "features"


def test_load_config_none_gives_defaults():
    assert load_config(None) == Config()


def test_default_yaml_matches_defaults():
    assert load_config(REPO_ROOT / "config.default.yaml") == Config()


def test_partial_yaml_overrides_only_what_it_sets(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("window_len: 14\ngate:\n  t_abs: 2.5\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.window_len == 14
    assert cfg.gate.t_abs == 2.5
    assert cfg.gate.t_bias == Config().gate.t_bias
    assert cfg.stride == Config().stride
    assert cfg.gemma == Config().gemma


def test_get_api_key_precedence(monkeypatch):
    monkeypatch.setattr(config, "load_dotenv", lambda *a, **k: False)
    monkeypatch.setenv("GEMMA4_API_KEY", "gemma-key")
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    assert get_api_key() == "gemma-key"

    monkeypatch.delenv("GEMMA4_API_KEY")
    assert get_api_key() == "gemini-key"

    monkeypatch.delenv("GEMINI_API_KEY")
    assert get_api_key() is None
