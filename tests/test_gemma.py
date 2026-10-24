import json
from types import SimpleNamespace

import httpx
import pytest
from google.genai import errors

from atlas import gemma
from atlas.config import GemmaConfig
from atlas.gemma import GemmaClient, explain_cluster, summarize_atlas
from atlas.prompts import system_prompt
from atlas.schema import AtlasSummary, ClusterInfo, GemmaClusterReport

PNG = b"\x89PNG fake image bytes"

REPORT = {
    "failure_type": "delayed_response",
    "name": "Late tracking of trends",
    "summary": "The forecast follows the actuals a few steps late.",
    "evidence": ["best_lag is 3.00", "slope_corr is 0.60"],
    "likely_causes": [{"cause": "Lag-heavy features", "why": "The model reacts after the change."}],
    "suggested_fixes": ["Model growth rates instead of levels"],
    "metric_to_monitor": "Peak cross-correlation lag",
    "confidence": 0.9,
}
REPORT_JSON = json.dumps(REPORT)


class FakeModels:
    """Replays a script of responses: a str is the reply text, an Exception is raised."""

    def __init__(self, script, on_call=None):
        self.script = list(script)
        self.calls = []
        self.on_call = on_call  # e.g. advance the fake clock to simulate a slow call

    def generate_content(self, model, contents, config):
        self.calls.append(SimpleNamespace(model=model, contents=contents, config=config))
        if self.on_call is not None:
            self.on_call()
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return SimpleNamespace(text=item)


class FakeClient:
    def __init__(self, script, on_call=None):
        self.models = FakeModels(script, on_call)


class FakeTime:
    """Fake clock: sleeping advances it, so throttle and backoff are observable and instant."""

    def __init__(self):
        self.now = 1000.0
        self.sleeps = []

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds

    def monotonic(self):
        return self.now

    def sleep_silently(self, seconds):
        self.now += seconds  # time passes, but it is not a backoff/throttle sleep


def api_error(code):
    return errors.APIError(code, {"error": {"code": code, "message": "boom", "status": "ERR"}})


def make_gc(tmp_path, script, rpm=0, raw_dir=None, call_cost=0.0):
    """`call_cost`: seconds every fake API call takes on the fake clock."""
    ft = FakeTime()
    cfg = GemmaConfig(cache_dir=tmp_path / "cache", rpm=rpm)
    client = FakeClient(script, on_call=(lambda: ft.sleep_silently(call_cost)) if call_cost else None)
    gc = GemmaClient(cfg, client=client, raw_dir=raw_dir, sleep=ft.sleep, clock=ft.monotonic)
    return gc, client.models, ft


def make_cluster(cluster_id=0, report=None) -> ClusterInfo:
    return ClusterInfo(
        cluster_id=cluster_id, size=30, n_series=6, medoid_window_ids=["a#0"],
        feature_median={"abs_err": 2.0, "best_lag": 3.0},
        feature_median_all_gated={"abs_err": 1.5, "best_lag": 0.0},
        model_card_dist={"LightGBM": 1.0}, report=report,
    )


def gen(gc, user="u", image=None, tag="t"):
    return gc.generate("sys", user, GemmaClusterReport, image=image, tag=tag)


# ---------------------------------------------------------------- parsing

def test_valid_json_returns_model(tmp_path):
    gc, models, _ = make_gc(tmp_path, [REPORT_JSON])
    out = gen(gc)
    assert isinstance(out, GemmaClusterReport) and out.failure_type == "delayed_response"
    assert len(models.calls) == 1


def test_fenced_json_is_parsed(tmp_path):
    gc, models, _ = make_gc(tmp_path, [f"```json\n{REPORT_JSON}\n```"])
    assert gen(gc).name == REPORT["name"]
    assert len(models.calls) == 1


def test_invalid_then_valid_uses_one_retry_with_error_appended(tmp_path):
    gc, models, _ = make_gc(tmp_path, ['{"failure_type": "nonsense"}', REPORT_JSON])
    out = gen(gc, user="the user text")
    assert out is not None and len(models.calls) == 2
    first, second = (c.contents[-1] for c in models.calls)
    assert first == "the user text"
    assert second.startswith("the user text") and "not valid" in second and "failure_type" in second


def test_invalid_twice_returns_none_and_is_not_cached(tmp_path, capsys):
    gc, models, _ = make_gc(tmp_path, ["not json", "still not json", REPORT_JSON])
    assert gen(gc) is None
    assert len(models.calls) == 2  # exactly one retry
    assert "no result" in capsys.readouterr().out
    assert gen(gc) is not None  # nothing was cached, so the next run calls again
    assert len(models.calls) == 3


# ---------------------------------------------------------------- call shape

def test_call_shape_system_image_schema_and_model(tmp_path):
    gc, models, _ = make_gc(tmp_path, [REPORT_JSON])
    gen(gc, user="hello", image=PNG)
    call = models.calls[0]
    assert call.model == "gemma-4-26b-a4b-it"
    image_part, text = call.contents
    assert image_part.inline_data.data == PNG and image_part.inline_data.mime_type == "image/png"
    assert text == "hello"
    assert call.config.system_instruction == "sys"
    assert call.config.temperature == 0.2
    assert call.config.response_mime_type == "application/json"
    assert call.config.response_schema is GemmaClusterReport


def test_no_image_sends_text_only(tmp_path):
    gc, models, _ = make_gc(tmp_path, [REPORT_JSON])
    gen(gc, user="hello")
    assert models.calls[0].contents == ["hello"]


# ---------------------------------------------------------------- cache

def test_cache_hit_makes_no_call_and_no_throttle_wait(tmp_path):
    gc, models, ft = make_gc(tmp_path, [REPORT_JSON], rpm=6)
    first = gen(gc)
    second = gen(gc)
    assert second == first
    assert len(models.calls) == 1 and ft.sleeps == []
    # a fresh client over the same cache dir (a re-run) also hits
    gc2, models2, _ = make_gc(tmp_path, [])
    assert gen(gc2) == first and models2.calls == []


def test_cache_key_depends_on_model_system_user_and_image(tmp_path):
    gc, models, _ = make_gc(tmp_path, [REPORT_JSON] * 4)
    gen(gc)
    gc.generate("other system", "u", GemmaClusterReport)
    gen(gc, user="other user")
    gen(gc, image=PNG)
    assert len(models.calls) == 4
    gen(gc, image=PNG)
    assert len(models.calls) == 4  # now all four are cached


def test_corrupt_cache_file_is_ignored(tmp_path):
    gc, models, _ = make_gc(tmp_path, [REPORT_JSON, REPORT_JSON])
    gen(gc)
    for f in (tmp_path / "cache").glob("*.json"):
        f.write_text("{broken", encoding="utf-8")
    assert gen(gc) is not None and len(models.calls) == 2


# ---------------------------------------------------------------- retry / failure

def test_503_then_success_is_retried_with_backoff(tmp_path):
    gc, models, ft = make_gc(tmp_path, [api_error(503), api_error(429), REPORT_JSON])
    assert gen(gc) is not None
    assert len(models.calls) == 3
    assert ft.sleeps == [2, 4]


def test_500_and_504_are_retried_too(tmp_path):
    gc, models, ft = make_gc(tmp_path, [api_error(500), api_error(504), REPORT_JSON])
    assert gen(gc) is not None and ft.sleeps == [2, 2]  # 500: backoff ladder; 504: its single retry


def test_504_is_retried_at_most_once(tmp_path, capsys):
    gc, models, ft = make_gc(tmp_path, [api_error(504), api_error(504), REPORT_JSON])
    assert gen(gc) is None
    assert len(models.calls) == 2 and ft.sleeps == [2]  # not the 2-4-8 ladder
    assert "no result" in capsys.readouterr().out


def test_504_then_503s_still_use_the_backoff_ladder(tmp_path):
    gc, models, ft = make_gc(tmp_path, [api_error(504), api_error(503), api_error(503), api_error(429), REPORT_JSON])
    assert gen(gc) is not None
    assert len(models.calls) == 5 and ft.sleeps == [2, 2, 4, 8]


def test_connection_timeout_is_retried_once(tmp_path):
    gc, models, ft = make_gc(tmp_path, [httpx.ReadTimeout("slow"), REPORT_JSON])
    assert gen(gc) is not None and ft.sleeps == [2]
    assert len(models.calls) == 2


def test_repeated_timeout_gives_up_after_one_retry(tmp_path, capsys):
    gc, models, ft = make_gc(tmp_path, [httpx.ReadTimeout("slow"), httpx.ConnectError("down"), REPORT_JSON])
    assert gen(gc) is None
    assert len(models.calls) == 2 and ft.sleeps == [2]  # not 5 attempts
    assert "no result" in capsys.readouterr().out


def test_persistent_error_returns_none_after_backoff_schedule(tmp_path, capsys):
    gc, models, ft = make_gc(tmp_path, [api_error(503)] * 5)
    assert gen(gc) is None
    assert len(models.calls) == 4  # first attempt + 3 retries
    assert ft.sleeps == [2, 4, 8]
    out = capsys.readouterr().out
    assert "no result" in out and "APIError" in out


def test_http_timeout_is_45_seconds():
    assert gemma.TIMEOUT_MS == 45_000


def test_total_wall_time_per_call_is_capped(tmp_path, capsys):
    # every attempt takes 40 s and fails with 503: attempt 3 ends at t=126 s, past the 120 s cap
    gc, models, ft = make_gc(tmp_path, [api_error(503)] * 10, call_cost=40.0)
    assert gen(gc) is None
    assert len(models.calls) == 3 and ft.sleeps == [2, 4]
    assert "gave up" in capsys.readouterr().out


def test_wall_time_cap_is_shared_with_the_validation_retry(tmp_path, capsys):
    gc, models, ft = make_gc(tmp_path, ["not json", REPORT_JSON], call_cost=130.0)
    assert gen(gc) is None
    assert len(models.calls) == 1  # no second request: the budget is already spent
    assert "gave up" in capsys.readouterr().out


def test_non_retryable_error_fails_fast(tmp_path):
    gc, models, ft = make_gc(tmp_path, [api_error(400), REPORT_JSON])
    assert gen(gc) is None
    assert len(models.calls) == 1 and ft.sleeps == []


def test_missing_api_key_returns_none(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("atlas.gemma.get_api_key", lambda: None)
    gc = GemmaClient(GemmaConfig(cache_dir=tmp_path / "cache"))
    assert gen(gc) is None
    assert "no API key" in capsys.readouterr().out


# ---------------------------------------------------------------- throttle / raw

def test_throttle_spaces_real_calls_by_60_over_rpm(tmp_path):
    gc, models, ft = make_gc(tmp_path, [REPORT_JSON] * 3, rpm=10)
    gen(gc, user="a")
    assert ft.sleeps == []  # first call is not delayed
    gen(gc, user="b")
    gen(gc, user="c")
    assert ft.sleeps == [pytest.approx(6.0), pytest.approx(6.0)]
    assert len(models.calls) == 3


def test_raw_response_written_per_tag(tmp_path):
    raw = tmp_path / "gemma_raw"
    gc, _, _ = make_gc(tmp_path, [REPORT_JSON], raw_dir=raw)
    gen(gc, tag="cluster_0")
    assert json.loads((raw / "cluster_0.json").read_text(encoding="utf-8")) == REPORT
    # a cache hit still leaves the raw file in a fresh run folder
    raw2 = tmp_path / "gemma_raw2"
    gc2, models2, _ = make_gc(tmp_path, [], raw_dir=raw2)
    gen(gc2, tag="cluster_0")
    assert models2.calls == [] and (raw2 / "cluster_0.json").exists()


# ---------------------------------------------------------------- pipeline entry points

def test_explain_cluster_builds_prompts_and_tags_raw(tmp_path):
    raw = tmp_path / "raw"
    gc, models, _ = make_gc(tmp_path, [REPORT_JSON], raw_dir=raw)
    system = system_prompt("TAXONOMY-TEXT", {"abs_err": "error"})
    report = explain_cluster(gc, make_cluster(cluster_id=3), PNG, 50, 400, system, {"abs_err": "error"})
    assert isinstance(report, GemmaClusterReport)
    call = models.calls[0]
    assert call.config.system_instruction == system and "TAXONOMY-TEXT" in system
    assert "Cluster 3: 30 windows from 6 series" in call.contents[1]
    assert "| abs_err | 2.00 | 1.50 |" in call.contents[1]
    assert (raw / "cluster_3.json").exists()


def test_summarize_atlas_uses_clusters_with_reports(tmp_path):
    summary = {"headline": "Lag dominates", "summary": "Most windows lag.", "actions": ["a", "b", "c"]}
    gc, models, _ = make_gc(tmp_path, [json.dumps(summary)])
    clusters = [make_cluster(0, GemmaClusterReport.model_validate(REPORT)), make_cluster(1)]
    out = summarize_atlas(gc, clusters)
    assert isinstance(out, AtlasSummary) and out.headline == "Lag dominates"
    user = models.calls[0].contents[-1]
    assert '"cluster_id": 0' in user and '"cluster_id": 1' not in user
    assert models.calls[0].config.response_schema is AtlasSummary


def test_summarize_atlas_without_reports_makes_no_call(tmp_path):
    gc, models, _ = make_gc(tmp_path, [])
    assert summarize_atlas(gc, [make_cluster(0)]) is None
    assert models.calls == []
