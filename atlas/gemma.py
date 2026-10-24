"""Gemini client: throttle, retry, cache, validation.

Gemma 4 through the Gemini Developer API (NOTES.md "P0.2"; SPEC.md §7.9). One call carries
system_instruction + optional PNG + JSON mode + response_schema. `generate` never raises:
a cache hit costs nothing, everything else is throttled, retried and validated, and a
failure comes back as None so the pipeline can carry on without an explanation.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Callable, TypeVar

import httpx
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, ValidationError

from atlas.config import GemmaConfig, get_api_key
from atlas.prompts import SUMMARY_SYSTEM, cluster_user_prompt, summary_user_prompt
from atlas.schema import AtlasSummary, ClusterInfo, GemmaClusterReport

T = TypeVar("T", bound=BaseModel)

TIMEOUT_MS = 90_000                # 31b calls were seen hanging for ~88 s
RETRY_CODES = {429, 500, 503, 504}  # rate limit, internal error, "high demand", server deadline
BACKOFF_S = (2, 4, 8, 16)
_FENCE = re.compile(r"^```[A-Za-z]*\s*(.*?)\s*```$", re.DOTALL)


def _parse(text: str, schema: type[T]) -> T:
    """Strip a ```json fence if there is one, then validate against the schema."""
    text = text.strip()
    fenced = _FENCE.match(text)
    if fenced:
        text = fenced.group(1)
    return schema.model_validate_json(text)


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, errors.APIError):
        return exc.code in RETRY_CODES
    return isinstance(exc, httpx.TransportError)  # timeouts, dropped connections


def _one_line(exc: Exception, limit: int = 200) -> str:
    """Short single-line description of an error (for warnings and the validation retry)."""
    if isinstance(exc, ValidationError):
        detail = "; ".join(
            f"{'.'.join(map(str, e['loc'])) or '(root)'}: {e['msg']}" for e in exc.errors(include_url=False)
        )
    else:
        detail = str(exc)
    return f"{type(exc).__name__}: {' '.join(detail.split())}"[:limit]


class GemmaClient:
    def __init__(
        self,
        cfg: GemmaConfig,
        api_key: str | None = None,
        raw_dir: Path | None = None,
        client=None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.cfg = cfg
        self.raw_dir = Path(raw_dir) if raw_dir is not None else None
        self._api_key = api_key
        self._client = client            # injected fake in tests; else created on first real call
        self._sleep = sleep
        self._clock = clock
        self._last_call: float | None = None

    # ------------------------------------------------------------ public

    def generate(
        self,
        system: str,
        user: str,
        schema: type[T],
        image: bytes | None = None,
        tag: str = "call",
    ) -> T | None:
        """One validated Gemma call. Returns the parsed model, or None (warning printed) on failure."""
        cache_path = self.cfg.cache_dir / f"{self._cache_key(system, user, schema, image)}.json"
        hit = self._read_cache(cache_path, schema)
        if hit is not None:
            result, text = hit
            self._write_raw(tag, text)
            return result

        try:
            text = self._request(system, user, image, schema, tag)
            try:
                result = _parse(text, schema)
            except ValidationError as err:
                retry_user = (
                    f"{user}\n\nYour previous reply was not valid ({_one_line(err, 500)}). "
                    "Return only a JSON object that matches the schema."
                )
                text = self._request(system, retry_user, image, schema, tag)
                result = _parse(text, schema)
        except Exception as exc:  # never raise to the pipeline
            print(f"gemma[{tag}]: no result ({_one_line(exc)})")
            return None

        self._write_cache(cache_path, text)
        return result

    # ------------------------------------------------------------ API call

    def _request(self, system: str, user: str, image: bytes | None, schema: type[T], tag: str) -> str:
        """One logical request: throttled, retried on 429/500/503/504 and transport errors with backoff; returns the raw text."""
        client = self._get_client()
        contents: list = [user]
        if image is not None:
            contents.insert(0, types.Part.from_bytes(data=image, mime_type="image/png"))
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=self.cfg.temperature,
            response_mime_type="application/json",
            response_schema=schema,
        )
        for delay in (*BACKOFF_S, None):
            self._throttle()
            try:
                resp = client.models.generate_content(
                    model=self.cfg.model, contents=contents, config=config
                )
            except Exception as exc:
                if delay is None or not _is_retryable(exc):
                    raise
                print(f"gemma[{tag}]: {_one_line(exc, 100)}; retrying in {delay}s")
                self._sleep(delay)
                continue
            text = resp.text or ""
            self._write_raw(tag, text)
            return text
        raise AssertionError("unreachable")  # the last iteration returns or raises

    def _get_client(self):
        if self._client is None:
            key = self._api_key or get_api_key()
            if not key:
                raise RuntimeError("no API key (set GEMMA4_API_KEY)")
            self._client = genai.Client(
                api_key=key, http_options=types.HttpOptions(timeout=TIMEOUT_MS)
            )
        return self._client

    def _throttle(self) -> None:
        """Keep at least 60/rpm seconds between real API calls (rpm <= 0 disables)."""
        if self.cfg.rpm > 0 and self._last_call is not None:
            wait = 60.0 / self.cfg.rpm - (self._clock() - self._last_call)
            if wait > 0:
                self._sleep(wait)
        self._last_call = self._clock()

    # ------------------------------------------------------------ cache / raw files

    def _cache_key(self, system: str, user: str, schema: type[BaseModel], image: bytes | None) -> str:
        h = hashlib.sha256()
        for part in (self.cfg.model.encode(), system.encode(), user.encode(), schema.__name__.encode(), image or b""):
            h.update(len(part).to_bytes(8, "big"))  # length prefix: no ambiguity between fields
            h.update(part)
        return h.hexdigest()

    @staticmethod
    def _read_cache(path: Path, schema: type[T]) -> tuple[T, str] | None:
        """Cached (parsed, raw text), or None on a miss, a corrupt file or a stale schema."""
        try:
            text = json.loads(path.read_text(encoding="utf-8"))["text"]
            return _parse(text, schema), text
        except (OSError, ValueError, KeyError, TypeError, AttributeError):  # ValidationError is a ValueError
            return None

    @staticmethod
    def _write_cache(path: Path, text: str) -> None:
        """Only validated responses are cached, so a failed call is retried on the next run."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"text": text}), encoding="utf-8")
        except OSError as exc:
            print(f"gemma: cache write failed ({_one_line(exc)})")

    def _write_raw(self, tag: str, text: str) -> None:
        if self.raw_dir is None:
            return
        try:
            self.raw_dir.mkdir(parents=True, exist_ok=True)
            safe_tag = re.sub(r"[^A-Za-z0-9_.-]", "_", tag)
            (self.raw_dir / f"{safe_tag}.json").write_text(text, encoding="utf-8")
        except OSError as exc:
            print(f"gemma: raw write failed ({_one_line(exc)})")


# ---------------------------------------------------------------- pipeline entry points

def explain_cluster(
    gc: GemmaClient,
    cluster: ClusterInfo,
    image: bytes | None,
    n_gated: int,
    n_windows: int,
    system: str,
    feature_docs: dict[str, str],
) -> GemmaClusterReport | None:
    """Gemma's report for one cluster: failure type, name, evidence, causes, fixes."""
    user = cluster_user_prompt(cluster, n_gated, n_windows, feature_docs)
    return gc.generate(system, user, GemmaClusterReport, image=image, tag=f"cluster_{cluster.cluster_id}")


def summarize_atlas(gc: GemmaClient, clusters: list[ClusterInfo]) -> AtlasSummary | None:
    """Executive summary over the clusters that have a report (no call if there are none)."""
    if not any(c.report is not None for c in clusters):
        return None
    return gc.generate(SUMMARY_SYSTEM, summary_user_prompt(clusters), AtlasSummary, tag="summary")
