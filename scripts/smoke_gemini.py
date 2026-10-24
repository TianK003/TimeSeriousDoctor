"""P0.2 smoke test: how must atlas/gemma.py call Gemma 4 with our API key?

Standalone (does not import atlas). For each model it runs five checks against the
real API (text, system_instruction, image, json_mode, response_schema), prints one
line per check, and ends with a markdown table to paste into NOTES.md.

    python scripts/smoke_gemini.py --env-file path/to/.env

The API key is read from GEMMA4_API_KEY (or GEMINI_API_KEY) and is never printed.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from google import genai  # noqa: E402
from google.genai import errors, types  # noqa: E402
from pydantic import BaseModel  # noqa: E402

DEFAULT_MODELS = ["gemma-4-26b-a4b-it", "gemma-4-31b-it"]
BACKENDS = [
    ("Gemini Developer API", "genai.Client(api_key=key)", {}),
    ("Vertex AI express", "genai.Client(vertexai=True, api_key=key)", {"vertexai": True}),
]
NOTE_MAX = 150


class Probe(BaseModel):
    failure_type: Literal["delayed_response", "missed_peak", "other"]
    confidence: float


def make_chart_png() -> bytes:
    """Small line chart (~400x300) with one obvious spike, generated in memory."""
    fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
    xs = list(range(30))
    ax.plot(xs, [10 + (25 if x == 20 else 0) + (x % 3) for x in xs], color="tab:red")
    ax.set_title("Weekly demand")
    ax.set_xlabel("week")
    ax.set_ylabel("units")
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return buf.getvalue()


def usage(resp: types.GenerateContentResponse) -> str:
    """Token counts as 'in/out/think' so we can see whether Gemma thinks by default."""
    u = resp.usage_metadata
    if u is None:
        return "tok=?"
    return f"tok={u.prompt_token_count}/{u.candidates_token_count}/{u.thoughts_token_count or 0}"


# ---- checks: each returns a short note on success and raises on failure -------------


def check_text(client: genai.Client, model: str) -> str:
    resp = client.models.generate_content(
        model=model, contents="Reply with the single word: pong"
    )
    return f"reply={resp.text.strip()[:40]!r}; {usage(resp)}"


def check_system_instruction(client: genai.Client, model: str) -> str:
    resp = client.models.generate_content(
        model=model,
        contents="Reply with the single word: pong",
        config=types.GenerateContentConfig(
            system_instruction="You always answer in UPPERCASE letters only."
        ),
    )
    text = resp.text.strip()
    honoured = text.isupper()
    verdict = "honoured" if honoured else "NOT honoured"
    return f"reply={text[:40]!r}; instruction {verdict}; {usage(resp)}"


def check_image(client: genai.Client, model: str) -> str:
    png = make_chart_png()
    resp = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=png, mime_type="image/png"),
            "Describe the chart in one sentence.",
        ],
    )
    return f"{usage(resp)} {resp.text.strip()}"


def check_json_mode(client: genai.Client, model: str) -> str:
    resp = client.models.generate_content(
        model=model,
        contents=(
            'Return a JSON object {"answer": str, "confidence": float} answering: '
            "what is the capital of France?"
        ),
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )
    raw = resp.text.strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        fenced = raw.startswith("```")
        raise ValueError(f"json.loads failed (code fence={fenced}): {raw[:60]!r}") from exc
    keys = sorted(data) if isinstance(data, dict) else type(data).__name__
    return f"parsed keys={keys}; raw fenced={raw.startswith('```')}; {usage(resp)}"


def check_response_schema(client: genai.Client, model: str) -> str:
    resp = client.models.generate_content(
        model=model,
        contents=(
            "A forecast reacted two days late to a demand surge. Classify the failure "
            "and give your confidence between 0 and 1."
        ),
        config=types.GenerateContentConfig(
            response_mime_type="application/json", response_schema=Probe
        ),
    )
    probe = Probe.model_validate_json(resp.text)
    parsed_ok = isinstance(resp.parsed, Probe)
    return f"{probe.failure_type} {probe.confidence}; resp.parsed is Probe={parsed_ok}; {usage(resp)}"


def check_combined(client: genai.Client, model: str) -> str:
    """The real atlas/gemma.py call: image + system_instruction + temperature + schema."""
    resp = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=make_chart_png(), mime_type="image/png"),
            "The chart shows weekly demand; the forecast reacted late to the spike. "
            "Classify the failure and give your confidence between 0 and 1.",
        ],
        config=types.GenerateContentConfig(
            system_instruction="You are a senior forecasting analyst. Return only JSON.",
            temperature=0.2,
            response_mime_type="application/json",
            response_schema=Probe,
        ),
    )
    probe = Probe.model_validate_json(resp.text)
    parsed = isinstance(resp.parsed, Probe)
    return f"{probe.failure_type} {probe.confidence}; resp.parsed is Probe={parsed}; {usage(resp)}"


CHECKS: list[tuple[str, Callable[[genai.Client, str], str]]] = [
    ("text", check_text),
    ("system_instruction", check_system_instruction),
    ("image", check_image),
    ("json_mode", check_json_mode),
    ("response_schema", check_response_schema),
    ("combined", check_combined),
]


# ---- runner ------------------------------------------------------------------------


def run_check(
    fn: Callable[[genai.Client, str], str], client: genai.Client, model: str, key: str
) -> tuple[bool, float, str]:
    """Run one check, timing it. A single 429 is retried after a 10 s backoff."""
    t0 = time.perf_counter()
    try:
        try:
            note, ok = fn(client, model), True
        except errors.APIError as exc:
            if exc.code != 429:
                raise
            time.sleep(10)
            t0 = time.perf_counter()
            note, ok = fn(client, model), True
    except Exception as exc:  # noqa: BLE001 - we want every failure as a table row
        note, ok = f"{type(exc).__name__}: {exc}", False
    latency = time.perf_counter() - t0
    note = " ".join(note.replace(key, "<key>").split())
    return ok, latency, note[: NOTE_MAX - 1] + "…" if len(note) > NOTE_MAX else note


def detect_backend(key: str, model: str) -> tuple[str, str, genai.Client] | None:
    """Return (backend name, client code, client) for the first backend that takes the key.

    Only auth/permission/not-found errors move on to the next backend; a 429/5xx means
    the backend accepted the key and the model was just busy.
    """
    for name, code, kwargs in BACKENDS:
        client = genai.Client(api_key=key, **kwargs)
        t0 = time.perf_counter()
        try:
            note = check_text(client, model)
            accepted = True
        except errors.APIError as exc:
            note = " ".join(f"{exc.code} {exc.status}: {exc.message}".replace(key, "<key>").split())
            accepted = exc.code not in (400, 401, 403, 404)
        except Exception as exc:  # noqa: BLE001
            note, accepted = f"{type(exc).__name__}: {exc}".replace(key, "<key>"), False
        print(f"backend probe | {name} | {'OK' if accepted else 'FAIL'} | "
              f"{time.perf_counter() - t0:.1f}s | {note[:NOTE_MAX]}")
        if accepted:
            return name, code, client
    return None


def list_gemma_models(client: genai.Client, key: str) -> list[str]:
    try:
        names = [m.name or "" for m in client.models.list()]
    except Exception as exc:  # noqa: BLE001
        print(f"model listing failed: {type(exc).__name__}: {str(exc).replace(key, '<key>')[:NOTE_MAX]}")
        return []
    gemma = sorted(n for n in names if "gemma" in n.lower())
    print(f"model listing OK ({len(names)} models); gemma models: {gemma}")
    return gemma


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--env-file", default=".env")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--checks", nargs="+", default=[n for n, _ in CHECKS], help="subset to run")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    load_dotenv(args.env_file)
    key = os.environ.get("GEMMA4_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        print(f"No GEMMA4_API_KEY / GEMINI_API_KEY found (env file: {args.env_file}).")
        return 1

    found = detect_backend(key, args.models[0])
    if found is None:
        print("No backend worked with this key; see the probe errors above.")
        return 2
    backend, client_code, client = found
    print(f"\nWORKING BACKEND: {backend}  ->  client = {client_code}")
    listed = list_gemma_models(client, key)
    for model in args.models:
        if listed and f"models/{model}" not in listed and model not in listed:
            print(f"warning: {model} is not in the model listing")

    rows: list[tuple[str, str, bool, float, str]] = []
    for model in args.models:
        for name, fn in CHECKS:
            if name not in args.checks:
                continue
            ok, latency, note = run_check(fn, client, model, key)
            rows.append((model, name, ok, latency, note))
            print(f"{model} | {name} | {'OK' if ok else 'FAIL'} | {latency:.1f}s | {note}")

    print(f"\nBackend: {backend} (`{client_code}`)\n")
    print("| model | check | result | latency (s) | note |")
    print("|---|---|---|---|---|")
    for model, name, ok, latency, note in rows:
        note = note.replace("|", "\\|")
        print(f"| {model} | {name} | {'OK' if ok else 'FAIL'} | {latency:.1f} | {note} |")
    return 0 if all(r[2] for r in rows) else 3


if __name__ == "__main__":
    raise SystemExit(main())
