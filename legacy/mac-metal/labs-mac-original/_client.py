"""Shared stdlib-only client for the labs.

No third-party dependencies on purpose: the `openai` package may or may not be
installed, and the raw HTTP surface is what we actually want to see.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

DEFAULT_BASE = "http://127.0.0.1:8000"


def _post(base_url: str, path: str, payload: dict, timeout: float = 300.0):
    req = urllib.request.Request(
        f"{base_url}{path}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(req, timeout=timeout)


def get_json(base_url: str, path: str, timeout: float = 30.0) -> dict:
    with urllib.request.urlopen(f"{base_url}{path}", timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def get_text(base_url: str, path: str, timeout: float = 30.0) -> str:
    with urllib.request.urlopen(f"{base_url}{path}", timeout=timeout) as resp:
        return resp.read().decode()


def list_models(base_url: str = DEFAULT_BASE) -> list[str]:
    return [m["id"] for m in get_json(base_url, "/v1/models")["data"]]


def wait_for_server(base_url: str = DEFAULT_BASE, timeout: float = 600.0) -> None:
    """Block until the server answers /v1/models, or raise after `timeout` seconds."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            list_models(base_url)
            return
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            time.sleep(1.0)
    raise TimeoutError(f"{base_url} did not become ready within {timeout:.0f}s")


def chat(
    prompt: str,
    base_url: str = DEFAULT_BASE,
    model: str | None = None,
    system: str | None = None,
    **sampling,
) -> tuple[str, float]:
    """One non-streaming chat request. Returns (text, elapsed_seconds)."""
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    payload = {
        "model": model or list_models(base_url)[0],
        "messages": messages,
        "max_tokens": sampling.pop("max_tokens", 64),
        **sampling,
    }
    start = time.perf_counter()
    with _post(base_url, "/v1/chat/completions", payload) as resp:
        body = json.loads(resp.read().decode())
    return body["choices"][0]["message"]["content"], time.perf_counter() - start


def chat_raw(
    prompt: str,
    base_url: str = DEFAULT_BASE,
    model: str | None = None,
    **sampling,
) -> tuple[str, float, dict]:
    """Like `chat`, but also returns the raw `usage` block (token counts)."""
    payload = {
        "model": model or list_models(base_url)[0],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": sampling.pop("max_tokens", 64),
        **sampling,
    }
    start = time.perf_counter()
    with _post(base_url, "/v1/chat/completions", payload) as resp:
        body = json.loads(resp.read().decode())
    return (
        body["choices"][0]["message"]["content"],
        time.perf_counter() - start,
        body.get("usage", {}),
    )


def completion(prompt: str, base_url: str = DEFAULT_BASE, model: str | None = None, **sampling):
    """Raw completion endpoint (no chat template). Returns (text, elapsed_seconds)."""
    payload = {
        "model": model or list_models(base_url)[0],
        "prompt": prompt,
        "max_tokens": sampling.pop("max_tokens", 64),
        **sampling,
    }
    start = time.perf_counter()
    with _post(base_url, "/v1/completions", payload) as resp:
        body = json.loads(resp.read().decode())
    return body["choices"][0]["text"], time.perf_counter() - start


def chat_stream(
    prompt: str,
    base_url: str = DEFAULT_BASE,
    model: str | None = None,
    system: str | None = None,
    **sampling,
):
    """Streaming chat. Yields (text_so_far, ttft, elapsed) then returns.

    `ttft` is filled in from the first chunk that carries content; before that it
    is None. Useful for measuring time-to-first-token accurately.
    """
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    payload = {
        "model": model or list_models(base_url)[0],
        "messages": messages,
        "max_tokens": sampling.pop("max_tokens", 64),
        "stream": True,
        **sampling,
    }
    start = time.perf_counter()
    ttft: float | None = None
    text = ""
    with _post(base_url, "/v1/chat/completions", payload) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            delta = chunk["choices"][0].get("delta", {})
            piece = delta.get("content") or ""
            if piece and ttft is None:
                ttft = time.perf_counter() - start
            text += piece
            yield text, ttft, time.perf_counter() - start


def measure_ttft(
    prompt: str, base_url: str = DEFAULT_BASE, system: str | None = None, **sampling
) -> tuple[float, float, str]:
    """Return (ttft_seconds, total_seconds, text) for one streaming request."""
    ttft = total = 0.0
    text = ""
    for text, ttft_v, total_v in chat_stream(
        prompt, base_url=base_url, system=system, **sampling
    ):
        if ttft_v is not None and not ttft:
            ttft = ttft_v
        total = total_v
    return ttft, total, text


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    line = "  ".join("-" * w for w in widths)
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print(line)
    for row in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(row, widths)))
