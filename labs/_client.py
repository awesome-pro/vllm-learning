"""Shared, standard-library-only HTTP client for the vLLM labs (02-05).

Raw `urllib` on purpose: these labs are about the engine's real HTTP surface,
and neither `requests` nor `openai` is an assumption this project makes. Use it
from a lab in this directory with `import _client as c`.

Environment: `PORT` (8000), `MODEL` (`Qwen/Qwen3-0.6B`), optionally `BASE_URL`.
Written against vLLM v0.30.0: `vllm/entrypoints/openai/` for the HTTP surface,
`vllm/v1/metrics/loggers.py` for the metric names.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

PORT = int(os.environ.get("PORT", "8000"))
MODEL = os.environ.get("MODEL", "Qwen/Qwen3-0.6B")
BASE = os.environ.get("BASE_URL", f"http://127.0.0.1:{PORT}")
SERVE_HINT = "bash scripts/serve.sh"

# Qwen3 emits a long "thinking" preamble by default, which would make "time to
# first *content* token" measure the thinking block instead of the prefill.
# `chat_template_kwargs` is a verified ChatCompletionRequest field (v0.30.0).
NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}


class ServerNotRunning(RuntimeError):
    """Nothing is listening on the endpoint -- say exactly how to fix that."""

    def __init__(self, base: str = BASE) -> None:
        super().__init__(
            f"No vLLM server answering at {base}.\n"
            f"  Start one in another terminal first:  {SERVE_HINT}"
        )


def _request(path: str, base: str, timeout: float, payload: dict | None = None):
    """urlopen a path, naming the two failure modes these labs care about."""
    req = urllib.request.Request(
        f"{base}{path}",
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={"Content-Type": "application/json"},
        method="POST" if payload is not None else "GET",
    )
    try:
        return urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as exc:  # subclass of URLError: catch it first
        raise RuntimeError(
            f"HTTP {exc.code} from {req.full_url}: "
            f"{exc.read().decode('utf-8', 'replace')[:300]}"
        ) from exc
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as exc:
        raise ServerNotRunning(base) from exc


def get_text(path: str, base: str = BASE, timeout: float = 10.0) -> str:
    """GET a path and return the body as text (/metrics and /health are text)."""
    with _request(path, base, timeout) as resp:
        return resp.read().decode("utf-8", "replace")


def models(base: str = BASE, timeout: float = 10.0) -> list[str]:
    """The model ids the server is serving, from /v1/models."""
    return [e["id"] for e in json.loads(get_text("/v1/models", base, timeout))["data"]]


def chat(
    prompt: str,
    model: str = MODEL,
    *,
    system: str | None = None,
    max_tokens: int = 64,
    temperature: float = 0.0,
    base: str = BASE,
    timeout: float = 180.0,
    **sampling,
) -> tuple[str, float, dict]:
    """One non-streaming chat completion; returns (text, elapsed, usage).

    `elapsed` is the whole request -- you get nothing until it finishes, so it is
    *not* a TTFT. Extra OpenAI/vLLM sampling fields (`ignore_eos=True`,
    `min_tokens=...`) pass through `**sampling`; `ignore_eos` is verified on
    ChatCompletionRequest (protocol.py, v0.30.0) and fixes the decode length.
    """
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        **NO_THINK,
        **sampling,
    }
    start = time.perf_counter()
    with _request("/v1/chat/completions", base, timeout, payload) as resp:
        body = json.loads(resp.read().decode())
    return (
        body["choices"][0]["message"]["content"] or "",
        time.perf_counter() - start,
        body.get("usage") or {},
    )


def stream_chat(
    prompt: str,
    model: str = MODEL,
    *,
    system: str | None = None,
    max_tokens: int = 64,
    temperature: float = 0.0,
    base: str = BASE,
    timeout: float = 180.0,
    **sampling,
):
    """Generator over a streamed chat completion: yields `(delta_text, t)`.

    `delta_text` is one SSE chunk's text (possibly "") and `t` is the seconds
    elapsed since the request was sent. TTFT is the `t` of the first non-empty
    delta (queueing + tokenization + prefill); the inter-token gaps are the
    differences between successive non-empty deltas (one decode step each --
    vLLM's default `stream_interval` is 1, `vllm/config/scheduler.py`). The final
    `include_usage` chunk carries no choices and is skipped.
    """
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},  # StreamOptions field
        **NO_THINK,
        **sampling,
    }
    start = time.perf_counter()
    with _request("/v1/chat/completions", base, timeout, payload) as resp:
        for raw in resp:  # server-sent events: one `data: {...}` line per chunk
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices") or []
            piece = choices[0].get("delta", {}).get("content") if choices else None
            yield piece or "", time.perf_counter() - start


def measure_stream(prompt: str, **kwargs) -> tuple[float, float, list[float], str]:
    """Consume `stream_chat` and return (ttft, total, gaps, text).

    `gaps` holds the seconds between successive content deltas; `gaps[0]` is the
    first decode step after the first token. One decode step is normally one
    token, so `len(gaps) / (total - ttft)` is the decode-only token rate.
    """
    ttft = total = float("nan")
    gaps: list[float] = []
    previous: float | None = None
    text = ""
    for delta, t in stream_chat(prompt, **kwargs):
        if not delta:
            continue
        if previous is None:
            ttft = t
        else:
            gaps.append(t - previous)
        previous = t
        text += delta
        total = t
    return ttft, total, gaps, text


def _samples(text: str):
    """Yield (name, labels, value) from Prometheus text, dropping `_total`."""
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        head, _, tail = line.rpartition(" ")
        try:
            value = float(tail)
        except ValueError:
            continue
        labels: dict = {}
        if "{" in head:
            name, _, rest = head.partition("{")
            for pair in rest.rstrip("}").split(","):
                key, _, val = pair.partition("=")
                if key:
                    labels[key.strip()] = val.strip().strip('"')
        else:
            name = head
        name = name.strip()
        yield (name[: -len("_total")] if name.endswith("_total") else name), labels, value


def metrics(base: str = BASE, timeout: float = 10.0) -> dict[str, float]:
    """Scrape /metrics once into a dict of name -> value.

    Counters are summed across label sets and the exposition's `_total` suffix is
    stripped, so `m["vllm:generation_tokens"]` works; gauges take the max across
    label sets (one engine here, so that is just the value). Histogram `_bucket`
    lines are deliberately NOT included - summing buckets over `le` would produce a
    meaningless number. Use `quantile()` for a percentile, `labels()` for one label
    set, and the `_count`/`_sum` keys (which are meaningful) for sample counts.
    """
    text = get_text("/metrics", base, timeout)
    kinds = {
        parts[2]: parts[3]
        for parts in (line.split() for line in text.splitlines())
        if len(parts) == 4 and parts[1] == "TYPE"
    }
    out: dict[str, float] = {}
    for name, _, value in _samples(text):
        if name.endswith("_bucket"):
            continue
        if kinds.get(name) == "gauge":
            out[name] = max(out.get(name, float("-inf")), value)
        else:
            out[name] = out.get(name, 0.0) + value
    return out


def labels(metric: str, base: str = BASE, timeout: float = 10.0) -> dict:
    """Merged labels of an info-style gauge such as `vllm:cache_config_info`.

    That gauge comes from `CacheConfig.metrics_info()`, which stringifies every
    field -- including ones still None when the logger is built -- and it can be
    emitted more than once. Real values win over "None"; anything left "None"
    means "read that number out of the server startup log instead".
    """
    out: dict = {}
    for name, lab, _ in _samples(get_text("/metrics", base, timeout)):
        if name == metric:
            for key, value in lab.items():
                if key not in out or out[key] == "None":
                    out[key] = value
    return out


def quantile(metric: str, q: float, base: str = BASE, timeout: float = 10.0) -> float:
    """Prometheus-style quantile from a histogram family, e.g. `..._seconds`.

    Buckets are cumulative, so the answer is only as precise as their boundaries
    (`vllm/v1/metrics/buckets.py`); it interpolates linearly inside the bucket
    holding the rank, exactly like PromQL's `histogram_quantile`.
    """
    buckets: dict[float, float] = {}
    for name, lab, value in _samples(get_text("/metrics", base, timeout)):
        if name == f"{metric}_bucket" and "le" in lab:
            buckets[float(lab["le"])] = buckets.get(float(lab["le"]), 0.0) + value
    items = sorted(buckets.items())
    if not items or items[-1][1] <= 0:
        return float("nan")
    rank, prev_le, prev_count = q * items[-1][1], 0.0, 0.0
    for le, count in items:
        if count >= rank:
            if le == float("inf") or count == prev_count:
                return prev_le
            return prev_le + (le - prev_le) * (rank - prev_count) / (count - prev_count)
        prev_le, prev_count = le, count
    return items[-1][0]


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    """Left-aligned fixed-width table; keeps the labs' output readable."""
    widths = [max(len(str(r[i])) for r in [headers, *rows]) for i in range(len(headers))]
    print("  ".join(str(h).ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(row, widths)))


def record(label: str, **numbers) -> None:
    """Print a copy-ready RECORD block listing what to write into notes/."""
    print("\n" + "=" * 66)
    print(f"RECORD: {label}")
    print("=" * 66)
    for key, value in numbers.items():
        print(f"  {key:<34} {f'{value:,.4g}' if isinstance(value, float) else value}")
    print("-" * 66)
    print("Copy this block into notes/ with the date and the vLLM version.")
