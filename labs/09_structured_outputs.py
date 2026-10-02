#!/usr/bin/env python3
"""Lab 09 — Stage 8: structured outputs, and what a grammar costs per step.

WHAT IT DEMONSTRATES
    Four arms, each asked to do the SAME task N times (default 10):

      freeform     no constraint at all — can the model just produce the JSON we want?
      json_schema  response_format = {"type": "json_schema", ...}
      regex        structured_outputs = {"regex": ...}
      choice       structured_outputs = {"choice": [...]}

    For each arm the lab counts how many results actually satisfy the constraint and
    reports the latency distribution, so you can see the trade: constrained decoding
    does strictly more work per decode step, and can still win end-to-end because it
    never needs a retry.

    It also demonstrates the trap: the legacy `guided_json` / `guided_regex` /
    `guided_choice` / `guided_grammar` fields were REMOVED in vLLM v0.29/v0.30 and are
    now SILENTLY IGNORED. A request using them returns HTTP 200 with unconstrained
    output and one warning in the server log. See "THE TRAP" below.

RUN
    bash scripts/serve.sh                      # in another terminal — this lab needs a server
    python labs/09_structured_outputs.py

    # knobs
    N=20 python labs/09_structured_outputs.py
    MAX_TOKENS=384 python labs/09_structured_outputs.py
    MODEL=Qwen/Qwen3-4B python labs/09_structured_outputs.py
    SKIP_TRAP=1 python labs/09_structured_outputs.py    # skip the legacy-field probe

PREREQUISITES
    * A running vLLM OpenAI server. Default endpoint http://127.0.0.1:8000
      (override with BASE_URL=...).
    * `labs/_client.py` on the import path — run from the labs/ directory or from the
      repo root; see the sys.path fix below.
    * Standard library only. No `requests`, no `openai`.

ENV KNOBS
    N=10             requests per arm
    MAX_TOKENS=256   cap per request (a free-form answer that hits this is TRUNCATED,
                     which is a real failure mode and part of the result)
    TEMPERATURE=0    determinism; leave it at 0
    PROBE=3          streaming probes per arm used to split TTFT from decode time
                     (set PROBE=0 to skip; costs extra requests)
    SKIP_TRAP=0      set to 1 to skip the 1-request legacy-guided-field probe
    BASE_URL         server root, default from _client.DEFAULT_BASE

-------------------------------------------------------------------------------
VERIFIED FACTS THIS LAB RELIES ON (vLLM v0.30.0, checked against the source tree)

  * The chat request field is `response_format: AnyResponseFormat | None`
    (`vllm/entrypoints/openai/chat_completion/protocol.py:231`), and its json_schema
    variant is `JsonSchemaResponseFormat` in `vllm/entrypoints/generate/base/protocol.py:117-123`:
        name: str
        description: str | None
        json_schema: dict | None = Field(default=None, alias="schema")
        strict: bool | None
    The alias is why the wire format nests the JSON Schema under "schema":
        {"type": "json_schema",
         "json_schema": {"name": ..., "schema": {...}, "strict": true}}
    and `structured_outputs_from_response_format()` (`.../base/protocol.py:161-191`) maps
    it to `StructuredOutputsParams(json=<the schema>)`.

  * The other constraints are NOT top-level `guided_*` fields in this version. They live
    under `structured_outputs` (`.../chat_completion/protocol.py:386-389`), whose type is
    `StructuredOutputsParams` (`vllm/sampling_params.py:88-99`):
        json, regex, choice, grammar, json_object, structural_tag,
        disable_any_whitespace, disable_additional_properties, whitespace_pattern
    Exactly one of the six constraint fields may be set (`__post_init__`, lines 106-127).

  * THE TRAP — `guided_json`, `guided_regex`, `guided_choice`, `guided_grammar`,
    `guided_decoding_backend`, `guided_whitespace_pattern` are listed in
    `_REMOVED_GUIDED_FIELDS` in `vllm/entrypoints/serve/engine/protocol.py:21-34`.
    The docstring above them says it plainly: "Legacy guided-decoding fields are treated
    like any other extra field since their removal, so requests still using them get
    HTTP 200 with unconstrained output and no visible signal (see #53975)." The server
    logs one warning_once. This lab proves it.

  * Default backend is "auto": `StructuredOutputsConfig.backend: StructuredOutputsBackend = "auto"`
    with `StructuredOutputsBackend = Literal["auto", "xgrammar", "guidance", "outlines",
    "lm-format-enforcer"]` (`vllm/config/structured_outputs.py:12-25`).

  * `structured_outputs.choice` is real and is compiled to an EBNF alternation:
    `choice_as_grammar()` builds `root ::= "a" | "b" | ...`
    (`vllm/v1/structured_output/utils.py:438-455`), which the xgrammar backend validates
    with `xgr.Grammar.from_ebnf` and then stores as `.grammar`
    (`vllm/v1/structured_output/backend_xgrammar.py:382-392`). So a choice constraint
    returns ONE of the listed strings, with no quotes.

  * WHERE THE GRAMMAR HOOKS IN — this is the whole point of Stage 8:
      1. the engine core asks the backend for a bitmask each step:
         `StructuredOutputManager.grammar_bitmask()` (`vllm/v1/structured_output/__init__.py:314`),
         which calls `grammar.fill_bitmask(self._grammar_bitmask, index)` for every
         structured-output request in the batch (`_fill_bitmasks`, lines 202-213);
      2. the bitmask is shipped to the worker inside `GrammarOutput` and applied
         IN PLACE to the logits: `apply_grammar_bitmask()`
         (`vllm/v1/worker/gpu/structured_outputs.py:95-156`), called from the model
         runner at `vllm/v1/worker/gpu/model_runner.py:1594-1603`;
      3. the Triton kernel sets every disallowed token's logit to -inf:
         `_apply_grammar_bitmask_kernel` (`vllm/v1/worker/gpu/structured_outputs.py:162-199`),
         with `tl.store(..., -float("inf"), mask=position_is_active & bitmask & ...)` at
         lines 195-199. The bitmask is unpacked one bit per vocab entry at line 190.
    So the mask is recomputed and re-applied at EVERY decode step, for every sequence in
    the batch — it is not a one-off cost at request start. The bitmask also crosses the
    host→device boundary per step (`async_tensor_h2d` on a dedicated copy stream,
    lines 105-109), and the DFA advance itself runs on the CPU in the engine core.

  * Compiled grammars ARE cached: the xgrammar compiler is built with
    `cache_enabled=True, cache_limit_bytes=VLLM_XGRAMMAR_CACHE_MB * 1024 * 1024`
    (`vllm/v1/structured_output/backend_xgrammar.py:74-78`), and
    `VLLM_XGRAMMAR_CACHE_MB` defaults to 512 (`vllm/envs.py:1694`). Regex compilation has
    a timeout, `VLLM_REGEX_COMPILATION_TIMEOUT_S = 5` (`vllm/envs.py:230`, 1695-1698).
    Consequence: the FIRST request with a given grammar pays compilation; later ones hit
    the cache. This lab warms up first so the numbers are not dominated by compilation.

  * `chat_template_kwargs: {"enable_thinking": false}` is a real chat request field
    (`vllm/entrypoints/openai/chat_completion/protocol.py:353-358`). Qwen3 models will
    otherwise emit reasoning text before the answer, which would break JSON parsing for
    reasons that have nothing to do with grammars.
-------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request

# The labs run with labs/ as the working directory, so `_client` is importable directly.
# This fallback keeps `python labs/09_structured_outputs.py` working from the repo root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _client import chat, stream_chat, metrics, record  # noqa: E402

# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
N = int(os.environ.get("N", "10"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "256"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0"))
PROBE = int(os.environ.get("PROBE", "3"))
SKIP_TRAP = os.environ.get("SKIP_TRAP", "0") == "1"

try:
    import _client as _c  # for DEFAULT_BASE only
    BASE = os.environ.get("BASE_URL") or getattr(_c, "DEFAULT_BASE", "http://127.0.0.1:8000")
except Exception:  # pragma: no cover - _client is required, but do not explode here
    BASE = os.environ.get("BASE_URL", "http://127.0.0.1:8000")
MODEL = os.environ.get("MODEL") or None

# `_client.chat` already injects `chat_template_kwargs={"enable_thinking": False}`
# (NO_THINK in labs/_client.py) so Qwen3 does not emit a reasoning preamble before the
# answer. Passing it again here would be redundant, so this lab only sets temperature.
CHAT_KWARGS = {"temperature": TEMPERATURE}

# ---------------------------------------------------------------------------
# the task: one review in, one structured verdict out
# ---------------------------------------------------------------------------
SCHEMA = {
    "type": "object",
    "properties": {
        "sentiment": {
            "type": "string",
            "enum": ["positive", "negative", "mixed", "neutral"],
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "topics": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "maxItems": 3,
        },
        "summary": {"type": "string", "maxLength": 160},
    },
    "required": ["sentiment", "confidence", "topics", "summary"],
    "additionalProperties": False,
}

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "review_verdict",
        # NOTE the alias: the wire key is "schema", not "json_schema".
        # JsonSchemaResponseFormat.json_schema has Field(..., alias="schema")
        # (vllm/entrypoints/generate/base/protocol.py:122).
        "schema": SCHEMA,
        "strict": True,
    },
}

# A compact, machine-readable line format. xgrammar compiles this to a grammar and the
# model may emit nothing else. Anchored implicitly: the grammar must match the whole output.
REGEX = r"(positive|negative|mixed|neutral);(0\.[0-9]{2}|1\.00);[a-z]+(,[a-z]+){0,2}"

CHOICES = ["positive", "negative", "mixed", "neutral"]

REVIEWS = [
    "The battery lasts two days and the screen is gorgeous, but the camera is awful in low light.",
    "Shipped a week late and arrived scratched. Support never answered my email.",
    "It does exactly what the listing says. Nothing more, nothing less.",
    "Fast, quiet, and cheaper than the competition. I bought a second one.",
    "The app crashes every time I try to sync, which makes the whole device useless.",
    "Great hardware let down by mediocre software; I keep going back and forth on it.",
    "Perfectly adequate. I have no strong feelings either way after a month of use.",
    "The keyboard is a joy to type on and the hinge feels solid after a year.",
    "Overheats under any real load, and the fan sounds like a small aircraft.",
    "Battery life is fine, the display is fine, the price is fine. It is fine.",
    "Setup took four minutes and it has not needed attention since.",
    "The update bricked my unit and the rollback instructions do not work.",
    "Comfortable, light, and the battery genuinely lasts all day on a train.",
    "Screen has a dead pixel out of the box; otherwise the build quality is excellent.",
    "It is slower than my old one but the battery more than makes up for it.",
    "Returned it after two days. The touchpad registers clicks I never made.",
    "Excellent value if you only need the basics, frustrating if you need anything else.",
    "The sound is surprisingly good for something this thin.",
    "Firmware updates fixed the worst bugs, so I have warmed up to it.",
    "Loud, hot, and expensive, but it renders in half the time of my old machine.",
]

PROMPT_TEMPLATE = (
    "Classify this product review. Reply with a JSON object with exactly these keys:\n"
    '  "sentiment": one of "positive", "negative", "mixed", "neutral"\n'
    '  "confidence": a number between 0 and 1\n'
    '  "topics": an array of 1 to 3 lowercase single-word topics\n'
    '  "summary": a string of at most 160 characters\n'
    "No other keys, no commentary, no markdown fences.\n\n"
    "Review: {review}"
)

REGEX_PROMPT_TEMPLATE = (
    "Classify this product review. Reply with exactly one line in this format and nothing else:\n"
    "sentiment;confidence;topics\n"
    "where sentiment is positive, negative, mixed or neutral, confidence is a number like "
    "0.85 or 1.00, and topics is one to three lowercase words separated by commas.\n\n"
    "Review: {review}"
)

CHOICE_PROMPT_TEMPLATE = (
    "Classify the overall sentiment of this product review as exactly one word: "
    "positive, negative, mixed or neutral. Reply with that word and nothing else.\n\n"
    "Review: {review}"
)


# ---------------------------------------------------------------------------
# validation — stdlib only, so it is hand-rolled on purpose
# ---------------------------------------------------------------------------
def validate_verdict(obj) -> tuple[bool, str]:
    """Check `obj` against SCHEMA by hand. Returns (ok, reason)."""
    if not isinstance(obj, dict):
        return False, "not a JSON object"
    if set(obj.keys()) != {"sentiment", "confidence", "topics", "summary"}:
        return False, f"keys are {sorted(obj.keys())}"
    if obj["sentiment"] not in CHOICES:
        return False, f"sentiment={obj['sentiment']!r}"
    conf = obj["confidence"]
    if isinstance(conf, bool) or not isinstance(conf, (int, float)):
        return False, "confidence is not a number"
    if not (0 <= float(conf) <= 1):
        return False, "confidence out of [0,1]"
    topics = obj["topics"]
    if not isinstance(topics, list) or not (1 <= len(topics) <= 3):
        return False, "topics is not a list of 1..3"
    if not all(isinstance(t, str) and t for t in topics):
        return False, "topics contains a non-string"
    summary = obj["summary"]
    if not isinstance(summary, str) or len(summary) > 160:
        return False, "summary missing or too long"
    return True, ""


def first_json_object(text: str):
    """Best-effort: pull the first balanced {...} block out of a chatty answer.

    This is the 'recoverable' column: it is what you would have to write by hand (or pay
    a retry for) if the model wraps its JSON in prose or markdown fences.
    """
    start = text.find("{")
    while start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def check_json_arm(text: str) -> tuple[bool, bool, str]:
    """(strict_ok, recoverable_ok, reason) for the free-form and json_schema arms."""
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        obj = first_json_object(text)
        if obj is None:
            return False, False, f"not JSON ({exc.msg})"
        ok, reason = validate_verdict(obj)
        return False, ok, f"JSON wrapped in prose ({reason or 'shape ok'})"
    ok, reason = validate_verdict(obj)
    return ok, ok, reason


def check_regex_arm(text: str) -> tuple[bool, str]:
    line = text.strip()
    if not re.fullmatch(REGEX, line):
        return False, "does not match the regex"
    sentiment, conf, topics = line.split(";")
    if not (0.0 <= float(conf) <= 1.0):
        return False, "confidence out of range"
    if not (1 <= len(topics.split(",")) <= 3):
        return False, "wrong number of topics"
    return True, ""


def check_choice_arm(text: str) -> tuple[bool, str]:
    word = text.strip()
    if word in CHOICES:
        return True, ""
    return False, f"{word[:40]!r} is not one of the choices"


# Uniform (ok, recoverable, reason) wrappers, so `run_arm` never needs a lambda.
def json_checker(text: str) -> tuple[bool, bool, str]:
    return check_json_arm(text)


def regex_checker(text: str) -> tuple[bool, bool, str]:
    ok, reason = check_regex_arm(text)
    return ok, ok, reason


def choice_checker(text: str) -> tuple[bool, bool, str]:
    ok, reason = check_choice_arm(text)
    return ok, ok, reason


# ---------------------------------------------------------------------------
# one request
# ---------------------------------------------------------------------------
def one_request(prompt: str, extra: dict) -> tuple[str, float, dict, str]:
    """Return (text, elapsed_seconds, usage, error).

    Never raises for a server-side problem: every failure comes back as a non-empty
    `error` string so the arm can report it instead of aborting the lab.
    """
    started = time.perf_counter()
    try:
        out = chat(prompt, model=MODEL, max_tokens=MAX_TOKENS, **CHAT_KWARGS, **extra)
    except TypeError as exc:
        # The shared client did not forward our extra fields. Say so precisely.
        return "", time.perf_counter() - started, {}, (
            f"chat() rejected the extra fields {sorted(extra)}: {exc}. "
            "labs/_client.py must forward **sampling kwargs into the request body."
        )
    except urllib.error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode()[:400]
        except Exception:
            pass
        return "", time.perf_counter() - started, {}, f"HTTP {exc.code}: {body}"
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return "", time.perf_counter() - started, {}, f"transport: {exc}"
    except Exception as exc:  # e.g. _client.ServerNotRunning
        return "", time.perf_counter() - started, {}, f"{type(exc).__name__}: {exc}"

    elapsed = time.perf_counter() - started
    usage: dict = {}
    # labs/_client.py returns (text, elapsed, usage); accept a shorter form too so this
    # lab does not break if the shared client changes shape.
    if isinstance(out, tuple):
        text = out[0] if out else ""
        if len(out) > 1 and isinstance(out[1], (int, float)):
            elapsed = float(out[1])
        if len(out) > 2 and isinstance(out[2], dict):
            usage = out[2]
    elif isinstance(out, str):
        text = out
    else:  # pragma: no cover - defensive
        text = str(out)
    return text, elapsed, usage, ""


def probe_streaming(prompt: str, extra: dict, n: int) -> tuple[float | None, float | None]:
    """Return (mean_ttft_s, mean_inter_token_s) from `n` streamed requests.

    Best-effort on purpose: we only know that stream_chat yields (delta, timestamp).
    Gaps between successive timestamps are correct regardless of the timestamp's epoch;
    for TTFT we detect whether the first timestamp is absolute (perf_counter-like) or
    already relative to the request.
    """
    ttfts: list[float] = []
    gaps: list[float] = []
    for _ in range(n):
        t0 = time.perf_counter()
        stamps: list[float] = []
        try:
            for item in stream_chat(prompt, model=MODEL, max_tokens=MAX_TOKENS,
                                    **CHAT_KWARGS, **extra):
                if isinstance(item, tuple) and len(item) >= 2 and isinstance(item[1], (int, float)):
                    stamps.append(float(item[1]))
                elif isinstance(item, (int, float)):
                    stamps.append(float(item))
        except Exception as exc:  # streaming is a bonus; never fail the lab for it
            print(f"    (streaming probe failed: {exc})")
            return None, None
        if not stamps:
            continue
        first = stamps[0]
        # Two shapes are possible for the timestamp: absolute (perf_counter/time.time at
        # the moment the chunk arrived) or already relative to the request start. Decide
        # by comparison, not by magnitude: an absolute stamp is >= the t0 we recorded
        # just before sending, a relative one is smaller than it.
        ttfts.append(max(0.0, (first - t0) if first >= t0 else first))
        if len(stamps) > 1:
            gaps.extend(b - a for a, b in zip(stamps, stamps[1:]))
    mean_ttft = statistics.fmean(ttfts) if ttfts else None
    mean_gap = statistics.fmean(gaps) if gaps else None
    return mean_ttft, mean_gap


# ---------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------
def run_arm(name: str, constraint: str, template: str, extra: dict, checker) -> dict:
    print(f"\n--- arm: {name} ---")
    print(f"    constraint : {constraint}")
    if extra:
        print(f"    request    : {json.dumps(extra)[:160]}")

    # Warmup: the first request with a new grammar pays compilation
    # (xgrammar's cache is enabled, VLLM_XGRAMMAR_CACHE_MB=512 by default), and the first
    # request with any config pays CUDA-graph/memory warmup. Exclude both.
    warm_text, warm_elapsed, _, warm_err = one_request(template.format(review=REVIEWS[0]), extra)
    if warm_err:
        print(f"    WARMUP FAILED: {warm_err}")
        return {"arm": name, "constraint": constraint, "n": 0, "valid": 0, "error": warm_err}
    print(f"    warmup     : {warm_elapsed:.2f}s, {len(warm_text)} chars")

    rows = []
    for i in range(N):
        review = REVIEWS[i % len(REVIEWS)]
        text, elapsed, usage, err = one_request(template.format(review=review), extra)
        ok, recoverable, reason = checker(text) if not err else (False, False, err)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        rows.append(
            {
                "i": i,
                "text": text,
                "elapsed": elapsed,
                "ok": ok,
                "recoverable": recoverable,
                "reason": reason,
                "error": err,
                "chars": len(text),
                "completion_tokens": completion_tokens,
                # Exact, not a heuristic: the server stops at exactly max_tokens when it
                # truncates, and `usage.completion_tokens` comes back on every response
                # (UsageInfo in vllm/entrypoints/serve/engine/protocol.py:135-140).
                "looks_truncated": completion_tokens >= MAX_TOKENS,
            }
        )
        flag = "ok " if ok else ("rec" if recoverable else "BAD")
        note = "" if ok else f"  <- {reason}"
        print(f"    [{i + 1:2d}/{N}] {flag} {elapsed:6.2f}s {len(text):5d} chars{note}")

    valid = sum(1 for r in rows if r["ok"])
    recoverable = sum(1 for r in rows if r["recoverable"])
    truncated = sum(1 for r in rows if r["looks_truncated"])
    lat = [r["elapsed"] for r in rows]
    errors = [r["error"] for r in rows if r["error"]]

    ttft = gap = None
    if PROBE > 0:
        print(f"    streaming probe ({PROBE} extra requests) ...")
        ttft, gap = probe_streaming(template.format(review=REVIEWS[0]), extra, PROBE)

    result = {
        "arm": name,
        "constraint": constraint,
        "n": N,
        "valid": valid,
        "recoverable": recoverable,
        "errors": len(errors),
        "truncated": truncated,
        "mean_s": statistics.fmean(lat) if lat else float("nan"),
        "p50_s": statistics.median(lat) if lat else float("nan"),
        "max_s": max(lat) if lat else float("nan"),
        "mean_chars": statistics.fmean(r["chars"] for r in rows) if rows else 0.0,
        "mean_completion_tokens": (
            statistics.fmean(r["completion_tokens"] for r in rows) if rows else 0.0
        ),
        "ttft_s": ttft,
        "itl_s": gap,
        "first_error": errors[0] if errors else "",
        "sample": rows[0]["text"][:200] if rows else "",
    }
    print(f"    => {valid}/{N} satisfied the constraint "
          f"({recoverable}/{N} recoverable), mean {result['mean_s']:.2f}s")
    if truncated:
        print(f"    !! {truncated}/{N} hit max_tokens exactly ({MAX_TOKENS}) and were cut off; "
              f"raise MAX_TOKENS and re-run before drawing a conclusion")
    return result


def print_table(results: list[dict]) -> None:
    print("\n" + "=" * 108)
    print("COMPARISON — same task, same N, temperature 0, identical max_tokens")
    print("=" * 108)
    hdr = (f"{'arm':<12} {'valid':>7} {'recov':>6} {'trunc':>6} {'mean s':>8} {'p50 s':>7} "
           f"{'max s':>7} {'chars':>6} {'TTFT s':>7} {'ITL ms':>7}")
    print(hdr)
    print("-" * 108)
    for r in results:
        if not r.get("n"):
            print(f"{r['arm']:<12} {'--':>7}   (arm did not run: {r.get('error', '')[:60]})")
            continue
        ttft = f"{r['ttft_s']:.3f}" if r["ttft_s"] is not None else "-"
        itl = f"{r['itl_s'] * 1000:.1f}" if r["itl_s"] is not None else "-"
        print(f"{r['arm']:<12} {r['valid']:>3}/{r['n']:<3} {r['recoverable']:>3}/{r['n']:<2} "
              f"{r.get('truncated', 0):>6} "
              f"{r['mean_s']:>8.3f} {r['p50_s']:>7.3f} {r['max_s']:>7.3f} "
              f"{r['mean_chars']:>6.0f} {ttft:>7} {itl:>7}")
    print("-" * 108)
    print("valid  = the output satisfied the constraint EXACTLY as parsed")
    print("recov  = a JSON object could be recovered by hand (fences/prose stripped)")
    print("trunc  = hit max_tokens exactly, so the answer was cut off (from usage)")
    print("chars  = mean output length; a constraint usually makes the model stop talking")
    print("TTFT   = first streamed chunk; ITL = gap between chunks (PROBE=%d per arm)" % PROBE)


def interpretation(results: list[dict]) -> None:
    by = {r["arm"]: r for r in results}
    free, js, rx, ch = (by.get("freeform"), by.get("json_schema"),
                        by.get("regex"), by.get("choice"))
    print("\n" + "=" * 108)
    print("WHY A GRAMMAR COSTS LATENCY, AND WHY IT CAN STILL BE FASTER")
    print("=" * 108)
    print("""
  The hook, in three lines of source:

    1. vllm/v1/structured_output/__init__.py:314   StructuredOutputManager.grammar_bitmask()
       — once per engine step, for the whole batch: it asks the backend to fill a bitmask
       for every structured-output request in that step (`_fill_bitmasks`, lines 202-213,
       which calls `grammar.fill_bitmask(self._grammar_bitmask, index)`).

    2. vllm/v1/worker/gpu/structured_outputs.py:95  apply_grammar_bitmask()
       — the bitmask is copied host->device on a dedicated stream (lines 105-109) and then
       handed to a Triton kernel. Called from the model runner at
       vllm/v1/worker/gpu/model_runner.py:1594-1603, right after `compute_logits`.

    3. vllm/v1/worker/gpu/structured_outputs.py:162-199  _apply_grammar_bitmask_kernel
       — unpacks one bit per vocabulary entry (line 190) and writes
       `-float("inf")` over every disallowed logit (lines 195-199).

  So the constraint is NOT applied once when the request starts. It is re-derived and
  re-applied at EVERY decode step. That is the cost:

    * CPU work in the engine core: advancing the grammar's DFA/parser one token per step,
      per sequence. This is the part that scales with batch size, and it is why a
      grammar-heavy workload can become CPU-bound rather than GPU-bound.
    * a host->device copy of the bitmask per step (async, but not free);
    * a Triton kernel over the vocabulary per sequence per step;
    * a subtle sampling effect: masked logits change which token wins. Under a grammar the
      model is forced down the canonical path the grammar allows, which can differ from
      the token it would otherwise have picked, and can occasionally make the sequence
      *longer* (it must emit every required brace, key and bracket).

  And yet a constrained sampler can be FASTER end to end, for reasons the table above
  shows directly:

    * No retries. The free-form arm's failures cost a whole extra round trip (or a
      human/parser doing repairs). Compare "mean latency" against
      "mean latency x (1 / valid_rate)" — the effective cost per USABLE answer.
    * No repair code. "recoverable" is the amount of hand-written parsing you owe if you
      do not constrain. That code is not free in engineering time, and it is not free at
      runtime either.
    * Shorter outputs. A grammar tells the model exactly when to stop; the free-form arm
      tends to add commentary and fences. Fewer generated tokens is less decode time and
      less money.
    * Grammar compilation is cached (xgrammar's compiler is built with cache_enabled=True,
      VLLM_XGRAMMAR_CACHE_MB defaults to 512 in vllm/envs.py:1694), so a fleet serving the
      same schema pays that cost once per process, not once per request. This lab warms up
      before measuring precisely so compilation is not charged to request #1.

  The rule of thumb: constrain when you are going to PARSE the output anyway. The
  constraint costs a little per step; a parse failure costs a whole extra generation.
""")

    if free and free.get("n") and js and js.get("n"):
        vr_free = free["valid"] / free["n"]
        vr_js = js["valid"] / js["n"]
        print("  Your numbers:")
        print(f"    free-form valid rate      : {vr_free:.0%}  "
              f"({free['valid']}/{free['n']})")
        print(f"    json_schema valid rate    : {vr_js:.0%}  "
              f"({js['valid']}/{js['n']})")
        if vr_free > 0:
            print(f"    effective latency per USABLE free-form answer: "
                  f"{free['mean_s'] / vr_free:.2f}s  (mean {free['mean_s']:.2f}s / {vr_free:.0%})")
        else:
            print("    effective latency per USABLE free-form answer: undefined "
                  "(not one usable answer)")
        if vr_js > 0:
            print(f"    effective latency per USABLE json_schema answer: "
                  f"{js['mean_s'] / vr_js:.2f}s")
        print(f"    mean output length: free-form {free['mean_chars']:.0f} chars vs "
              f"json_schema {js['mean_chars']:.0f} chars")
    if js and rx and ch and js.get("n") and rx.get("n") and ch.get("n"):
        print(f"\n    Constraint cost (same task, same N):")
        print(f"      choice  (4-way grammar) : {ch['mean_s']:.3f}s mean")
        print(f"      regex   (line grammar)  : {rx['mean_s']:.3f}s mean")
        print(f"      json    (object grammar): {js['mean_s']:.3f}s mean")
        print("    A tighter grammar usually wins: a 4-way choice is a one-token decision,")
        print("    while a JSON object locks in a long canonical path of braces and keys.")


def trap_probe() -> None:
    """Show that the legacy guided_* fields are silently ignored in v0.30.0."""
    print("\n" + "=" * 108)
    print("THE TRAP — legacy `guided_regex` is accepted and IGNORED (HTTP 200, no constraint)")
    print("=" * 108)
    if SKIP_TRAP:
        print("  (skipped: SKIP_TRAP=1)")
        return
    # NOTE: temperature already comes from CHAT_KWARGS, and _client injects the Qwen3
    # no-thinking flag itself, so this dict must contain ONLY the legacy field under test.
    legacy_extra = {"guided_regex": REGEX}
    text, elapsed, _, err = one_request(
        REGEX_PROMPT_TEMPLATE.format(review=REVIEWS[0]), legacy_extra
    )
    if err:
        print(f"  request failed: {err}")
        return
    ok, reason = check_regex_arm(text)
    print(f"  request: {{'guided_regex': <the same regex the regex arm used>}}")
    print(f"  response: HTTP 200, {elapsed:.2f}s, {len(text)} chars")
    print(f"  output  : {text[:160]!r}")
    print(f"  matches the regex? {'YES' if ok else 'NO' + (' — ' + reason if reason else '')}")
    print("""
  This is the removal, not a bug you can fix from the client. `guided_json`,
  `guided_regex`, `guided_choice`, `guided_grammar`, `guided_decoding_backend` and
  `guided_whitespace_pattern` are in `_REMOVED_GUIDED_FIELDS`
  (vllm/entrypoints/serve/engine/protocol.py:21-34). The request model allows extra
  fields, so they are accepted and dropped, and the engine logs exactly one warning_once
  ("... which are ignored; output will NOT be constrained. Use `structured_outputs` (or
  `response_format`) instead").

  Check your server log for that line — if a workload "sometimes" returns bad JSON, this
  is the first thing to grep for:
      grep -i "removed guided-decoding" <server log>

  Use `structured_outputs`: {"regex": ...} / {"choice": [...]} / {"json": {...}} /
  {"grammar": ...} / {"json_object": true}, or `response_format` for JSON Schema.""")


def metrics_check() -> None:
    """Best-effort use of the shared metrics() helper: did the constrained arm generate
    fewer tokens? Key names differ across client versions, so this never fails the lab."""
    print("\n" + "--- metrics() cross-check ---")
    try:
        snap = metrics()
    except Exception as exc:
        print(f"  metrics() unavailable: {exc}")
        return
    if not isinstance(snap, dict):
        print(f"  metrics() returned {type(snap).__name__}, not a dict; skipping.")
        return
    # labs/_client.py:metrics() strips the exposition's `_total` suffix and sums counters
    # across label sets, so these are the keys to look for. The extra spellings are only
    # there so the lab does not silently do nothing if that convention changes.
    wanted = ["vllm:generation_tokens", "vllm:prompt_tokens",
              "vllm:prefix_cache_hits", "vllm:prefix_cache_queries",
              "generation_tokens", "vllm:generation_tokens_total",
              "vllm:prefix_cache_hits_total"]
    found = {k: snap[k] for k in wanted if k in snap}
    if not found:
        print(f"  metrics() has {len(snap)} series but none of {wanted}.")
        print(f"  keys seen (first 12): {sorted(snap)[:12]}")
        print("  Look for the generation-token counter by hand to confirm the constrained")
        print("  arm emits fewer tokens for the same task:")
        print(f"      curl -s {BASE}/metrics | grep -E 'generation_tokens'")
        return
    for k, v in found.items():
        print(f"  {k:34s} {v}")
    print("  (these are cumulative for the whole server lifetime; re-run the lab against a")
    print("   freshly started server if you want the per-run delta)")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def preflight() -> None:
    url = f"{BASE}/health"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            code = resp.status
    except urllib.error.HTTPError as exc:
        code = exc.code
    except Exception as exc:
        print(f"ERROR: no vLLM server at {url} ({exc})", file=sys.stderr)
        print("  Start one in another terminal first:  bash scripts/serve.sh", file=sys.stderr)
        print("  Or point this lab at it:              BASE_URL=http://host:port "
              "python labs/09_structured_outputs.py", file=sys.stderr)
        sys.exit(2)
    if code != 200:
        print(f"ERROR: {url} answered HTTP {code} — the engine is not healthy.", file=sys.stderr)
        print("  /health returns 503 when the engine is dead "
              "(vllm/entrypoints/serve/instrumentator/health.py:22-33).", file=sys.stderr)
        sys.exit(2)


def main() -> None:
    print("=" * 108)
    print("Lab 09 — Stage 8: structured outputs and the cost of a grammar")
    print("=" * 108)
    print(f"  endpoint    : {BASE}")
    print(f"  model       : {MODEL or '(first model the server lists)'}")
    print(f"  N per arm   : {N}")
    print(f"  max_tokens  : {MAX_TOKENS}")
    print(f"  temperature : {TEMPERATURE}   (determinism; do not raise this for a measurement)")

    preflight()

    results = [
        run_arm(
            "freeform",
            "none — the model is asked politely for JSON",
            PROMPT_TEMPLATE,
            {},
            json_checker,
        ),
        run_arm(
            "json_schema",
            'response_format={"type": "json_schema", ...}',
            PROMPT_TEMPLATE,
            {"response_format": RESPONSE_FORMAT},
            json_checker,
        ),
        run_arm(
            "regex",
            'structured_outputs={"regex": <regex>}',
            REGEX_PROMPT_TEMPLATE,
            {"structured_outputs": {"regex": REGEX}},
            regex_checker,
        ),
        run_arm(
            "choice",
            'structured_outputs={"choice": ["positive", "negative", "mixed", "neutral"]}',
            CHOICE_PROMPT_TEMPLATE,
            {"structured_outputs": {"choice": CHOICES}},
            choice_checker,
        ),
    ]

    print_table(results)
    interpretation(results)
    trap_probe()
    metrics_check()

    print("\n" + "=" * 108)
    print("CHECKPOINT — answer these out loud")
    print("=" * 108)
    print("""  - Where exactly does the grammar hook into the sampling loop, and how often does it run?
  - Why must the mask be computed for every sequence in the batch, not once per request?
  - Your free-form valid rate and your json_schema valid rate: what is the real cost of the
    difference once you include the retries you did not run?
  - Which arm had the LOWEST mean latency, and does that match "tighter grammar = cheaper"?
  - A colleague sends `guided_json` and gets unconstrained output with HTTP 200. What do
    you tell them, and which log line proves it?""")

    # ---- the machine-readable record -------------------------------------
    nums = {}
    for r in results:
        if not r.get("n"):
            continue
        nums[f"{r['arm']}_valid"] = r["valid"]
        nums[f"{r['arm']}_recoverable"] = r["recoverable"]
        nums[f"{r['arm']}_mean_s"] = round(r["mean_s"], 4)
        nums[f"{r['arm']}_p50_s"] = round(r["p50_s"], 4)
        nums[f"{r['arm']}_mean_chars"] = round(r["mean_chars"], 1)
        nums[f"{r['arm']}_mean_tokens"] = round(r.get("mean_completion_tokens", 0.0), 1)
        nums[f"{r['arm']}_truncated"] = r.get("truncated", 0)
        if r["ttft_s"] is not None:
            nums[f"{r['arm']}_ttft_s"] = round(r["ttft_s"], 4)
        if r["itl_s"] is not None:
            nums[f"{r['arm']}_itl_ms"] = round(r["itl_s"] * 1000, 3)
    nums["n_per_arm"] = N
    nums["max_tokens"] = MAX_TOKENS
    nums["temperature"] = TEMPERATURE
    record("lab09_structured_outputs", **nums)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# RECORD: write these into notes/stage-08-structured-outputs.md
#
#   date / vLLM version / torch version / GPU + driver / model / server command
#   N, MAX_TOKENS, TEMPERATURE (temperature must be 0 for a measurement)
#
#   For EACH of the four arms (freeform, json_schema, regex, choice):
#     - valid / N            (satisfied the constraint exactly)
#     - recoverable / N      (JSON that a human-written extractor could salvage)
#     - mean latency, s
#     - p50 latency, s
#     - mean output length, chars, and mean completion tokens (from the usage block)
#     - probe TTFT, s  and  probe inter-token latency, ms
#     - one sample output, verbatim
#
#   Then:
#     - effective latency per USABLE free-form answer
#       = mean_freeform / (valid_freeform / N)
#     - effective latency per USABLE json_schema answer, same formula
#     - which two arms differed most in mean output length, and by how much
#     - whether `guided_regex` constrained anything (paste the response and the server's
#       "removed guided-decoding field(s)" warning line)
#
# All numbers this lab prints are predictions of SHAPE only. Nothing here was run on a GPU
# while this project was written; the real numbers come from your pod.
# ---------------------------------------------------------------------------
