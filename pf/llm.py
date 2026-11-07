"""Model adapter.

Every model call in this project goes through call(). Swapping provider or
model is a config change, never a code change:

    llm:
      provider: gemini          # gemini | groq | openrouter | openai | anthropic | mock
      tag_model: gemini-2.0-flash
      judge_model: gemini-2.0-flash

The tag pass and the judge pass can run on DIFFERENT models. That is
deliberate: a judge that did not make the pick cannot prefer its own work.

Retries are capped at 1 (see README: a retry storm is what trips a free-tier
rate limit). Responses are cached on disk by a hash of prompt+model, so
re-ranking never re-calls the model.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request

CACHE_DIR = os.environ.get("PROMO_CACHE", ".cache")
MAX_RETRIES = 1          # hard cap, see README
TIMEOUT = 120

# Cloudflare sits in front of some providers (Groq does) and rejects urllib's
# default "Python-urllib/3.x" User-Agent outright: HTTP 403, Cloudflare code
# 1010, on every endpoint including /models. It is not an auth failure and no
# key or model name will fix it. Naming the client honestly is enough to pass.
USER_AGENT = "promo-finder/1.0"

# Provider knobs that are not model names. Reasoning models (Groq's gpt-oss
# family) spend output tokens thinking before they answer. With no explicit
# cap they can spend the entire default budget on reasoning and return an
# EMPTY completion, which Groq reports as json_validate_failed with
# failed_generation "". Measured on this pack's episode-1 tag prompt:
# gpt-oss-20b at default effort burned 4,561 output tokens for 591 characters
# of JSON; at "low" effort gpt-oss-120b produced equally valid output in 581.
# That 8x matters because the free tier allows 8,000 tokens per minute.
# run.py overrides both from config.yaml, so these are defaults, not policy.
MAX_TOKENS = 8192
# Per-model overrides, because one budget cannot serve every model: a verbose
# reasoner needs thousands of output tokens for the same JSON that an
# efficient one produces in 600, while some models enforce an output-per-
# minute ceiling that a large max_tokens trips before the call even runs.
MAX_TOKENS_BY_MODEL = {}
REASONING_EFFORT = "low"
# Also per-model: some models return an EMPTY completion when this parameter
# is set at all, so for those it must be omitted rather than tuned.
REASONING_EFFORT_BY_MODEL = {}


def _max_tokens_for(model: str) -> int:
    return int(MAX_TOKENS_BY_MODEL.get(model, MAX_TOKENS))


def _reasoning_for(model: str) -> str:
    if model in REASONING_EFFORT_BY_MODEL:
        return REASONING_EFFORT_BY_MODEL[model] or ""
    return REASONING_EFFORT or ""


# Minimum seconds between two REAL provider calls (cache hits are free and
# never wait). Free tiers meter tokens per minute over a rolling window, and
# Groq pre-checks each request as `input + max_tokens` against that window:
# at 8,000 TPM a ~4,000-token tag prompt simply cannot be sent twice in one
# minute. Pacing the calls is what keeps the run from 429-ing; it is the same
# reasoning as the 1-retry cap, applied before the failure instead of after.
MIN_CALL_INTERVAL = 0.0
_last_call_at = 0.0


def _pace():
    """Wait, if needed, so real calls stay MIN_CALL_INTERVAL apart."""
    global _last_call_at
    if MIN_CALL_INTERVAL > 0 and _last_call_at:
        wait = MIN_CALL_INTERVAL - (time.monotonic() - _last_call_at)
        if wait > 0:
            time.sleep(wait)
    _last_call_at = time.monotonic()


def _with_ua(headers: dict) -> dict:
    """Same headers, plus our User-Agent unless the caller set one."""
    out = {"User-Agent": USER_AGENT}
    out.update(headers or {})
    return out


class LLMError(Exception):
    pass


class LLMClientError(LLMError):
    """A 4xx that will not succeed on retry (bad model name, bad request,
    bad key). Retrying these wastes free-tier quota, so they fail at once."""


ENV_KEYS = {
    "gemini": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "mock": None,
}


def detect_provider() -> str:
    """Pick the first provider whose key is present. Keeps setup to one env var."""
    for name in ("gemini", "groq", "openrouter", "anthropic", "openai"):
        if os.environ.get(ENV_KEYS[name]):
            return name
    return "mock"


def _redact(url: str) -> str:
    return re.sub(r"key=[^&]+", "key=***", url)


def _post(url: str, payload: dict, headers: dict) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=body, headers=_with_ua(headers),
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:700]
        msg = f"HTTP {exc.code} from {_redact(url)}\n    {detail}"
        # 429 is worth one retry; other 4xx will fail identically every time.
        if exc.code == 429:
            err = LLMError(msg)
            # Wait as long as the provider asks. This is why ONE retry is
            # enough: a tokens-per-minute window needs a real pause, and
            # retrying instantly would just burn the quota again.
            err.retry_after = _retry_after_seconds(exc, detail)
            raise err from None
        raise LLMClientError(msg) from None


def _retry_after_seconds(exc, detail, default=20.0, cap=75.0):
    """How long the provider says to wait before retrying, in seconds."""
    raw = exc.headers.get("retry-after") if exc.headers else None
    if raw:
        try:
            return min(float(raw) + 1.0, cap)
        except (TypeError, ValueError):
            pass
    # Groq states it in the message body: "Please try again in 6.52s".
    match = re.search(r"try again in ([0-9.]+)\s*(ms|s)\b", detail, re.I)
    if match:
        value = float(match.group(1))
        if match.group(2).lower() == "ms":
            value /= 1000.0
        return min(value + 1.0, cap)
    return default


def list_models(provider: str, key: str):
    """Ask the provider which models this key may call. Used by --check."""
    if provider == "gemini":
        url = "https://generativelanguage.googleapis.com/v1beta/models"
        try:
            req = urllib.request.Request(url, headers=_with_ua({"x-goog-api-key": key}))
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
            return [m["name"].split("/")[-1] for m in data.get("models", [])
                    if "generateContent" in (m.get("supportedGenerationMethods") or [])]
        except urllib.error.HTTPError:
            url = f"{url}?key={key}"      # fall back to the legacy query parameter
    elif provider in ("groq", "openrouter", "openai"):
        base = {"groq": "https://api.groq.com/openai/v1",
                "openrouter": "https://openrouter.ai/api/v1",
                "openai": "https://api.openai.com/v1"}[provider]
        url = f"{base}/models"
    else:
        return []
    headers = {} if provider == "gemini" else {"Authorization": f"Bearer {key}"}
    try:
        req = urllib.request.Request(url, headers=_with_ua(headers))
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        raise LLMClientError(
            f"HTTP {exc.code} listing models: "
            f"{exc.read().decode('utf-8', 'replace')[:400]}") from None

    if provider == "gemini":
        return [m["name"].split("/")[-1] for m in data.get("models", [])
                if "generateContent" in (m.get("supportedGenerationMethods") or [])]
    return [m.get("id") for m in data.get("data", []) if m.get("id")]


def _extract_json(text: str):
    """Models wrap JSON in prose or fences more often than they should."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        i, j = text.find(opener), text.rfind(closer)
        if i != -1 and j > i:
            try:
                return json.loads(text[i:j + 1])
            except json.JSONDecodeError:
                continue
    raise LLMError(f"could not parse JSON from model output: {text[:300]!r}")


# ---------------------------------------------------------------- providers

def _call_gemini(prompt, model, key, temperature):
    """Google documents the x-goog-api-key header. The older ?key= query
    parameter still works for legacy AIza keys but is rejected for the newer
    AQ. authentication keys that AI Studio now issues, so the header is tried
    first and the query parameter only as a fall-back."""
    base = (f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent")
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature,
                             "responseMimeType": "application/json"},
    }
    try:
        data = _post(base, payload,
                     {"Content-Type": "application/json", "x-goog-api-key": key})
    except LLMClientError as header_err:
        try:
            data = _post(f"{base}?key={key}", payload,
                         {"Content-Type": "application/json"})
        except LLMClientError as query_err:
            raise LLMClientError(
                f"both auth styles were rejected.\n"
                f"  x-goog-api-key header -> {header_err}\n"
                f"  ?key= query parameter -> {query_err}") from None
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        raise LLMError(f"unexpected gemini response: {json.dumps(data)[:300]}")


def _call_openai_compatible(prompt, model, key, temperature, base):
    payload = {"model": model, "temperature": temperature,
               "messages": [{"role": "user", "content": prompt}],
               "response_format": {"type": "json_object"},
               "max_tokens": _max_tokens_for(model)}
    # Only sent when set: some providers reject this parameter outright and
    # some return an empty completion when it is present.
    effort = _reasoning_for(model)
    if effort:
        payload["reasoning_effort"] = effort
    data = _post(
        f"{base}/chat/completions",
        payload,
        {"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    try:
        return data["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        raise LLMError(f"unexpected response: {json.dumps(data)[:300]}")


def _call_anthropic(prompt, model, key, temperature):
    data = _post(
        "https://api.anthropic.com/v1/messages",
        {"model": model, "max_tokens": 8192, "temperature": temperature,
         "messages": [{"role": "user", "content": prompt}]},
        {"Content-Type": "application/json", "x-api-key": key,
         "anthropic-version": "2023-06-01"},
    )
    try:
        return data["content"][0]["text"]
    except (KeyError, IndexError):
        raise LLMError(f"unexpected anthropic response: {json.dumps(data)[:300]}")


def _call_mock(prompt, model, temperature):
    """Deterministic stand-in so every code path is testable without a key.

    It understands the two prompt shapes this project uses and returns
    well-formed, schema-valid output derived from the ids in the prompt.
    """
    ids = re.findall(r"\[([a-z]\d{3}b\d{3}(?:#\d+)?)\]", prompt)
    seed = int(hashlib.sha256(prompt.encode()).hexdigest()[:8], 16)

    if "CANDIDATE PROMOS" in prompt:
        cands = re.findall(r"^(cand_\d+):", prompt, re.M)
        return json.dumps({"scores": [
            {"candidate_id": c,
             "hook": 2 + (seed + i * 7) % 4,
             "cliffhanger": 2 + (seed + i * 13) % 4,
             "standalone": 2 + (seed + i * 5) % 4,
             "escalation": 2 + (seed + i * 3) % 4,
             "why": "Mock judge output for pipeline testing.",
             "rank_within_episode": i + 1}
            for i, c in enumerate(cands)]})

    if "CLUSTERS" in prompt:
        groups = re.findall(r"^(cluster_\d+):", prompt, re.M)
        return json.dumps({"clusters": [
            {"cluster_id": g, "name": f"Mock group {i + 1}", "merge_with": None}
            for i, g in enumerate(groups)]})

    # tag pass
    if not ids:
        return json.dumps({"hooks": [], "cliffhangers": [], "tags": {},
                           "context_dependency": {}})
    step = max(1, len(ids) // 7)
    hooks = ids[::step][:6]
    cliffs = ids[::-1][::step][:5]
    vocab = ["humiliation", "reversal", "betrayal", "accident", "reveal",
             "quotable", "emotional_peak"]
    return json.dumps({
        "hooks": hooks,
        "cliffhangers": cliffs,
        "tags": {b: [vocab[(seed + i) % len(vocab)]] for i, b in enumerate(hooks + cliffs)},
        "context_dependency": {b: 1 + (seed + i) % 5 for i, b in enumerate(hooks + cliffs)},
    })


# ---------------------------------------------------------------- public API

def call(prompt: str, model: str, provider: str | None = None,
         temperature: float = 0.0, use_cache: bool = True) -> dict:
    """Send one prompt, return parsed JSON. The only model entry point."""
    provider = provider or detect_provider()
    key = os.environ.get(ENV_KEYS.get(provider) or "", "")
    if provider != "mock" and not key:
        raise LLMError(
            f"provider {provider!r} needs {ENV_KEYS[provider]} in the environment. "
            f"Set it, or use provider: mock to exercise the pipeline without a key."
        )

    digest = hashlib.sha256(f"{provider}|{model}|{temperature}|{prompt}".encode()).hexdigest()[:20]
    cache_path = os.path.join(CACHE_DIR, f"{digest}.json")
    if use_cache and os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as fh:
            return json.load(fh)

    last = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            if provider != "mock":
                _pace()    # real call: respect the free-tier rolling window
            if provider == "mock":
                raw = _call_mock(prompt, model, temperature)
            elif provider == "gemini":
                raw = _call_gemini(prompt, model, key, temperature)
            elif provider == "groq":
                raw = _call_openai_compatible(prompt, model, key, temperature,
                                              "https://api.groq.com/openai/v1")
            elif provider == "openrouter":
                raw = _call_openai_compatible(prompt, model, key, temperature,
                                              "https://openrouter.ai/api/v1")
            elif provider == "openai":
                raw = _call_openai_compatible(prompt, model, key, temperature,
                                              "https://api.openai.com/v1")
            elif provider == "anthropic":
                raw = _call_anthropic(prompt, model, key, temperature)
            else:
                raise LLMError(f"unknown provider {provider!r}")

            parsed = _extract_json(raw)
            os.makedirs(CACHE_DIR, exist_ok=True)
            with open(cache_path, "w", encoding="utf-8") as fh:
                json.dump(parsed, fh)
            return parsed

        except LLMClientError:
            raise                      # a 4xx will not fix itself; do not burn quota
        except (urllib.error.HTTPError, urllib.error.URLError, LLMError, TimeoutError) as exc:
            last = exc
            if attempt < MAX_RETRIES:
                time.sleep(getattr(exc, "retry_after", None) or 2)
    raise LLMError(f"model call failed after {MAX_RETRIES + 1} attempts: {last}")
