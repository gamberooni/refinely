"""Wire DSPy's language model to refinely's `Settings` + OpenAI-compatible gateway."""

import time
from typing import Any

from refinely.core.settings import Settings
from refinely.dspy._imports import _dspy
from refinely.llm.usage import TokenUsage

_last_usage: TokenUsage | None = None
_last_latency: float | None = None


def last_usage() -> TokenUsage | None:
    """Token usage of the most recent dspy LM call, or None when unavailable."""
    return _last_usage


def last_latency() -> float | None:
    """Wall-clock latency of the most recent dspy LM call, or None when unavailable."""
    return _last_latency


def _extract_usage(lm: Any) -> TokenUsage | None:
    """Read the last history entry's usage defensively across dspy versions."""
    history = getattr(lm, "history", None)
    if not history:
        return None
    entry = history[-1]
    usage = None
    if hasattr(entry, "usage"):
        try:
            usage = entry.usage
        except Exception:  # noqa: BLE001 - version-dependent shape
            usage = None
    elif isinstance(entry, dict):
        raw = entry.get("usage")
        usage = raw if isinstance(raw, dict) else None
    if not isinstance(usage, dict):
        return None
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    if prompt is None or completion is None:
        return None
    return TokenUsage(prompt_tokens=int(prompt), completion_tokens=int(completion))


class _UsageTrackingLM:
    """Wraps a dspy.LM, recording the most recent call's usage and latency."""

    def __init__(self, lm: Any) -> None:
        self._lm = lm

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        global _last_usage, _last_latency
        start = time.perf_counter()
        result = self._lm(*args, **kwargs)
        elapsed = time.perf_counter() - start
        usage = _extract_usage(self._lm)
        if usage is not None:
            _last_usage = usage
            _last_latency = elapsed
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self._lm, name)


def configure_lm(settings: Settings, temperature: float = 0.0, **kwargs: Any) -> Any:
    """Create a usage-tracking `dspy.LM` wrapper from `Settings` and configure it.

    The provider prefix is always ``openai/`` because the model is served by an
    OpenAI-compatible gateway; `api_base` is forwarded only when `base_url` is
    set so direct OpenAI runs keep the default endpoint. Last-call usage and
    latency are exposed via `last_usage()` / `last_latency()`.
    """
    global _last_usage, _last_latency
    _last_usage = None
    _last_latency = None
    dspy = _dspy()
    model = f"openai/{settings.model_name}"
    lm_kwargs: dict[str, Any] = {
        "temperature": temperature,
        "api_key": settings.openai_api_key,
        **kwargs,
    }
    if settings.base_url:
        lm_kwargs["api_base"] = settings.base_url
    lm = dspy.LM(model, **lm_kwargs)
    wrapped = _UsageTrackingLM(lm)
    dspy.configure(lm=wrapped)
    return wrapped
