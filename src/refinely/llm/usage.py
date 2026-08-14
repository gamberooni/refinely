from typing import Any, Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class TokenUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class Result(BaseModel):
    """Output of an application execution, plus execution metadata.

    ``token_usage`` and ``latency_seconds`` are None when the measurement is
    unavailable (e.g. a compiled DSPy program whose LM does not surface usage);
    the cost/latency metrics then raise `MetricUnavailableError` and the
    metrics drop out of that run's aggregate instead of scoring a fake value.
    """

    output: dict[str, Any] | str
    token_usage: TokenUsage | None = None
    latency_seconds: float | None = None


class ChatResult(BaseModel, Generic[T]):
    """A chat completion: parsed content plus token usage."""

    content: T
    token_usage: TokenUsage
