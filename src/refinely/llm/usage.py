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
    """Output of an application execution, plus execution metadata."""

    output: dict[str, Any] | str
    token_usage: TokenUsage
    latency_seconds: float


class ChatResult(BaseModel, Generic[T]):
    """A chat completion: parsed content plus token usage."""

    content: T
    token_usage: TokenUsage
