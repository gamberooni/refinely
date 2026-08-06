"""Stub `LLMClient` test double with canned responses — no network access."""

import json
from typing import Any

from pydantic import BaseModel

from crucible.llm.usage import ChatResult, TokenUsage


class StubLLMClient:
    """Minimal fake implementing the `LLMClient` protocol.

    `structured_responses` maps a JSON-schema-ish key to a canned dict; a plain
    dict value is returned for any model. `text_responses` is a queue of canned
    text answers. Token usage is canned per call.
    """

    def __init__(
        self,
        structured_responses: list[dict[str, Any]] | None = None,
        text_responses: list[str] | None = None,
        prompt_tokens: int = 10,
        completion_tokens: int = 5,
    ) -> None:
        self.structured_responses = list(structured_responses or [])
        self.text_responses = list(text_responses or ["canned answer"])
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.structured_calls: list[dict[str, Any]] = []
        self.text_calls: list[dict[str, Any]] = []

    async def close(self) -> None:
        pass

    async def chat_structured(
        self,
        model: str,
        messages: list[dict],
        response_model: type[BaseModel],
        temperature: float = 0.0,
        seed: int = 42,
    ) -> ChatResult[BaseModel]:
        self.structured_calls.append(
            {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "seed": seed,
            }
        )
        canned = self.structured_responses.pop(0) if self.structured_responses else {}
        if isinstance(canned, str):
            parsed = json.loads(canned)
        else:
            parsed = dict(canned)
        return ChatResult(content=response_model.model_validate(parsed), token_usage=self._usage())

    async def chat_text(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.0,
        seed: int = 42,
    ) -> ChatResult[str]:
        self.text_calls.append(
            {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "seed": seed,
            }
        )
        canned = self.text_responses.pop(0) if self.text_responses else "canned answer"
        return ChatResult(content=canned, token_usage=self._usage())

    def _usage(self) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
        )
