import json
from typing import Any, Protocol, TypeVar, cast

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from crucible.core.exceptions import LLMError
from crucible.llm.usage import ChatResult, TokenUsage

T = TypeVar("T", bound=BaseModel)

# Models that do not accept a `temperature` parameter (e.g. newer Bedrock Claude releases).
_NO_TEMPERATURE_MODELS: tuple[str, ...] = (
    "claude-sonnet-5",
    "claude-opus-4",
)


def _supports_temperature(model: str) -> bool:
    """Return False for models known to reject the temperature parameter."""
    model_lower = model.lower()
    return not any(m in model_lower for m in _NO_TEMPERATURE_MODELS)


def _strip_json_fences(text: str) -> str:
    """Strip markdown code fences from an LLM response."""
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        inner = lines[1:] if lines[0].startswith("```") else lines
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        stripped = "\n".join(inner)
    return stripped.strip()


def _extract_json_from_prose(text: str) -> str | None:
    """Try to extract the first JSON object or array from prose text."""
    for start_char, end_char in [("{", "}"), ("[", "]")]:
        start = text.find(start_char)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escape_next = False
        for i, ch in enumerate(text[start:], start):
            if escape_next:
                escape_next = False
                continue
            if ch == "\\" and in_string:
                escape_next = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == start_char:
                depth += 1
            elif ch == end_char:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return None


class LLMClient(Protocol):
    async def close(self) -> None: ...

    async def chat_structured(
        self,
        model: str,
        messages: list[dict],
        response_model: type[T],
        temperature: float = 0.0,
        seed: int = 42,
    ) -> ChatResult[T]: ...

    async def chat_text(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.0,
        seed: int = 42,
    ) -> ChatResult[str]: ...


class AsyncOpenAIClient:
    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        max_retries: int = 3,
        temperature: float = 0.0,
        seed: int = 42,
    ) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._default_temperature = temperature
        self._default_seed = seed
        # Retry is wired here (not as a class-level @retry decorator) because the
        # decorator is evaluated at class-definition time and cannot see the
        # per-instance max_retries value.
        self.chat_structured = retry(  # type: ignore[method-assign]
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type((TimeoutError, ConnectionError)),
        )(self.chat_structured)
        self.chat_text = retry(  # type: ignore[method-assign]
            stop=stop_after_attempt(max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=10),
            retry=retry_if_exception_type((TimeoutError, ConnectionError)),
        )(self.chat_text)

    async def close(self) -> None:
        await self._client.close()

    async def chat_structured(
        self,
        model: str,
        messages: list[dict],
        response_model: type[T],
        temperature: float | None = None,
        seed: int | None = None,
    ) -> ChatResult[T]:
        temp = temperature if temperature is not None else self._default_temperature
        s = seed if seed is not None else self._default_seed
        try:
            return await self._chat_structured_fallback(model, messages, response_model, temp, s)
        except LLMError:
            raise
        except Exception as e:
            raise LLMError(f"LLM call failed: {e}") from e

    async def _chat_structured_fallback(
        self,
        model: str,
        messages: list[dict],
        response_model: type[T],
        temperature: float,
        seed: int,
    ) -> ChatResult[T]:
        schema = response_model.model_json_schema()
        schema_json = json.dumps(schema, indent=2)

        system_msg = {
            "role": "system",
            "content": (
                "You must respond with valid JSON matching the following schema.\n"
                f"```json\n{schema_json}\n```\n"
                "Do not include any text outside the JSON block."
            ),
        }

        fixed_messages = cast(
            list[ChatCompletionMessageParam],
            [system_msg] + [m for m in messages if m["role"] != "system"],
        )

        completion = await self._client.chat.completions.create(  # type: ignore[call-overload]
            model=model,
            messages=fixed_messages,
            response_format={"type": "json_object"},
            **(
                {}
                if not _supports_temperature(model)
                else {"temperature": temperature, "seed": seed}
            ),
        )

        content = completion.choices[0].message.content or ""
        usage = completion.usage
        token_usage = TokenUsage(
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )

        # Attempt 1: strip fences, parse directly
        candidates = [_strip_json_fences(content), _extract_json_from_prose(content)]
        for candidate in candidates:
            if not candidate:
                continue
            try:
                return ChatResult(
                    content=response_model.model_validate_json(candidate),
                    token_usage=token_usage,
                )
            except Exception:  # noqa: BLE001, S110 - JSON candidate fallback
                pass

        # Attempt 2: ask the model to re-emit strictly as JSON
        repair_messages = cast(
            list[ChatCompletionMessageParam],
            [
                {
                    "role": "system",
                    "content": (
                        "The previous response was not valid JSON. "
                        "Output ONLY a JSON object matching this schema, with no other text:\n"
                        f"{json.dumps(schema, indent=2)}"
                    ),
                },
                {
                    "role": "user",
                    "content": f"Previous response:\n{content}\n\nNow output valid JSON only.",
                },
            ],
        )
        repair_completion = await self._client.chat.completions.create(  # type: ignore[call-overload]
            model=model,
            messages=repair_messages,
            response_format={"type": "json_object"},
            **({} if not _supports_temperature(model) else {"temperature": 0.0, "seed": seed}),
        )
        repair_content = repair_completion.choices[0].message.content or ""
        repair_usage = repair_completion.usage
        token_usage = TokenUsage(
            prompt_tokens=token_usage.prompt_tokens
            + (repair_usage.prompt_tokens if repair_usage else 0),
            completion_tokens=token_usage.completion_tokens
            + (repair_usage.completion_tokens if repair_usage else 0),
        )

        for candidate in [
            _strip_json_fences(repair_content),
            _extract_json_from_prose(repair_content),
        ]:
            if not candidate:
                continue
            try:
                return ChatResult(
                    content=response_model.model_validate_json(candidate),
                    token_usage=token_usage,
                )
            except Exception:  # noqa: BLE001, S110 - JSON candidate fallback
                pass

        raise LLMError(
            f"Failed to parse LLM JSON response after repair attempt. Raw: {repair_content[:200]}"
        )

    async def chat_text(
        self,
        model: str,
        messages: list[dict],
        temperature: float | None = None,
        seed: int | None = None,
    ) -> ChatResult[str]:
        temp = temperature if temperature is not None else self._default_temperature
        s = seed if seed is not None else self._default_seed

        try:
            extra: dict[str, Any] = {}
            if _supports_temperature(model):
                extra["temperature"] = temp
                extra["seed"] = s
            completion = await self._client.chat.completions.create(
                model=model,
                messages=cast(list[ChatCompletionMessageParam], messages),
                **extra,
            )
        except Exception as e:
            raise LLMError(f"LLM call failed: {e}") from e

        content = completion.choices[0].message.content or ""
        usage = completion.usage
        token_usage = TokenUsage(
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )
        return ChatResult(content=content, token_usage=token_usage)
