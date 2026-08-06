import pytest
from pydantic import BaseModel

from crucible.llm.client import (
    LLMClient,
    _extract_json_from_prose,
    _strip_json_fences,
)
from crucible.llm.usage import ChatResult, TokenUsage
from tests.stub_llm import StubLLMClient


class _ExampleModel(BaseModel):
    sentiment: str


def test_strip_json_fences_removes_fenced_block() -> None:
    text = '```json\n{"sentiment": "positive"}\n```'
    assert _strip_json_fences(text) == '{"sentiment": "positive"}'


def test_strip_json_fences_passthrough_plain_json() -> None:
    text = '{"sentiment": "positive"}'
    assert _strip_json_fences(text) == text


def test_extract_json_from_prose() -> None:
    text = 'Here is the result: {"sentiment": "negative"} hope this helps.'
    assert _extract_json_from_prose(text) == '{"sentiment": "negative"}'


def test_extract_json_from_prose_handles_nested_braces() -> None:
    text = 'Prefix {"a": {"b": "c"}, "d": 1} suffix'
    assert _extract_json_from_prose(text) == '{"a": {"b": "c"}, "d": 1}'


def test_extract_json_from_prose_returns_none_when_absent() -> None:
    assert _extract_json_from_prose("no json here") is None


def test_stub_client_satisfies_protocol_and_returns_canned_usage() -> None:
    stub = StubLLMClient(
        structured_responses=[{"sentiment": "positive"}],
        prompt_tokens=100,
        completion_tokens=50,
    )

    client: LLMClient = stub  # protocol conformance check

    result = _run(client)

    assert result.content.sentiment == "positive"
    assert result.token_usage == TokenUsage(prompt_tokens=100, completion_tokens=50)
    assert result.token_usage.total == 150


def _run(client: LLMClient) -> ChatResult[_ExampleModel]:
    import asyncio

    return asyncio.run(
        client.chat_structured("gpt-4o-mini", [{"role": "user", "content": "hi"}], _ExampleModel)
    )


def test_stub_client_chat_text_returns_canned_answer() -> None:
    stub = StubLLMClient(text_responses=["42 is the answer"])

    import asyncio

    result = asyncio.run(stub.chat_text("gpt-4o-mini", [{"role": "user", "content": "q"}]))

    assert result.content == "42 is the answer"
    assert result.token_usage.prompt_tokens == 10


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            'Here is the result: {"sentiment": "negative"} hope this helps.',
            '{"sentiment": "negative"}',
        ),
    ],
)
def test_prose_extraction_parametrized(raw: str, expected: str) -> None:
    assert _extract_json_from_prose(raw) == expected
