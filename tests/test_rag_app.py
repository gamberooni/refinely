import pytest

from apps.rag import RAGApp
from crucible.core.exceptions import EvalError
from crucible.llm.usage import TokenUsage
from tests.stub_llm import StubLLMClient

CORPUS = [
    "The capital of France is Paris, a city on the Seine.",
    "Python is a general-purpose programming language.",
    "Paris is known as the City of Light.",
    "SQLite is an embedded SQL database engine.",
    "The Eiffel Tower is a landmark in Paris.",
]


def test_rag_app_is_adapter() -> None:
    app = RAGApp(client=StubLLMClient(), corpus=CORPUS)
    assert hasattr(app, "execute") and callable(app.execute)


def test_rag_app_default_path_single_call() -> None:
    stub = StubLLMClient(structured_responses=[{"answer": "Paris", "cited_snippets": [0]}])
    app = RAGApp(client=stub, corpus=CORPUS)

    result = app.execute(
        {"question": "Paris capital of France?"},
        {"temperature": 0.2, "top_k": 2, "system_prompt_variant": "strict"},
    )

    assert len(stub.structured_calls) == 1
    assert len(stub.text_calls) == 0
    assert result.output["answer"] == "Paris"
    assert result.output["cited_indices"] == [0]
    assert 0 in result.output["retrieved_indices"]
    assert result.token_usage == TokenUsage(prompt_tokens=10, completion_tokens=5)
    assert result.latency_seconds >= 0.0
    assert stub.structured_calls[0]["temperature"] == 0.2


def test_rag_app_prompt_uses_real_corpus_indices() -> None:
    stub = StubLLMClient(structured_responses=[{"answer": "Paris", "cited_snippets": [0]}])
    app = RAGApp(client=stub, corpus=CORPUS)

    app.execute({"question": "Paris capital of France?"}, {"top_k": 2})

    user_content = stub.structured_calls[0]["messages"][-1]["content"]
    assert "[snippet 0]" in user_content
    assert "capital of France" in user_content


def test_rag_app_query_expansion_adds_text_call() -> None:
    stub = StubLLMClient(
        structured_responses=[{"answer": "Paris", "cited_snippets": [0]}],
        text_responses=["Capital of France city, Eiffel Tower height"],
    )
    app = RAGApp(client=stub, corpus=CORPUS)

    result = app.execute(
        {"question": "What is the capital of France?"},
        {"query_expansion": True},
    )

    assert len(stub.text_calls) == 1
    assert len(stub.structured_calls) == 1
    assert result.output["answer"] == "Paris"
    assert result.token_usage.prompt_tokens == 20
    user_content = stub.structured_calls[0]["messages"][-1]["content"]
    assert "Capital of France city, Eiffel Tower height" in user_content


def test_rag_app_rerank_reorders_candidates() -> None:
    stub = StubLLMClient(
        structured_responses=[
            {"scores": [1, 5]},
            {"answer": "Paris", "cited_snippets": [2]},
        ]
    )
    app = RAGApp(client=stub, corpus=CORPUS)

    result = app.execute(
        {"question": "Paris capital of France?"},
        {"top_k": 2, "rerank": True},
    )

    assert len(stub.structured_calls) == 2
    assert stub.structured_calls[0]["messages"][-1]["content"].startswith("Question:")
    assert result.output["retrieved_indices"] == [2, 0]
    assert result.token_usage.prompt_tokens == 20
    assert result.token_usage.completion_tokens == 10


def test_rag_app_rerank_skipped_with_single_candidate() -> None:
    stub = StubLLMClient(structured_responses=[{"answer": "Paris", "cited_snippets": [0]}])
    app = RAGApp(client=stub, corpus=CORPUS)

    app.execute(
        {"question": "Capital of France?"},
        {"rerank": True, "retrieval_strategy": "keyword"},
    )

    assert len(stub.structured_calls) == 1


def test_rag_app_retrieval_strategy_switch() -> None:
    corpus = ["ananas are a tropical fruit", "bread is tasty"]
    question = "Is banana bread tasty?"

    keyword_stub = StubLLMClient(structured_responses=[{"answer": "x", "cited_snippets": []}])
    hybrid_stub = StubLLMClient(structured_responses=[{"answer": "x", "cited_snippets": []}])

    keyword_result = RAGApp(client=keyword_stub, corpus=corpus).execute(
        {"question": question}, {"retrieval_strategy": "keyword"}
    )
    hybrid_result = RAGApp(client=hybrid_stub, corpus=corpus).execute(
        {"question": question}, {"retrieval_strategy": "hybrid"}
    )

    assert keyword_result.output["retrieved_indices"] == [1]
    assert hybrid_result.output["retrieved_indices"] == [1, 0]


def test_rag_app_unknown_variant_raises_without_llm_calls() -> None:
    stub = StubLLMClient()
    app = RAGApp(client=stub, corpus=CORPUS)

    with pytest.raises(EvalError, match="Unknown system_prompt_variant"):
        app.execute({"question": "x"}, {"system_prompt_variant": "bogus"})

    assert len(stub.structured_calls) == 0
    assert len(stub.text_calls) == 0
