import pytest

from apps.extraction import ExactMatchMetric, ExtractionApp
from apps.qa import QAApp
from apps.rag import CitationAccuracyMetric, RetrievalRecallMetric
from crucible.core.exceptions import EvalError
from crucible.eval.datasets import EvalCase
from crucible.llm.usage import Result, TokenUsage
from crucible.retrieval import retrieve_snippets, retrieve_snippets_indexed
from tests.stub_llm import StubLLMClient

CORPUS = [
    "The capital of France is Paris, a city on the Seine.",
    "Python is a general-purpose programming language.",
    "Paris is known as the City of Light.",
    "SQLite is an embedded SQL database engine.",
    "The Eiffel Tower is a landmark in Paris.",
]


def _case(expected: object, question: str | None = None) -> EvalCase:
    input_data = {"question": question} if question else {"text": "some input"}
    return EvalCase(id="c1", input=input_data, expected=expected)


def _result(output: object, latency: float = 0.5, tokens: tuple[int, int] = (100, 20)) -> Result:
    return Result(
        output=output,
        token_usage=TokenUsage(prompt_tokens=tokens[0], completion_tokens=tokens[1]),
        latency_seconds=latency,
    )


def test_retrieval_returns_correct_snippets() -> None:
    question = "What is the capital of France?"
    snippets = retrieve_snippets(question, CORPUS, top_k=2)

    assert len(snippets) == 2
    assert "capital of France" in snippets[0]


def test_retrieval_respects_top_k() -> None:
    question = "Tell me about Paris"
    snippets = retrieve_snippets(question, CORPUS, top_k=5)

    assert 1 <= len(snippets) <= 5
    assert all(s in CORPUS for s in snippets)


def test_retrieval_top_k_zero_returns_empty() -> None:
    assert retrieve_snippets("anything", CORPUS, top_k=0) == []


def test_retrieval_indexed_returns_positions() -> None:
    question = "What is the capital of France?"
    results = retrieve_snippets_indexed(question, CORPUS, top_k=2)

    assert len(results) == 2
    assert results[0][0] == 0
    assert "capital of France" in results[0][1]
    assert all(isinstance(idx, int) and CORPUS[idx] == snippet for idx, snippet in results)


def test_retrieval_keyword_strategy_excludes_substring_only_matches() -> None:
    corpus = ["ananas are a tropical fruit", "bread is tasty"]
    question = "Is banana bread tasty?"

    keyword = retrieve_snippets_indexed(question, corpus, top_k=2, strategy="keyword")
    hybrid = retrieve_snippets_indexed(question, corpus, top_k=2, strategy="hybrid")
    keyword_idx = {idx for idx, _ in keyword}
    hybrid_idx = {idx for idx, _ in hybrid}

    assert keyword_idx == {1}
    assert hybrid_idx == {1, 0}
    assert keyword_idx <= hybrid_idx


def test_retrieval_unknown_strategy_raises() -> None:
    with pytest.raises(EvalError, match="Unknown retrieval strategy"):
        retrieve_snippets_indexed("anything", CORPUS, strategy="bogus")


def test_extraction_app_is_adapter() -> None:
    app = ExtractionApp(client=StubLLMClient())
    assert hasattr(app, "execute") and callable(app.execute)


def test_extraction_app_returns_result_with_structured_output() -> None:
    stub = StubLLMClient(
        structured_responses=[{"field_name": "sentiment", "field_value": "positive"}],
        prompt_tokens=30,
        completion_tokens=7,
    )
    app = ExtractionApp(client=stub)

    result = app.execute({"text": "I love this product!"}, {"temperature": 0.2})

    assert result.output == {"field_name": "sentiment", "field_value": "positive"}
    assert result.token_usage == TokenUsage(prompt_tokens=30, completion_tokens=7)
    assert result.latency_seconds >= 0.0
    assert stub.structured_calls[0]["temperature"] == 0.2


def test_extraction_app_uses_variant_prompt() -> None:
    stub = StubLLMClient(structured_responses=[{"field_name": "a", "field_value": "b"}])
    app = ExtractionApp(client=stub)

    app.execute({"text": "x"}, {"system_prompt_variant": "verbose"})

    assert stub.structured_calls[0]["messages"][0]["role"] == "system"
    assert "careful data extraction" in stub.structured_calls[0]["messages"][0]["content"]


def test_extraction_app_unknown_variant_raises() -> None:
    app = ExtractionApp(client=StubLLMClient())

    with pytest.raises(EvalError):
        app.execute({"text": "x"}, {"system_prompt_variant": "bogus"})


def test_exact_match_dict_equal() -> None:
    expected = {"field_name": "sentiment", "field_value": "positive"}
    m = ExactMatchMetric().evaluate(_case(expected), _result(expected))
    assert m.value == 1.0


def test_exact_match_dict_unequal() -> None:
    expected = {"field_name": "sentiment", "field_value": "negative"}
    m = ExactMatchMetric().evaluate(
        _case(expected), _result({"field_name": "sentiment", "field_value": "positive"})
    )
    assert m.value == 0.0


def test_exact_match_numeric_tolerance() -> None:
    m = ExactMatchMetric().evaluate(_case(10.0), _result("10.005"))
    assert m.value == 1.0


def test_exact_match_string_case_insensitive() -> None:
    m = ExactMatchMetric().evaluate(_case("Paris"), _result({"answer": "paris"}))
    assert m.value == 1.0


def test_retrieval_recall_full() -> None:
    expected = {"answer": "x", "source_indices": [0, 1]}
    m = RetrievalRecallMetric().evaluate(_case(expected), _result({"retrieved_indices": [0, 1, 2]}))
    assert m.value == 1.0


def test_retrieval_recall_partial() -> None:
    expected = {"answer": "x", "source_indices": [0, 1]}
    m = RetrievalRecallMetric().evaluate(_case(expected), _result({"retrieved_indices": [1, 2]}))
    assert m.value == pytest.approx(0.5)


def test_retrieval_recall_none_retrieved() -> None:
    expected = {"answer": "x", "source_indices": [0, 1]}
    m = RetrievalRecallMetric().evaluate(_case(expected), _result({"retrieved_indices": []}))
    assert m.value == 0.0


def test_retrieval_recall_ignores_non_dict_output() -> None:
    expected = {"answer": "x", "source_indices": [0, 1]}
    m = RetrievalRecallMetric().evaluate(_case(expected), _result("plain text"))
    assert m.value == 0.0


def test_citation_accuracy_full() -> None:
    expected = {"answer": "x", "source_indices": [0, 1]}
    m = CitationAccuracyMetric().evaluate(_case(expected), _result({"cited_indices": [0, 1]}))
    assert m.value == 1.0


def test_citation_accuracy_partial_precision() -> None:
    expected = {"answer": "x", "source_indices": [0, 1]}
    m = CitationAccuracyMetric().evaluate(_case(expected), _result({"cited_indices": [0, 5]}))
    assert m.value == pytest.approx(0.5)


def test_citation_accuracy_no_citations_scores_zero() -> None:
    expected = {"answer": "x", "source_indices": [0, 1]}
    m = CitationAccuracyMetric().evaluate(_case(expected), _result({"cited_indices": []}))
    assert m.value == 0.0


def test_qa_app_is_adapter() -> None:
    app = QAApp(client=StubLLMClient(), corpus=CORPUS)
    assert hasattr(app, "execute") and callable(app.execute)


def test_qa_app_returns_answer_and_metadata() -> None:
    stub = StubLLMClient(structured_responses=[{"answer": "Paris"}])
    app = QAApp(client=stub, corpus=CORPUS)

    result = app.execute(
        {"question": "What is the capital of France?"},
        {"temperature": 0.1, "top_k": 2, "system_prompt_variant": "strict"},
    )

    assert result.output == {"answer": "Paris"}
    assert result.token_usage.prompt_tokens == 10
    assert result.latency_seconds >= 0.0
    assert stub.structured_calls[0]["temperature"] == 0.1


def test_qa_app_retrieves_snippets_into_prompt() -> None:
    stub = StubLLMClient(structured_responses=[{"answer": "Paris"}])
    app = QAApp(client=stub, corpus=CORPUS)

    app.execute({"question": "What is the capital of France?"}, {"top_k": 1})

    user_content = stub.structured_calls[0]["messages"][-1]["content"]
    assert "capital of France" in user_content
    assert "[snippet 1]" in user_content
