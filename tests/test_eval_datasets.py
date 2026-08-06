import pytest

from crucible.core.exceptions import EvalError
from crucible.eval.datasets import dataset_version, load_corpus, load_dataset


def test_loader_parses_extraction_dataset() -> None:
    cases = load_dataset("datasets/extraction_v1.json")

    assert len(cases) == 10
    assert all(c.id and c.input and c.expected is not None for c in cases)
    assert cases[0].input["text"]
    assert cases[0].expected == {"field_name": "sentiment", "field_value": "positive"}


def test_loader_parses_qa_dataset() -> None:
    cases = load_dataset("datasets/qa_v1.json")

    assert len(cases) == 10
    assert all(c.input["question"] for c in cases)
    assert cases[0].expected == "Paris"


def test_loader_parses_rag_dataset() -> None:
    cases = load_dataset("datasets/rag_v1.json")

    assert len(cases) == 10
    assert all(c.input["question"] for c in cases)
    for case in cases:
        assert isinstance(case.expected, dict)
        assert case.expected["answer"]
        assert isinstance(case.expected["source_indices"], list)
        assert len(case.expected["source_indices"]) >= 2
        assert all(isinstance(i, int) and 0 <= i < 12 for i in case.expected["source_indices"])


def test_load_corpus_from_rag_dataset() -> None:
    corpus = load_corpus("datasets/rag_v1.json")

    assert isinstance(corpus, list)
    assert all(isinstance(s, str) for s in corpus)
    assert len(corpus) == 12


def test_loader_missing_required_field_raises(tmp_path: pytest.TempPathFactory) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text('[{"input": {"text": "x"}}]')

    with pytest.raises(EvalError, match="missing required field"):
        load_dataset(bad)


def test_loader_missing_file_raises(tmp_path: pytest.TempPathFactory) -> None:
    with pytest.raises(EvalError, match="not found"):
        load_dataset(tmp_path / "nope.json")


def test_loader_accepts_plain_list(tmp_path: pytest.TempPathFactory) -> None:
    p = tmp_path / "plain.json"
    p.write_text('[{"id": "a", "input": {"text": "x"}, "expected": "y"}]')

    cases = load_dataset(p)

    assert len(cases) == 1
    assert cases[0].id == "a"


def test_dataset_version_from_file() -> None:
    assert dataset_version("datasets/extraction_v1.json") == "extraction_v1"


def test_dataset_version_falls_back_to_filename(
    tmp_path: pytest.TempPathFactory,
) -> None:
    p = tmp_path / "my_dataset.json"
    p.write_text("[]")

    assert dataset_version(p) == "my_dataset"


def test_load_corpus_from_qa_dataset() -> None:
    corpus = load_corpus("datasets/qa_v1.json")

    assert isinstance(corpus, list)
    assert all(isinstance(s, str) for s in corpus)
    assert len(corpus) > 0


def test_load_corpus_missing_key_raises() -> None:
    with pytest.raises(EvalError, match="no 'corpus' key"):
        load_corpus("datasets/extraction_v1.json")
