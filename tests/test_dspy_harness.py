"""Tests for the DSPy compile harness — fully hermetic (no real LLM / dspy calls)."""

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from crucible.core.exceptions import EvalError
from crucible.core.settings import Settings
from crucible.dspy.bridge import (
    CASE_ATTR,
    example_case,
    make_dspy_metric,
    prediction_result,
    score_result,
)
from crucible.dspy.compile import CompileResult, _split_train_val, compile_program
from crucible.dspy.spec import DspyProgramSpec
from crucible.eval.datasets import EvalCase
from crucible.eval.metrics import MetricResult
from crucible.llm.usage import Result, TokenUsage

# ---------------------------------------------------------------------------
# Hermetic settings fixture (mirrors rest of test suite)
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_real_env_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(Settings.model_config, "env_file", None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _case(id: str = "c1") -> EvalCase:
    return EvalCase(id=id, input={"text": "hello"}, expected="world")


def _result(output: dict | None = None) -> Result:
    return Result(
        output=output or {"answer": "world"},
        token_usage=TokenUsage(prompt_tokens=5, completion_tokens=3),
        latency_seconds=0.1,
    )


class _ExactMetric:
    """Toy metric: 1.0 if output.answer == case.expected, else 0.0."""

    name = "exact"

    def evaluate(self, case: EvalCase, result: Result) -> MetricResult:
        output = result.output
        val = 1.0 if output.get("answer") == case.expected else 0.0
        return MetricResult(metric_name="exact", value=val)


class _ThrowingMetric:
    name = "throw"

    def evaluate(self, case: EvalCase, result: Result) -> MetricResult:
        raise RuntimeError("boom")


# ---------------------------------------------------------------------------
# bridge.example_case
# ---------------------------------------------------------------------------


class _FakeExample:
    """Minimal duck-type for a dspy.Example with attr access."""

    def __init__(self, case: EvalCase | None) -> None:
        self._case = case

    def __getattr__(self, name: str) -> Any:
        if name == CASE_ATTR:
            return self._case
        raise AttributeError(name)

    def inputs(self):
        return {}


def test_example_case_via_attr():
    case = _case()
    example = _FakeExample(case)
    setattr(example, CASE_ATTR, case)
    recovered = example_case(example)
    assert recovered is case


def test_example_case_via_item():
    case = _case()
    example = {CASE_ATTR: case, "question": "hi"}
    assert example_case(example) is case


def test_example_case_missing_raises():
    with pytest.raises(EvalError, match=CASE_ATTR):
        example_case({"question": "no case here"})


def test_example_case_wrong_type_raises():
    bad = {"question": "hi", CASE_ATTR: "not-an-evalcase"}
    with pytest.raises(EvalError, match=CASE_ATTR):
        example_case(bad)


# ---------------------------------------------------------------------------
# bridge.prediction_result
# ---------------------------------------------------------------------------


def test_prediction_result_zero_usage():
    spec = DspyProgramSpec(
        build=lambda: None,
        prepare_example=lambda case: None,
        prediction_to_output=lambda pred: {"answer": pred},
    )
    result = prediction_result(spec, "hello")
    assert result.output == {"answer": "hello"}
    assert result.token_usage.prompt_tokens == 0
    assert result.token_usage.completion_tokens == 0
    assert result.latency_seconds == 0.0


# ---------------------------------------------------------------------------
# bridge.score_result
# ---------------------------------------------------------------------------


def test_score_result_correct():
    case = _case()
    result = _result({"answer": "world"})
    scores, agg = score_result(case, result, [_ExactMetric()], {"exact": 1.0})
    assert scores == {"exact": 1.0}
    assert agg == pytest.approx(1.0)


def test_score_result_wrong():
    case = _case()
    result = _result({"answer": "wrong"})
    scores, agg = score_result(case, result, [_ExactMetric()], {"exact": 1.0})
    assert scores == {"exact": 0.0}
    assert agg == pytest.approx(0.0)


def test_score_result_metric_throw_gives_zero():
    case = _case()
    result = _result()
    scores, agg = score_result(case, result, [_ThrowingMetric()], {"throw": 1.0})
    assert scores == {"throw": 0.0}
    assert agg == pytest.approx(0.0)


def test_score_result_multi_metric():
    case = _case()
    result = _result({"answer": "world"})
    metrics = [_ExactMetric(), _ThrowingMetric()]
    weights = {"exact": 0.6, "throw": 0.4}
    scores, agg = score_result(case, result, metrics, weights)
    assert scores["exact"] == pytest.approx(1.0)
    assert scores["throw"] == pytest.approx(0.0)
    assert agg == pytest.approx(0.6 * 1.0 + 0.4 * 0.0)


# ---------------------------------------------------------------------------
# bridge.make_dspy_metric
# ---------------------------------------------------------------------------


def test_make_dspy_metric_correct_prediction():
    spec = DspyProgramSpec(
        build=lambda: None,
        prepare_example=lambda case: None,
        prediction_to_output=lambda pred: {"answer": pred},
    )
    gold = {CASE_ATTR: _case(), "question": "q"}
    metric = make_dspy_metric(spec, [_ExactMetric()], {"exact": 1.0})
    score = metric(gold, "world")
    assert isinstance(score, float)
    assert score == pytest.approx(1.0)


def test_make_dspy_metric_wrong_prediction():
    spec = DspyProgramSpec(
        build=lambda: None,
        prepare_example=lambda case: None,
        prediction_to_output=lambda pred: {"answer": pred},
    )
    gold = {CASE_ATTR: _case(), "question": "q"}
    metric = make_dspy_metric(spec, [_ExactMetric()], {"exact": 1.0})
    assert metric(gold, "WRONG") == pytest.approx(0.0)


def test_make_dspy_metric_accepts_trace_kwarg():
    spec = DspyProgramSpec(
        build=lambda: None,
        prepare_example=lambda case: None,
        prediction_to_output=lambda pred: {"answer": pred},
    )
    gold = {CASE_ATTR: _case()}
    metric = make_dspy_metric(spec, [_ExactMetric()], {"exact": 1.0})
    assert isinstance(metric(gold, "world", trace=None), float)


# ---------------------------------------------------------------------------
# compile._split_train_val
# ---------------------------------------------------------------------------


def _cases(n: int) -> list[EvalCase]:
    return [_case(str(i)) for i in range(n)]


def test_split_proportions_default():
    train, val = _split_train_val(_cases(10))
    assert len(train) == 7
    assert len(val) == 3


def test_split_max_examples_caps():
    train, val = _split_train_val(_cases(20), max_examples=6)
    assert len(train) + len(val) == 6


def test_split_ensures_at_least_one_val():
    train, val = _split_train_val(_cases(2))
    assert len(val) >= 1
    assert len(train) >= 1


def test_split_too_few_raises():
    with pytest.raises(EvalError, match="at least 2"):
        _split_train_val(_cases(1))


def test_split_zero_raises():
    with pytest.raises(EvalError, match="at least 2"):
        _split_train_val(_cases(0))


def test_split_deterministic_with_seed():
    a_train, a_val = _split_train_val(_cases(10), seed=7)
    b_train, b_val = _split_train_val(_cases(10), seed=7)
    assert [c.id for c in a_train] == [c.id for c in b_train]
    assert [c.id for c in a_val] == [c.id for c in b_val]


# ---------------------------------------------------------------------------
# compile.compile_program — no real dspy calls
# ---------------------------------------------------------------------------


def _build_stub_dspy(compiled_prediction: dict | None = None) -> MagicMock:
    """Return a fake dspy module whose BootstrapFewShot compiles trivially."""
    compiled_prediction = compiled_prediction or {"answer": "world"}

    fake_compiled = MagicMock()
    fake_compiled.save = MagicMock()

    # compiled(**inputs()) → dict-like prediction
    def _call_compiled(**kwargs):
        m = MagicMock()
        m.__iter__ = lambda s: iter(compiled_prediction.items())
        m.__contains__ = lambda s, k: k in compiled_prediction
        m.get = lambda k, d=None: compiled_prediction.get(k, d)
        for k, v in compiled_prediction.items():
            setattr(m, k, v)
        return m

    fake_compiled.side_effect = _call_compiled

    fake_optimizer = MagicMock()
    fake_optimizer.compile.return_value = fake_compiled

    fake_dspy = MagicMock()
    fake_dspy.BootstrapFewShot.return_value = fake_optimizer
    fake_dspy.configure = MagicMock()
    fake_dspy.LM = MagicMock(return_value=MagicMock())

    return fake_dspy


def test_compile_program_no_dspy_factory(tmp_path: Path):
    import apps  # noqa: F401
    from crucible.registry import AppRegistration, register_app

    register_app(
        AppRegistration(
            name="nodspy_app",
            build_adapter=lambda client, settings, program_path=None: MagicMock(),
            metrics_factory=lambda client, settings: [_ExactMetric()],
            search_space=lambda trial: {},
            default_config={},
            weights={"exact": 1.0},
            dspy_factory=None,
        )
    )
    dataset = _cases(4)

    with pytest.raises(EvalError, match="does not declare a DSPy program"):
        compile_program(
            app_name="nodspy_app",
            dataset=dataset,
            dataset_version="v1",
            client=MagicMock(),
            output_dir=tmp_path,
        )


def test_compile_program_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """compile_program runs end-to-end with a stubbed dspy module."""
    import apps  # noqa: F401 — triggers registration of 'extraction'

    fake_dspy = _build_stub_dspy({"field_name": "sentiment", "field_value": "positive"})

    # Patch _dspy() in every module that calls it during compile
    monkeypatch.setattr("crucible.dspy.compile._dspy", lambda: fake_dspy)
    monkeypatch.setattr("crucible.dspy.lm._dspy", lambda: fake_dspy)

    # Stub out configure_lm so it doesn't actually call dspy.configure
    monkeypatch.setattr(
        "crucible.dspy.compile.configure_lm",
        lambda settings, temperature=0.0, **kw: None,
    )

    from apps.extraction import DATASET_PATH
    from crucible.eval.datasets import load_dataset

    dataset = load_dataset(DATASET_PATH)
    assert len(dataset) >= 2

    from tests.stub_llm import StubLLMClient

    # Build enough structured responses for the EvaluationRunner baseline pass
    # (one per val case; extraction returns {field_name, field_value})
    n_val = max(1, len(dataset) - round(len(dataset) * 0.7))
    stub_responses = [{"field_name": "sentiment", "field_value": "positive"} for _ in range(n_val)]
    client = StubLLMClient(structured_responses=stub_responses)

    result = compile_program(
        app_name="extraction",
        dataset=dataset,
        dataset_version="v_test",
        client=client,
        settings=Settings(openai_api_key="sk-test"),
        output_dir=tmp_path,
        output_name="out.json",
        max_examples=6,
    )

    assert isinstance(result, CompileResult)
    assert result.app_name == "extraction"
    assert result.dataset_version == "v_test"
    assert result.optimizer == "BootstrapFewShot"
    assert result.artifact_path == tmp_path / "out.json"
    assert result.n_train + result.n_val <= 6
    assert 0.0 <= result.baseline_score <= 1.0
    assert 0.0 <= result.compiled_score <= 1.0
    fake_dspy.BootstrapFewShot.assert_called_once()
    fake_dspy.BootstrapFewShot().compile.assert_called_once()


# ---------------------------------------------------------------------------
# 7.3 — dspy smoke (skipped if dspy not installed)
# ---------------------------------------------------------------------------


def test_dspy_importorskip_compile_program():
    pytest.importorskip("dspy")
    from crucible.dspy import (
        CompileResult,
        DspyProgramSpec,
        compile_program,
        configure_lm,
    )

    assert callable(compile_program)
    assert callable(configure_lm)
    assert DspyProgramSpec is not None
    assert CompileResult is not None


def test_dspy_lm_wiring_from_settings(monkeypatch: pytest.MonkeyPatch):
    """configure_lm passes api_base only when base_url is set."""
    dspy = pytest.importorskip("dspy")

    captured: dict = {}

    class FakeLM:
        def __init__(self, model, **kwargs):
            captured["model"] = model
            captured["kwargs"] = kwargs

    monkeypatch.setattr(dspy, "LM", FakeLM)
    monkeypatch.setattr(dspy, "configure", lambda **kw: None)

    from crucible.dspy.lm import configure_lm

    s = Settings(openai_api_key="sk-test", model_name="my-model", base_url="http://gw/v1")
    configure_lm(s, temperature=0.1)

    assert captured["model"] == "openai/my-model"
    assert captured["kwargs"]["temperature"] == pytest.approx(0.1)
    assert captured["kwargs"]["api_base"] == "http://gw/v1"


def test_dspy_lm_no_base_url(monkeypatch: pytest.MonkeyPatch):
    """configure_lm omits api_base when base_url is None."""
    dspy = pytest.importorskip("dspy")

    captured: dict = {}

    class FakeLM:
        def __init__(self, model, **kwargs):
            captured["kwargs"] = kwargs

    monkeypatch.setattr(dspy, "LM", FakeLM)
    monkeypatch.setattr(dspy, "configure", lambda **kw: None)

    from crucible.dspy.lm import configure_lm

    s = Settings(openai_api_key="sk-test", base_url=None)
    configure_lm(s)

    assert "api_base" not in captured["kwargs"]


# ---------------------------------------------------------------------------
# retrieval.format_snippet_block
# ---------------------------------------------------------------------------


def test_format_snippet_block_default_ordinals():
    from crucible.retrieval import format_snippet_block

    block = format_snippet_block(["alpha", "beta"])
    assert block == "[snippet 1] alpha\n\n[snippet 2] beta"


def test_format_snippet_block_explicit_labels():
    from crucible.retrieval import format_snippet_block

    block = format_snippet_block(["alpha", "beta"], labels=[3, 7])
    assert block == "[snippet 3] alpha\n\n[snippet 7] beta"


def test_format_snippet_block_empty():
    from crucible.retrieval import format_snippet_block

    assert format_snippet_block([]) == ""


# ---------------------------------------------------------------------------
# dspy.load.load_program
# ---------------------------------------------------------------------------


def test_load_program_builds_loads_and_configures(monkeypatch: pytest.MonkeyPatch):
    loaded: list[str] = []

    class FakeProgram:
        def load(self, path: str) -> None:
            loaded.append(path)

    built = FakeProgram()

    spec = DspyProgramSpec(
        build=lambda: built,
        prepare_example=lambda case: None,
        prediction_to_output=lambda pred: {},
    )
    configured: dict = {}

    monkeypatch.setattr(
        "crucible.dspy.load._dspy",
        lambda: MagicMock(),
    )
    monkeypatch.setattr(
        "crucible.dspy.load.configure_lm",
        lambda settings, temperature=0.0, **kw: configured.update(
            {"settings": settings, "temperature": temperature}
        ),
    )

    from crucible.dspy.load import load_program

    s = Settings(openai_api_key="sk-test")
    program = load_program(spec, "artifacts/prog.json", s, temperature=0.25)

    assert program is built
    assert loaded == ["artifacts/prog.json"]
    assert configured["settings"] is s
    assert configured["temperature"] == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# dspy.adapter.CompiledProgramAdapter
# ---------------------------------------------------------------------------


def test_compiled_program_adapter_execute():
    from crucible.dspy.adapter import CompiledProgramAdapter

    class _Example:
        def inputs(self):
            return {"text": "hi", "field": "sentiment"}

    called: dict = {}

    def _program(**kwargs):
        called.update(kwargs)
        return MagicMock(field_name="sentiment", field_value="positive")

    spec = DspyProgramSpec(
        build=lambda: None,
        prepare_example=lambda case: _Example(),
        prediction_to_output=lambda pred: {
            "field_name": pred.field_name,
            "field_value": pred.field_value,
        },
    )

    adapter = CompiledProgramAdapter(spec, _program)
    result = adapter.execute(input={"text": "hi"}, config={})

    assert called == {"text": "hi", "field": "sentiment"}
    assert result.output == {"field_name": "sentiment", "field_value": "positive"}
    assert result.token_usage.prompt_tokens == 0
    assert result.token_usage.completion_tokens == 0
    assert result.latency_seconds == 0.0
