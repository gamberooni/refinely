import asyncio
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from crucible.core.exceptions import EvalError
from crucible.core.settings import Settings
from crucible.dspy.bridge import CASE_ATTR
from crucible.dspy.spec import DspyProgramSpec
from crucible.eval.datasets import EvalCase
from crucible.eval.metrics import CostMetric, LatencyMetric, Metric, MetricResult
from crucible.llm.client import LLMClient
from crucible.llm.usage import Result, TokenUsage
from crucible.registry import AppRegistration, register_app

DATASET_PATH = Path(__file__).resolve().parents[1] / "datasets" / "extraction_v1.json"


def _to_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().replace(",", "").replace("$", "")
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


class ExactMatchMetric:
    """Field-level equality against the expected value, with numeric tolerance."""

    name = "exact_match"

    def __init__(self, tolerance: float = 0.01) -> None:
        self._tolerance = tolerance

    def evaluate(self, case: EvalCase, output: Result) -> MetricResult:
        expected = case.expected
        actual = output.output
        matched = self._match(expected, actual)
        return MetricResult(
            metric_name=self.name,
            value=1.0 if matched else 0.0,
            raw={"expected": expected, "actual": actual},
        )

    def _match(self, expected: Any, actual: Any) -> bool:
        if isinstance(expected, dict):
            if not isinstance(actual, dict):
                return False
            return all(
                key in actual and self._match(val, actual[key]) for key, val in expected.items()
            )
        if isinstance(actual, dict):
            return any(self._match(expected, v) for v in actual.values() if not isinstance(v, dict))
        exp_num = _to_number(expected)
        act_num = _to_number(actual)
        if exp_num is not None and act_num is not None:
            return abs(exp_num - act_num) <= self._tolerance
        if isinstance(expected, str) and isinstance(actual, str):
            return expected.strip().lower() == actual.strip().lower()
        return expected == actual


class ExtractionResponse(BaseModel):
    """Structured field extraction from free text (e.g. sentiment, invoice total)."""

    field_name: str = Field(description="Name of the extracted field")
    field_value: str = Field(description="Extracted value as text")


SYSTEM_PROMPTS: dict[str, str] = {
    "strict": (
        "You are a precise data extraction engine. Extract the requested field "
        "from the user's text exactly as written. Do not infer, normalize, or "
        "add information. Output ONLY the extracted value."
    ),
    "verbose": (
        "You are a careful data extraction assistant. Read the user's text, "
        "identify the requested field, and extract its value. If the text is "
        "ambiguous, choose the most likely interpretation and note it. Output "
        "the extracted value clearly."
    ),
}


class ExtractionApp:
    """Structured field extraction from free text via `chat_structured`."""

    def __init__(
        self,
        client: LLMClient,
        settings: Settings | None = None,
        field_name: str = "sentiment",
        program_path: str | None = None,
    ) -> None:
        self._client = client
        self._settings = settings or Settings()
        self._field_name = field_name
        self._dspy_program: Any = None
        if program_path is not None:
            from crucible.dspy.load import load_program

            self._dspy_program = load_program(
                _extraction_dspy_factory(self._settings),
                program_path,
                self._settings,
                temperature=EXTRACTION_DEFAULT_CONFIG.get("temperature", 0.0),
            )

    def execute(self, input: dict, config: dict) -> Result:
        temperature = float(config.get("temperature", 0.0))
        variant = str(config.get("system_prompt_variant", "strict"))
        if variant not in SYSTEM_PROMPTS:
            raise EvalError(f"Unknown system_prompt_variant: {variant!r}")

        text = input.get("text", "")
        field = input.get("field", self._field_name)

        if self._dspy_program is not None:
            start = time.perf_counter()
            prediction = self._dspy_program(text=text, field=field)
            latency = time.perf_counter() - start
            return Result(
                output={
                    "field_name": prediction.field_name,
                    "field_value": prediction.field_value,
                },
                token_usage=TokenUsage(prompt_tokens=0, completion_tokens=0),
                latency_seconds=latency,
            )

        system_prompt = SYSTEM_PROMPTS[variant]
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (f"Extract the '{field}' field from this text:\n\n{text}"),
            },
        ]

        start = time.perf_counter()
        response, usage = asyncio.run(
            self._client.chat_structured(
                self._settings.model_name,
                messages,
                ExtractionResponse,
                temperature=temperature,
            )
        )
        latency = time.perf_counter() - start

        return Result(
            output=response.model_dump(),
            token_usage=TokenUsage(
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
            ),
            latency_seconds=latency,
        )


def sample_extraction_config(trial) -> dict[str, Any]:
    """Sample a config for the extraction app: temperature + prompt variant."""
    return {
        "temperature": trial.suggest_float("temperature", 0.0, 1.0),
        "system_prompt_variant": trial.suggest_categorical(
            "system_prompt_variant", ["strict", "verbose"]
        ),
    }


EXTRACTION_DEFAULT_CONFIG = {"temperature": 0.0, "system_prompt_variant": "strict"}

EXTRACTION_WEIGHTS = {
    "exact_match": 0.7,
    "latency": 0.15,
    "cost": 0.15,
}


def _extraction_dspy_factory(settings: Settings) -> DspyProgramSpec:
    def build():
        dspy = __import__("dspy")
        return dspy.Predict("text, field -> field_name, field_value")

    def prepare_example(case: EvalCase):
        dspy = __import__("dspy")
        field = case.input.get("field", "sentiment")
        example = dspy.Example(
            text=case.input.get("text", ""),
            field=field,
            field_name=case.expected.get("field_name", field)
            if isinstance(case.expected, dict)
            else field,
            field_value=case.expected.get("field_value", "")
            if isinstance(case.expected, dict)
            else str(case.expected),
        ).with_inputs("text", "field")
        example[CASE_ATTR] = case
        return example

    def prediction_to_output(pred) -> dict:
        return {
            "field_name": getattr(pred, "field_name", ""),
            "field_value": getattr(pred, "field_value", ""),
        }

    return DspyProgramSpec(
        build=build,
        prepare_example=prepare_example,
        prediction_to_output=prediction_to_output,
    )


def _build_adapter(
    client: LLMClient, settings: Settings, program_path: str | None = None
) -> "ExtractionApp":
    return ExtractionApp(client, settings, program_path=program_path)


def _metrics_factory(client: LLMClient, settings: Settings) -> list[Metric]:
    return [ExactMatchMetric(), LatencyMetric(), CostMetric()]


register_app(
    AppRegistration(
        name="extraction",
        build_adapter=_build_adapter,
        metrics_factory=_metrics_factory,
        search_space=sample_extraction_config,
        default_config=EXTRACTION_DEFAULT_CONFIG,
        weights=EXTRACTION_WEIGHTS,
        dataset_path=DATASET_PATH,
        dspy_factory=_extraction_dspy_factory,
    )
)
