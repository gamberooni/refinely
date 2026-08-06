import asyncio
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from apps.common import format_snippet_block, retrieve_snippets
from crucible.core.exceptions import EvalError
from crucible.core.settings import Settings
from crucible.dspy.bridge import CASE_ATTR
from crucible.dspy.spec import DspyProgramSpec
from crucible.eval.datasets import EvalCase, load_corpus
from crucible.eval.metrics import (
    CostMetric,
    FuzzyMatchMetric,
    LatencyMetric,
    LLMJudgeMetric,
    Metric,
)
from crucible.llm.client import LLMClient
from crucible.llm.usage import Result, TokenUsage
from crucible.registry import AppRegistration, register_app

DATASET_PATH = Path(__file__).resolve().parents[1] / "datasets" / "qa_v1.json"


class QAAnswer(BaseModel):
    answer: str


SYSTEM_PROMPTS: dict[str, str] = {
    "strict": (
        "You are a concise Q&A assistant. Answer the question using ONLY the "
        "provided snippets. If the snippets do not contain the answer, say so. "
        "Do not use outside knowledge."
    ),
    "verbose": (
        "You are a helpful Q&A assistant. Answer the question using the "
        "provided snippets, explaining your reasoning in 1-3 sentences."
    ),
}


class QAApp:
    """Retrieval-lite Q&A: in-memory snippet retrieval + LLM answer."""

    def __init__(
        self,
        client: LLMClient,
        corpus: list[str],
        settings: Settings | None = None,
        program_path: str | None = None,
    ) -> None:
        self._client = client
        self._corpus = list(corpus)
        self._settings = settings or Settings()
        self._dspy_program: Any = None
        if program_path is not None:
            from crucible.dspy.load import load_program

            self._dspy_program = load_program(
                _qa_dspy_factory(self._corpus, self._settings),
                program_path,
                self._settings,
                temperature=QA_DEFAULT_CONFIG.get("temperature", 0.0),
            )

    def execute(self, input: dict, config: dict) -> Result:
        temperature = float(config.get("temperature", 0.0))
        top_k = int(config.get("top_k", 3))
        variant = str(config.get("system_prompt_variant", "strict"))
        if variant not in SYSTEM_PROMPTS:
            raise EvalError(f"Unknown system_prompt_variant: {variant!r}")

        question = input.get("question", "")
        snippets = retrieve_snippets(question, self._corpus, top_k=top_k)
        snippet_block = format_snippet_block(snippets)

        if self._dspy_program is not None:
            start = time.perf_counter()
            prediction = self._dspy_program(question=question, snippets=snippet_block)
            latency = time.perf_counter() - start
            return Result(
                output={"answer": getattr(prediction, "answer", "")},
                token_usage=TokenUsage(prompt_tokens=0, completion_tokens=0),
                latency_seconds=latency,
            )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPTS[variant]},
            {
                "role": "user",
                "content": (f"Question: {question}\n\nRelevant snippets:\n{snippet_block}"),
            },
        ]

        start = time.perf_counter()
        completion = asyncio.run(
            self._client.chat_structured(
                self._settings.model_name,
                messages,
                QAAnswer,
                temperature=temperature,
            )
        )
        response = completion.content
        usage = completion.token_usage
        latency = time.perf_counter() - start

        return Result(
            output=response.model_dump(),
            token_usage=TokenUsage(
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
            ),
            latency_seconds=latency,
        )


def sample_qa_config(trial) -> dict[str, Any]:
    """Sample a config for the QA app: temperature, top_k, prompt variant."""
    return {
        "temperature": trial.suggest_float("temperature", 0.0, 1.0),
        "top_k": trial.suggest_int("top_k", 1, 5),
        "system_prompt_variant": trial.suggest_categorical(
            "system_prompt_variant", ["strict", "verbose"]
        ),
    }


QA_DEFAULT_CONFIG = {
    "temperature": 0.0,
    "top_k": 3,
    "system_prompt_variant": "strict",
}

QA_WEIGHTS = {
    "fuzzy_match": 0.4,
    "llm_judge": 0.3,
    "latency": 0.15,
    "cost": 0.15,
}


def _qa_dspy_factory(corpus: list[str], settings: Settings) -> DspyProgramSpec:
    def build():
        dspy = __import__("dspy")
        return dspy.Predict("question, snippets -> answer")

    def prepare_example(case: EvalCase):
        dspy = __import__("dspy")
        question = case.input.get("question", "")
        snippets = retrieve_snippets(question, corpus, top_k=3)
        snippet_block = format_snippet_block(snippets)
        example = dspy.Example(
            question=question,
            snippets=snippet_block,
            answer=case.expected if isinstance(case.expected, str) else str(case.expected),
        ).with_inputs("question", "snippets")
        example[CASE_ATTR] = case
        return example

    def prediction_to_output(pred) -> dict:
        return {"answer": getattr(pred, "answer", "")}

    return DspyProgramSpec(
        build=build,
        prepare_example=prepare_example,
        prediction_to_output=prediction_to_output,
    )


def _build_adapter(client: LLMClient, settings: Settings, program_path: str | None = None) -> QAApp:
    return QAApp(client, load_corpus(DATASET_PATH), settings, program_path=program_path)


def _metrics_factory(client: LLMClient, settings: Settings) -> list[Metric]:
    return [
        FuzzyMatchMetric(),
        LLMJudgeMetric(client=client, model=settings.model_name),
        LatencyMetric(),
        CostMetric(),
    ]


register_app(
    AppRegistration(
        name="qa",
        build_adapter=_build_adapter,
        metrics_factory=_metrics_factory,
        search_space=sample_qa_config,
        default_config=QA_DEFAULT_CONFIG,
        weights=QA_WEIGHTS,
        dataset_path=DATASET_PATH,
        dspy_factory=lambda settings: _qa_dspy_factory(load_corpus(DATASET_PATH), settings),
    )
)
