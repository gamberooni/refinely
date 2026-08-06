import asyncio
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel

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
    MetricResult,
)
from crucible.llm.client import LLMClient
from crucible.llm.usage import Result, TokenUsage
from crucible.registry import AppRegistration, register_app
from crucible.retrieval import format_snippet_block, retrieve_snippets_indexed

DATASET_PATH = Path(__file__).resolve().parents[1] / "datasets" / "rag_v1.json"


def _indices(value: Any) -> set[int]:
    if not isinstance(value, list):
        return set()
    return {int(v) for v in value}


class RetrievalRecallMetric:
    """Fraction of expected source snippets present in the retrieved set."""

    name = "retrieval_recall"

    def evaluate(self, case: EvalCase, output: Result) -> MetricResult:
        expected_idx = _indices(
            case.expected.get("source_indices") if isinstance(case.expected, dict) else None
        )
        retrieved_idx = _indices(
            output.output.get("retrieved_indices") if isinstance(output.output, dict) else None
        )
        score = 0.0
        if expected_idx:
            score = len(expected_idx & retrieved_idx) / len(expected_idx)
        return MetricResult(
            metric_name=self.name,
            value=score,
            raw={
                "expected_indices": sorted(expected_idx),
                "retrieved_indices": sorted(retrieved_idx),
            },
        )


class CitationAccuracyMetric:
    """Fraction of cited snippets that are expected sources (citation precision)."""

    name = "citation_accuracy"

    def evaluate(self, case: EvalCase, output: Result) -> MetricResult:
        expected_idx = _indices(
            case.expected.get("source_indices") if isinstance(case.expected, dict) else None
        )
        cited_idx = _indices(
            output.output.get("cited_indices") if isinstance(output.output, dict) else None
        )
        score = 0.0
        if cited_idx:
            score = len(cited_idx & expected_idx) / len(cited_idx)
        return MetricResult(
            metric_name=self.name,
            value=score,
            raw={
                "cited_indices": sorted(cited_idx),
                "expected_indices": sorted(expected_idx),
            },
        )


class RAGAnswer(BaseModel):
    answer: str
    cited_snippets: list[int]


class SnippetScores(BaseModel):
    scores: list[int]


SYSTEM_PROMPTS: dict[str, str] = {
    "strict": (
        "You are a concise RAG assistant. Answer the question using ONLY the "
        "provided snippets, and cite each claim with the number of the snippet "
        "that supports it. If the snippets do not contain the answer, say so."
    ),
    "verbose": (
        "You are a helpful RAG assistant. Answer the question using the "
        "provided snippets, explaining your reasoning in 1-3 sentences, and "
        "cite each claim with the number of the snippet that supports it."
    ),
}


class RAGApp:
    """RAG pipeline: optional query expansion, retrieval, optional rerank, generation.

    All LLM calls for a case run inside a single event loop; token usage and
    latency are aggregated across stages.
    """

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
                _rag_dspy_factory(self._corpus),
                program_path,
                self._settings,
                temperature=RAG_DEFAULT_CONFIG.get("temperature", 0.0),
            )

    def execute(self, input: dict, config: dict) -> Result:
        return asyncio.run(self._execute_async(input, config))

    async def _execute_async(self, input: dict, config: dict) -> Result:
        temperature = float(config.get("temperature", 0.0))
        top_k = int(config.get("top_k", 3))
        variant = str(config.get("system_prompt_variant", "strict"))
        strategy = str(config.get("retrieval_strategy", "hybrid"))
        expand = bool(config.get("query_expansion", False))
        rerank = bool(config.get("rerank", False))
        if variant not in SYSTEM_PROMPTS:
            raise EvalError(f"Unknown system_prompt_variant: {variant!r}")

        question = str(input.get("question", ""))
        prompt_tokens = 0
        completion_tokens = 0
        start = time.perf_counter()

        if expand:
            question, usage = await self._expand_query(question, temperature)
            prompt_tokens += usage.prompt_tokens
            completion_tokens += usage.completion_tokens

        candidates = retrieve_snippets_indexed(
            question, self._corpus, top_k=top_k, strategy=strategy
        )

        if rerank and len(candidates) > 1:
            candidates, usage = await self._rerank(question, candidates, top_k, temperature)
            prompt_tokens += usage.prompt_tokens
            completion_tokens += usage.completion_tokens

        retrieved_indices = [idx for idx, _ in candidates]
        snippet_block = format_snippet_block(
            [snippet for _, snippet in candidates], labels=[idx for idx, _ in candidates]
        )

        if self._dspy_program is not None:
            prediction = self._dspy_program(question=question, snippets=snippet_block)
            latency = time.perf_counter() - start
            cited = getattr(prediction, "cited_snippets", [])
            if isinstance(cited, str):
                import json as _json

                try:
                    cited = _json.loads(cited)
                except _json.JSONDecodeError:
                    cited = []
            return Result(
                output={
                    "answer": getattr(prediction, "answer", ""),
                    "retrieved_indices": retrieved_indices,
                    "cited_indices": cited if isinstance(cited, list) else [],
                },
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

        response, usage = await self._client.chat_structured(
            self._settings.model_name,
            messages,
            RAGAnswer,
            temperature=temperature,
        )
        prompt_tokens += usage.prompt_tokens
        completion_tokens += usage.completion_tokens
        latency = time.perf_counter() - start

        return Result(
            output={
                "answer": response.answer,
                "retrieved_indices": retrieved_indices,
                "cited_indices": response.cited_snippets,
            },
            token_usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            ),
            latency_seconds=latency,
        )

    async def _expand_query(self, question: str, temperature: float) -> tuple[str, TokenUsage]:
        messages = [
            {
                "role": "system",
                "content": (
                    "Rewrite the question to maximize retrieval recall. "
                    "Return only the rewritten question."
                ),
            },
            {"role": "user", "content": f"Question: {question}"},
        ]
        text, usage = await self._client.chat_text(
            self._settings.model_name, messages, temperature=temperature
        )
        return text.strip(), usage

    async def _rerank(
        self,
        question: str,
        candidates: list[tuple[int, str]],
        top_k: int,
        temperature: float,
    ) -> tuple[list[tuple[int, str]], TokenUsage]:
        snippet_block = "\n\n".join(f"[{idx}] {snippet}" for idx, snippet in candidates)
        messages = [
            {
                "role": "system",
                "content": (
                    "Score each snippet 1-5 for relevance to the question. "
                    'Return ONLY JSON with a "scores" list in snippet order.'
                ),
            },
            {
                "role": "user",
                "content": f"Question: {question}\n\nSnippets:\n{snippet_block}",
            },
        ]
        response, usage = await self._client.chat_structured(
            self._settings.model_name,
            messages,
            SnippetScores,
            temperature=temperature,
        )
        scores = response.scores + [0] * (len(candidates) - len(response.scores))
        ranked = sorted(zip(candidates, scores), key=lambda pair: -pair[1])
        return [candidate for candidate, _ in ranked[:top_k]], usage


def sample_rag_config(trial) -> dict[str, Any]:
    """Sample a config for the RAG app: six mixed-type pipeline parameters."""
    return {
        "temperature": trial.suggest_float("temperature", 0.0, 1.0),
        "system_prompt_variant": trial.suggest_categorical(
            "system_prompt_variant", ["strict", "verbose"]
        ),
        "retrieval_strategy": trial.suggest_categorical(
            "retrieval_strategy", ["keyword", "hybrid"]
        ),
        "top_k": trial.suggest_int("top_k", 1, 6),
        "query_expansion": trial.suggest_categorical("query_expansion", [True, False]),
        "rerank": trial.suggest_categorical("rerank", [True, False]),
    }


RAG_DEFAULT_CONFIG = {
    "temperature": 0.0,
    "system_prompt_variant": "strict",
    "retrieval_strategy": "hybrid",
    "top_k": 3,
    "query_expansion": False,
    "rerank": False,
}

RAG_WEIGHTS = {
    "fuzzy_match": 0.2,
    "llm_judge": 0.2,
    "retrieval_recall": 0.25,
    "citation_accuracy": 0.1,
    "latency": 0.1,
    "cost": 0.15,
}


def _rag_dspy_factory(corpus: list[str]) -> DspyProgramSpec:
    def build():
        dspy = __import__("dspy")
        return dspy.Predict("question, snippets -> answer, cited_snippets")

    def prepare_example(case: EvalCase):
        dspy = __import__("dspy")
        question = case.input.get("question", "")
        candidates = retrieve_snippets_indexed(question, corpus, top_k=3, strategy="hybrid")
        snippet_block = format_snippet_block(
            [snippet for _, snippet in candidates], labels=[idx for idx, _ in candidates]
        )
        expected = case.expected if isinstance(case.expected, dict) else {}
        example = dspy.Example(
            question=question,
            snippets=snippet_block,
            answer=expected.get("answer", ""),
            cited_snippets=expected.get("source_indices", []),
        ).with_inputs("question", "snippets")
        example[CASE_ATTR] = case
        return example

    def prediction_to_output(pred) -> dict:
        cited = getattr(pred, "cited_snippets", [])
        if isinstance(cited, str):
            import json as _json

            try:
                cited = _json.loads(cited)
            except _json.JSONDecodeError:
                cited = []
        return {
            "answer": getattr(pred, "answer", ""),
            "retrieved_indices": [],
            "cited_indices": cited if isinstance(cited, list) else [],
        }

    return DspyProgramSpec(
        build=build,
        prepare_example=prepare_example,
        prediction_to_output=prediction_to_output,
    )


def _build_adapter(
    client: LLMClient, settings: Settings, program_path: str | None = None
) -> RAGApp:
    return RAGApp(client, load_corpus(DATASET_PATH), settings, program_path=program_path)


def _metrics_factory(client: LLMClient, settings: Settings) -> list[Metric]:
    return [
        FuzzyMatchMetric(),
        LLMJudgeMetric(client=client, model=settings.model_name),
        RetrievalRecallMetric(),
        CitationAccuracyMetric(),
        LatencyMetric(),
        CostMetric(),
    ]


register_app(
    AppRegistration(
        name="rag",
        build_adapter=_build_adapter,
        metrics_factory=_metrics_factory,
        search_space=sample_rag_config,
        default_config=RAG_DEFAULT_CONFIG,
        weights=RAG_WEIGHTS,
        dataset_path=DATASET_PATH,
        dspy_factory=lambda settings: _rag_dspy_factory(load_corpus(DATASET_PATH)),
    )
)
