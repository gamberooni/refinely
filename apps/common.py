"""Shared helpers for demo apps: deterministic keyword/hybrid snippet retrieval over an in-memory corpus."""

import re

from refinely.core.exceptions import EvalError

STOP_WORDS = {
    "the",
    "and",
    "for",
    "with",
    "what",
    "how",
    "who",
    "which",
    "that",
    "this",
    "your",
    "you",
    "are",
    "was",
    "can",
    "does",
    "from",
    "where",
}

STRATEGIES = ("keyword", "hybrid")


def _keywords(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9']+", text.lower()) if len(w) > 2}


def _question_substrings(question: str) -> set[str]:
    substrings: set[str] = set()
    lower_q = question.lower()
    for i in range(len(lower_q)):
        for j in range(i + 4, min(i + 12, len(lower_q) + 1)):
            substrings.add(lower_q[i:j])
    return substrings


def retrieve_snippets_indexed(
    question: str,
    corpus: list[str],
    top_k: int = 3,
    strategy: str = "hybrid",
) -> list[tuple[int, str]]:
    """Return up to `top_k` (corpus index, snippet) pairs best matching the question.

    Scoring: +3 per shared keyword (excluding stop words); the hybrid strategy
    adds +1 per shared substring of the question (length >= 4) found in the
    snippet. Corpus order breaks ties, preferring earlier snippets.
    """
    if strategy not in STRATEGIES:
        raise EvalError(f"Unknown retrieval strategy: {strategy!r}")
    q_keywords = _keywords(question) - STOP_WORDS
    q_substrings = _question_substrings(question)

    scored: list[tuple[float, int, int, str]] = []
    for idx, snippet in enumerate(corpus):
        s_lower = snippet.lower()
        score = 0.0
        for kw in q_keywords:
            if kw in s_lower:
                score += 3.0
        if strategy == "hybrid":
            for sub in q_substrings:
                if sub in s_lower:
                    score += 1.0
        if score > 0:
            scored.append((score, -idx, idx, snippet))

    scored.sort(key=lambda t: (-t[0], t[1]))
    return [(idx, snippet) for _, _, idx, snippet in scored[:top_k]]


def retrieve_snippets(
    question: str,
    corpus: list[str],
    top_k: int = 3,
) -> list[str]:
    """Return up to `top_k` corpus snippets best matching the question (hybrid)."""
    return [s for _, s in retrieve_snippets_indexed(question, corpus, top_k)]


def format_snippet_block(snippets: list[str], labels: list[int] | None = None) -> str:
    """Format snippets as a labeled block for LLM prompts.

    Labels default to 1-based ordinals; pass explicit `labels` (e.g. corpus
    indices) to label snippets differently.
    """
    if labels is None:
        labels = list(range(1, len(snippets) + 1))
    return "\n\n".join(f"[snippet {label}] {snippet}" for label, snippet in zip(labels, snippets))
