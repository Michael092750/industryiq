"""Pure prompt construction (and response parsing) for retrieval -- no I/O, no
provider, fully testable.

Holds the query-condense prompt used by
:class:`~industryiq.core.retrieval.adapters.rewriting.LlmQueryRewriter` and the
search-strategy prompt + parser used by
:class:`~industryiq.core.retrieval.adapters.strategy.LlmStrategyRouter`. Kept in
the retrieval package (rather than chat) so the adapters never import ``chat``.
The shared :func:`~industryiq.core.conversation.format_history` helper comes from
the neutral conversation module.
"""

import json
from typing import Any

from industryiq.core.conversation import Turn, format_history
from industryiq.core.vectorstore import MetadataFilter, SearchPlan, SearchStrategy

_STRATEGY_VALUES = {strategy.value for strategy in SearchStrategy}
_FILTER_KEYS = ("publisher", "source_type", "category", "published_from", "published_to")


def build_condense_prompt(history: list[Turn], question: str) -> str:
    """Prompt that rewrites a follow-up into a standalone question.

    If the question already stands alone, the model is told to return it
    unchanged. It is also told to preserve literal tokens verbatim: paraphrasing
    away a quoted phrase, acronym, code, or name would strip the exact term a
    lexical (BM25) search needs, so this keeps the rewrite safe for every strategy.
    """
    return (
        "Given the conversation so far and a follow-up question, rewrite the "
        "follow-up as a standalone question that can be understood without the "
        "conversation. If it already stands alone, return it unchanged. Keep any "
        "quoted phrases, acronyms, codes, numbers, and proper names exactly as "
        "written -- do not paraphrase them. Respond with only the rewritten "
        "question.\n\n"
        f"Conversation:\n{format_history(history)}\n\n"
        f"Follow-up: {question}\n\n"
        "Standalone question:"
    )


def build_strategy_prompt(question: str, kb_description: str) -> str:
    """Prompt that classifies a standalone question into a search :class:`SearchPlan`.

    Asks the model for a strict JSON object naming the strategy, optional fusion
    weights, and an optional metadata filter; :func:`parse_strategy_plan` turns the
    response back into a :class:`SearchPlan`. ``kb_description`` grounds the choice
    in what the corpus actually holds.
    """
    return (
        f"You route searches over a knowledge base of {kb_description}. Decide how "
        "to search for the question below.\n\n"
        "Choose exactly one strategy:\n"
        '- "semantic": conceptual or paraphrased questions where meaning matters '
        "more than exact wording.\n"
        '- "lexical": the question is essentially a bare exact term, acronym, code, '
        "or proper name to find verbatim.\n"
        '- "hybrid_weighted": a specific term embedded in a question; also give '
        '"weights" as [dense, sparse] (raise the sparse weight when exact terms '
        "matter more).\n"
        '- "hybrid_rrf": the safe default for anything mixed or unclear.\n\n'
        "Optionally add a filter when the question explicitly scopes one:\n"
        '- "publisher": a named source/organisation (e.g. "according to McKinsey").\n'
        '- "published_from"/"published_to": ISO dates for recency (e.g. "since 2024").\n'
        '- "category"/"source_type": only when explicitly stated.\n\n'
        "Respond with ONLY a JSON object, no prose:\n"
        '{"strategy": "hybrid_rrf", "weights": null, "filter": null}\n\n'
        f"Question: {question}\n"
        "JSON:"
    )


def _extract_json(response: str) -> str:
    """Slice out the first ``{...}`` object from a model response (tolerates fences/prose)."""
    start = response.find("{")
    end = response.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("no JSON object found")
    return response[start : end + 1]


def _coerce_weights(value: Any) -> tuple[float, float] | None:
    if isinstance(value, list | tuple) and len(value) == 2:
        try:
            return (float(value[0]), float(value[1]))
        except (TypeError, ValueError):
            return None
    return None


def _coerce_filter(value: Any) -> MetadataFilter | None:
    if not isinstance(value, dict):
        return None
    fields = {
        key: value[key]
        for key in _FILTER_KEYS
        if isinstance(value.get(key), str) and value[key].strip()
    }
    metadata_filter = MetadataFilter(**fields)
    return None if metadata_filter.is_empty() else metadata_filter


def parse_strategy_plan(response: str) -> SearchPlan:
    """Parse a strategy response into a :class:`SearchPlan`, defaulting on any error.

    Deliberately forgiving: a malformed/hallucinated response falls back to the
    default plan (hybrid-RRF, no filter) rather than raising, so a router hiccup
    degrades to today's behaviour instead of breaking the turn.
    """
    try:
        obj = json.loads(_extract_json(response))
    except (ValueError, TypeError):
        return SearchPlan()
    if not isinstance(obj, dict):
        return SearchPlan()
    raw_strategy = obj.get("strategy")
    strategy = (
        SearchStrategy(raw_strategy)
        if raw_strategy in _STRATEGY_VALUES
        else SearchStrategy.HYBRID_RRF
    )
    weights = (
        _coerce_weights(obj.get("weights")) if strategy is SearchStrategy.HYBRID_WEIGHTED else None
    )
    return SearchPlan(strategy=strategy, filter=_coerce_filter(obj.get("filter")), weights=weights)
