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

    The strategy descriptions spell out each method's mechanism, the question shape
    it suits, and the answer shape it recovers, so the router picks on signal rather
    than defaulting to a blanket weighted blend. Two failure modes are called out
    explicitly because they cost the most recall: (1) exact-figure lookups belong in
    pure ``lexical`` -- the dense half of a hybrid search buries a chunk the
    embedding model can't match; (2) a year that is part of the *fact* being asked
    ("sales in 2020") is content, not a publication date, and must never become a
    date filter -- a report about a year is routinely published in another year.
    """
    return (
        f"You are the search router for a retrieval system over a knowledge base of "
        f"{kb_description}. Every document is chunked and indexed two ways: as dense "
        "vector embeddings (semantic similarity) and in a BM25 lexical index (exact "
        "keyword/token match). Pick the ONE strategy most likely to surface the chunk "
        "that answers the question, plus an optional metadata filter, and return JSON.\n\n"
        "STRATEGIES -- choose exactly one:\n\n"
        '- "hybrid_rrf" -- THE DEFAULT. Runs the dense AND the lexical search and fuses '
        "them by rank, so the answer only has to score well in one of them. Robust for "
        "anything that mixes concepts with specific terms, or whenever you are unsure. "
        "Most questions belong here.\n"
        "    Question shape: a normal factual or descriptive question that names a topic "
        "and some specifics.\n"
        '    Example: "What did the report say about risks from unrealized securities '
        'losses?"\n\n'
        '- "lexical" -- BM25 keyword match only. Choose it ONLY when the answer is pinned '
        "down by a RARE, distinctive literal token the question already carries: a named "
        "company/product tied to a figure, an acronym, a section/product code, or a quoted "
        'phrase (e.g. "US semiconductor company sales" -> "$208 billion"). Lexical drops '
        "the dense signal entirely, so it only wins when that literal token alone identifies "
        "the right chunk.\n"
        "    Do NOT choose lexical for questions built from COMMON domain vocabulary -- "
        '"efficiency ratio", "how many ... projects", "sample size", generic ratios / counts '
        "/ growth. Those words match many chunks, so BM25 cannot pick the right one and the "
        "dense signal is needed: use hybrid_rrf. When torn between lexical and hybrid_rrf, "
        "choose hybrid_rrf.\n"
        '    Answer shape: one exact number, name, or string (e.g. "$208 billion").\n'
        '    Example: "What did US semiconductor company sales total in 2020?"\n\n'
        '- "semantic" -- dense embeddings only. Choose it for conceptual, definitional, '
        "or paraphrased questions that share few exact words with the source and have no "
        "distinctive figure or name to key on. Weak at matching exact numbers.\n"
        "    Question shape: a why/how question about a concept or mechanism.\n"
        '    Example: "Why are semiconductor supply chains considered fragile?"\n\n'
        '- "hybrid_weighted" -- dense + lexical fused by weighted score, with '
        '"weights" as [dense, sparse]. Only if you specifically want to blend meaning and '
        "keywords AND tilt the balance; otherwise prefer hybrid_rrf, which is more robust. "
        "Rarely the right choice.\n\n"
        "METADATA FILTER -- optional; default to null.\n"
        "Add a filter ONLY when the question itself restricts the source. A wrong filter "
        "removes the answer entirely, so when in doubt use null.\n"
        '- "publisher": the source\'s registrable web DOMAIN, not a display name -- the '
        'corpus tags publisher by domain (e.g. "mckinsey.com", "deloitte.com", '
        '"weforum.org", "imf.org"). Set it only when the question restricts to the '
        'ORGANISATION THAT AUTHORED the report ("in the McKinsey report", "per the IMF\'s '
        'outlook"); map that organisation to its domain. A source merely CITED inside the '
        'answer -- "according to Moody\'s", "the Fed estimates" -- is usually quoted by '
        "another publisher's report, not the author, so leave publisher null unless that "
        "org clearly authored the document.\n"
        '- "published_from"/"published_to": ISO dates, and ONLY for questions about '
        'PUBLICATION recency itself -- "the latest / most recent report", "published since '
        '2024".\n'
        '    CRITICAL: a year that is part of the FACT being asked -- "sales in 2020", '
        '"exports in 2023", "incidents in 2025" -- is NOT a publication date. A report '
        "about a year is routinely published in a different year (a 2025 report can cover "
        "2023; a 2022 outlook can discuss 2023). Never set a date filter for such a year; "
        "leave it null and let the search match the content.\n"
        '- "category"/"source_type": only when explicitly stated in the question.\n\n'
        "Return ONLY a JSON object, no prose, in exactly this shape:\n"
        '{"strategy": "hybrid_rrf", "weights": null, "filter": null}\n\n'
        "Examples:\n"
        'Q: "According to Deloitte, what is the outlook for consumer spending in 2024?"\n'
        '{"strategy": "hybrid_rrf", "weights": null, "filter": {"publisher": "deloitte.com"}}\n'
        'Q: "What did US semiconductor company sales total in 2020?"\n'
        '{"strategy": "lexical", "weights": null, "filter": null}\n'
        'Q: "How much did food and agriculture export sales to China total in 2023?"\n'
        '{"strategy": "lexical", "weights": null, "filter": null}\n'
        'Q: "What did the community bank efficiency ratio rise to, and by how many basis points?"\n'
        '{"strategy": "hybrid_rrf", "weights": null, "filter": null}\n'
        'Q: "How many AI-related GitHub projects were there in 2025?"\n'
        '{"strategy": "hybrid_rrf", "weights": null, "filter": null}\n'
        'Q: "What did the most recent AI Index report say about private investment?"\n'
        '{"strategy": "hybrid_rrf", "weights": null, "filter": {"published_from": "2025-01-01"}}\n'
        'Q: "Why are semiconductor supply chains considered fragile?"\n'
        '{"strategy": "semantic", "weights": null, "filter": null}\n\n'
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
