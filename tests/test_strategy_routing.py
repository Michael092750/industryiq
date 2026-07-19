"""Unit tests for search-strategy routing: the adapters + the prompt/response contract.

All offline -- the LLM router is exercised with a stub returning canned JSON, and the
parser is tested directly, so no provider or network is involved.
"""

from industryiq.core.retrieval.adapters.strategy import FixedStrategyRouter, LlmStrategyRouter
from industryiq.core.retrieval.prompting import (
    build_condense_prompt,
    build_strategy_prompt,
    parse_strategy_plan,
)
from industryiq.core.vectorstore import SearchPlan, SearchStrategy


class StubLLM:
    """An LLM double returning a canned generate() response, recording prompts."""

    def __init__(self, response: str) -> None:
        self._response = response
        self.prompts: list[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self._response


# --- FixedStrategyRouter ------------------------------------------------------


def test_fixed_router_defaults_to_hybrid_rrf() -> None:
    assert FixedStrategyRouter().select("anything").is_default()


def test_fixed_router_returns_its_configured_plan() -> None:
    plan = SearchPlan(strategy=SearchStrategy.SEMANTIC)
    assert FixedStrategyRouter(plan).select("anything") is plan


# --- LlmStrategyRouter --------------------------------------------------------


def test_llm_router_parses_the_model_plan() -> None:
    llm = StubLLM('{"strategy": "lexical", "weights": null, "filter": null}')
    plan = LlmStrategyRouter(llm, "industry reports").select("GDPR Article 22")
    assert plan.strategy is SearchStrategy.LEXICAL


def test_llm_router_prompt_includes_kb_description_and_question() -> None:
    llm = StubLLM('{"strategy": "hybrid_rrf"}')
    LlmStrategyRouter(llm, "widget filings").select("what is a widget?")
    assert "widget filings" in llm.prompts[0]
    assert "what is a widget?" in llm.prompts[0]


def test_llm_router_falls_back_to_default_on_garbage() -> None:
    plan = LlmStrategyRouter(StubLLM("not json at all"), "reports").select("q")
    assert plan.is_default()


# --- parse_strategy_plan ------------------------------------------------------


def test_parse_semantic() -> None:
    assert parse_strategy_plan('{"strategy": "semantic"}').strategy is SearchStrategy.SEMANTIC


def test_parse_weighted_keeps_weights() -> None:
    plan = parse_strategy_plan('{"strategy": "hybrid_weighted", "weights": [0.4, 0.6]}')
    assert plan.strategy is SearchStrategy.HYBRID_WEIGHTED
    assert plan.weights == (0.4, 0.6)


def test_parse_weights_ignored_for_non_weighted_strategy() -> None:
    # Weights only make sense for hybrid_weighted; drop them otherwise.
    plan = parse_strategy_plan('{"strategy": "semantic", "weights": [0.4, 0.6]}')
    assert plan.weights is None


def test_parse_filter_publisher_and_date() -> None:
    plan = parse_strategy_plan(
        '{"strategy": "hybrid_rrf", "filter": {"publisher": "McKinsey", "published_from": "2024"}}'
    )
    assert plan.filter is not None
    assert plan.filter.publisher == "McKinsey"
    assert plan.filter.published_from == "2024"


def test_parse_empty_filter_becomes_none_and_plan_stays_default() -> None:
    plan = parse_strategy_plan('{"strategy": "hybrid_rrf", "filter": {"publisher": ""}}')
    assert plan.filter is None
    assert plan.is_default()


def test_parse_unknown_strategy_defaults_to_hybrid_rrf() -> None:
    assert parse_strategy_plan('{"strategy": "magic"}').strategy is SearchStrategy.HYBRID_RRF


def test_parse_tolerates_prose_and_fences_around_json() -> None:
    response = 'Sure!\n```json\n{"strategy": "lexical"}\n```'
    assert parse_strategy_plan(response).strategy is SearchStrategy.LEXICAL


def test_parse_garbage_returns_default_plan() -> None:
    assert parse_strategy_plan("no json here").is_default()


# --- prompt contracts ---------------------------------------------------------


def test_condense_prompt_instructs_literal_preservation() -> None:
    prompt = build_condense_prompt([], "what about GDPR?")
    assert "do not paraphrase" in prompt.lower()


def test_strategy_prompt_lists_every_strategy() -> None:
    prompt = build_strategy_prompt("q", "reports")
    for strategy in SearchStrategy:
        assert strategy.value in prompt
