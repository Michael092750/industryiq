"""Search-strategy adapters: implementations of the :class:`SearchStrategyRouter` port.

* :class:`FixedStrategyRouter` -- always returns one plan (default hybrid-RRF).
  Deterministic, the offline default and a convenient test double; equivalent to
  today's single-strategy behaviour.
* :class:`LlmStrategyRouter` -- classify, with an LLM, how to search: strategy +
  optional metadata filter + fusion weights. The production choice.
"""

from industryiq.core.generation import LLM
from industryiq.core.retrieval.ports import SearchStrategyRouter
from industryiq.core.retrieval.prompting import build_strategy_prompt, parse_strategy_plan
from industryiq.core.vectorstore import SearchPlan


class FixedStrategyRouter(SearchStrategyRouter):
    """Always return the same :class:`SearchPlan` (default: hybrid-RRF, no filter).

    Reproduces today's single-strategy retrieval; the safe default when no LLM
    router is configured, and the deterministic double tests build against.
    """

    def __init__(self, plan: SearchPlan | None = None) -> None:
        self._plan = plan if plan is not None else SearchPlan()

    def select(self, question: str) -> SearchPlan:
        return self._plan


class LlmStrategyRouter(SearchStrategyRouter):
    """Classify, with an LLM, how to search for a question.

    One generation call per turn returns a strategy (dense / lexical / hybrid),
    optional fusion weights, and an optional metadata filter, parsed back into a
    :class:`SearchPlan`. ``kb_description`` grounds the choice in what the corpus
    holds. A malformed response falls back to the default plan (see
    :func:`~industryiq.core.retrieval.prompting.parse_strategy_plan`), so a router
    hiccup degrades to today's hybrid-RRF behaviour rather than breaking the turn.
    """

    def __init__(self, llm: LLM, kb_description: str) -> None:
        self._llm = llm
        self._kb_description = kb_description

    def select(self, question: str) -> SearchPlan:
        prompt = build_strategy_prompt(question, self._kb_description)
        return parse_strategy_plan(self._llm.generate(prompt))
