"""Retrieval-routing adapters: implementations of the :class:`RetrievalRouter` port.

* :class:`AlwaysRetrieveRouter` -- always consult the knowledge base. Deterministic,
  the offline default and a convenient test double.
* :class:`LlmRouter` -- ask an LLM whether the question needs a lookup (intent
  classification). The production choice for skipping retrieval on small talk.
"""

from industryiq.core.chat.models import RouteDecision, Turn
from industryiq.core.chat.ports import RetrievalRouter
from industryiq.core.chat.prompting import build_route_prompt
from industryiq.core.generation import LLM


class AlwaysRetrieveRouter(RetrievalRouter):
    """Always consult the knowledge base."""

    def route(self, history: list[Turn], question: str) -> RouteDecision:
        return RouteDecision(should_retrieve=True)


class LlmRouter(RetrievalRouter):
    """Classify, with an LLM, whether a question needs a knowledge-base lookup.

    ``kb_description`` is a short, human description of what the knowledge base
    holds (e.g. "industry analysis reports"). It is injected into the prompt so
    the model judges scope against the real corpus rather than guessing blind.
    """

    def __init__(self, llm: LLM, kb_description: str) -> None:
        self._llm = llm
        self._kb_description = kb_description

    def route(self, history: list[Turn], question: str) -> RouteDecision:
        prompt = build_route_prompt(history, question, self._kb_description)
        verdict = self._llm.generate(prompt).strip().lower()
        return RouteDecision(should_retrieve=verdict.startswith("y"))
