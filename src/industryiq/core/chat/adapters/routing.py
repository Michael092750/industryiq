"""Turn-routing adapters: implementations of the :class:`TurnRouter` port.

* :class:`AlwaysRetrieveRouter` -- always a simple knowledge-base lookup, never
  planning. Deterministic, the offline default and a convenient test double.
* :class:`LlmRouter` -- ask an LLM to classify the turn into the three tiers
  (none / simple / complex). The production choice: skip retrieval on small talk,
  and escalate a genuinely complex question to the agent planner.
"""

from industryiq.core.chat.models import RouteDecision, Turn
from industryiq.core.chat.ports import TurnRouter
from industryiq.core.chat.prompting import build_route_prompt
from industryiq.core.generation import LLM


class AlwaysRetrieveRouter(TurnRouter):
    """Always a simple knowledge-base lookup; never escalate to planning."""

    def route(self, history: list[Turn], question: str) -> RouteDecision:
        return RouteDecision(should_retrieve=True, needs_planning=False)


class LlmRouter(TurnRouter):
    """Classify, with an LLM, which tier a turn needs: none / simple / complex.

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
        if verdict.startswith("complex"):
            return RouteDecision(should_retrieve=True, needs_planning=True)
        if verdict.startswith("no"):  # NONE / no
            return RouteDecision(should_retrieve=False, needs_planning=False)
        # SIMPLE -- and any unrecognized verdict -- errs toward a plain retrieve.
        return RouteDecision(should_retrieve=True, needs_planning=False)
