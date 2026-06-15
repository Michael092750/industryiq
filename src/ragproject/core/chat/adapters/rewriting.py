"""Query-rewriting adapters: implementations of the :class:`QueryRewriter` port.

* :class:`LlmQueryRewriter` -- uses an LLM to condense follow-ups (production).
* :class:`NoOpQueryRewriter` -- returns the question unchanged; the trivial
  Liskov-substitutable implementation, useful as a default and a test double.
"""

from ragproject.core.chat.models import Turn
from ragproject.core.chat.prompting import build_condense_prompt
from ragproject.core.generation import LLM


class LlmQueryRewriter:
    """Condense follow-up questions into standalone ones with an LLM."""

    def __init__(self, llm: LLM) -> None:
        self._llm = llm

    def condense(self, history: list[Turn], question: str) -> str:
        # Nothing to condense against on the first turn -- skip the LLM call.
        if not history:
            return question
        rewritten = self._llm.generate(build_condense_prompt(history, question)).strip()
        return rewritten or question


class NoOpQueryRewriter:
    """Pass questions through unchanged (no history awareness)."""

    def condense(self, history: list[Turn], question: str) -> str:
        return question
