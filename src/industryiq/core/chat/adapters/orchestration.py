"""AgentTurnOrchestrator: answer a complex chat turn via the agent planner.

The bridge from chat to the agent orchestrator (``chat -> agents``, one way). It
condenses the follow-up into a standalone question, plans it into subtasks over the
capability registry, runs the fan-out on a :class:`PlanExecutor` (in-process or
distributed, chosen at wiring time), and streams the synthesized answer -- yielding
the same ``StreamStatus`` / ``StreamStart`` / ``StreamToken`` events the simple chat
path does, so :class:`~industryiq.core.chat.service.ChatService` forwards them and
stays the sole owner of persistence + the final ``StreamEnd``.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any
from uuid import uuid4

from industryiq.core.agents.ports import Capability, PlanExecutor, Planner
from industryiq.core.agents.synthesis import Synthesizer, merge_sources
from industryiq.core.chat.models import StreamEvent, StreamStart, StreamStatus, StreamToken, Turn
from industryiq.core.chat.ports import TurnOrchestrator
from industryiq.core.retrieval.ports import QueryRewriter
from industryiq.core.vectorstore import Hit


class AgentTurnOrchestrator(TurnOrchestrator):
    """Plan -> fan out -> stream synthesis, for a complex chat turn."""

    def __init__(
        self,
        rewriter: QueryRewriter,
        planner: Planner,
        registry: Mapping[str, Capability],
        executor: PlanExecutor,
        synthesizer: Synthesizer,
    ) -> None:
        self._rewriter = rewriter
        self._planner = planner
        self._registry = registry
        self._executor = executor
        self._synthesizer = synthesizer

    def run_stream(self, history: list[Turn], question: str) -> Iterator[StreamEvent]:
        yield StreamStatus(phase="planning")
        # The planner is stateless; condense the follow-up against history first so
        # it plans over a standalone question (reuses the retrieval rewriter).
        standalone = self._rewriter.condense(history, question)
        catalog = {name: capability.description for name, capability in self._registry.items()}
        plan = self._planner.plan(uuid4().hex, standalone, catalog)

        yield StreamStatus(phase="retrieving")
        results = self._executor.execute(plan)  # fan-out fills the blackboard
        sources = merge_sources(results.values())
        yield StreamStart(standalone_question=standalone, hits=_hits_from_sources(sources))

        yield StreamStatus(phase="generating")
        for token in self._synthesizer.stream(plan, results):
            yield StreamToken(text=token)


def _hits_from_sources(sources: list[dict[str, Any]]) -> list[Hit]:
    """Map merged citation dicts to lightweight ``Hit`` stand-ins for ``StreamStart``,
    so the chat route serializes a complex turn's sources exactly like a simple turn's.
    """
    return [
        Hit(
            id=str(src.get("source") or ""),
            score=float(src.get("score") or 0.0),
            metadata={"source": src.get("source"), "text": ""},
        )
        for src in sources
    ]
