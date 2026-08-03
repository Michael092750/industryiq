"""Synthesis: compose the final grounded answer from the per-node results.

Shared by both executors. It reads each node's :class:`CapabilityResult` off the
blackboard, merges (and de-dups) their citations, and produces one answer -- either
LLM-composed when an LLM is supplied, or a deterministic concatenation (the offline
default, used in tests). Keeping citations through the merge is what stops the
fan-out from losing the grounding the single-shot path has.

:meth:`Synthesizer.synthesize` returns the whole :class:`RunResult` (for
``/agents/run``); :meth:`Synthesizer.stream` yields the answer token by token (for
the chat turn), over the same prompt.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from industryiq.core.agents.models import CapabilityResult, Plan, RunResult
from industryiq.core.agents.ports import Blackboard
from industryiq.core.generation import GenerativeLLM

_EMPTY = "No results were produced for this request."


def collect_results(plan: Plan, blackboard: Blackboard) -> dict[str, CapabilityResult]:
    """Read back the results the workers/executor posted for ``plan``'s nodes."""
    results: dict[str, CapabilityResult] = {}
    for node in plan.nodes:
        raw = blackboard.read(plan.run_id, node.node_id)
        if isinstance(raw, dict):
            results[node.node_id] = CapabilityResult.from_dict(raw)
    return results


def merge_sources(results: Iterable[CapabilityResult]) -> list[dict[str, Any]]:
    """Flatten + de-dup the citations across results (by ``source`` label)."""
    return _dedup_sources(result.sources for result in results)


def _dedup_sources(source_lists: Iterable[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    merged: list[dict[str, Any]] = []
    for sources in source_lists:
        for src in sources:
            key = str(src.get("source"))
            if key not in seen:
                seen.add(key)
                merged.append(src)
    return merged


def _ordered(
    plan: Plan, results: Mapping[str, CapabilityResult]
) -> list[tuple[str, CapabilityResult]]:
    """The results that landed, in plan-node order."""
    return [(node.node_id, results[node.node_id]) for node in plan.nodes if node.node_id in results]


class Synthesizer:
    """Turn the per-node results into one answer.

    LLM-composed when an ``llm`` is given; otherwise a deterministic concatenation
    -- so an executor can be unit-tested end to end with no network.
    """

    def __init__(self, llm: GenerativeLLM | None = None) -> None:
        self._llm = llm

    def synthesize(self, plan: Plan, results: Mapping[str, CapabilityResult]) -> RunResult:
        ordered = _ordered(plan, results)
        if not ordered:
            answer = _EMPTY
        elif self._llm is None:
            answer = _concat(ordered)
        else:
            answer = self._llm.generate(_synthesis_prompt(plan.question, ordered))
        return RunResult(
            run_id=plan.run_id,
            question=plan.question,
            answer=answer,
            sources=merge_sources(result for _node_id, result in ordered),
            completed=tuple(node_id for node_id, _result in ordered),
            failed=tuple(node.node_id for node in plan.nodes if node.node_id not in results),
        )

    def stream(self, plan: Plan, results: Mapping[str, CapabilityResult]) -> Iterator[str]:
        """Yield the synthesized answer token by token (chunks concatenate to the
        same text :meth:`synthesize` produces)."""
        ordered = _ordered(plan, results)
        if not ordered:
            yield _EMPTY
        elif self._llm is None:
            yield _concat(ordered)
        else:
            yield from self._llm.stream(_synthesis_prompt(plan.question, ordered))


def _concat(ordered: list[tuple[str, CapabilityResult]]) -> str:
    return "\n\n".join(f"[{node_id}] {result.summary}" for node_id, result in ordered)


def _synthesis_prompt(question: str, ordered: list[tuple[str, CapabilityResult]]) -> str:
    parts = "\n\n".join(f"Subtask {node_id}:\n{result.summary}" for node_id, result in ordered)
    return (
        "Combine the subtask findings below into one coherent, cited answer to the "
        f"question; keep the [n] citations.\n\nQuestion: {question}\n\n{parts}\n\nAnswer:"
    )
