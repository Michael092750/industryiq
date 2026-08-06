"""Synthesis: compose the final grounded answer from the per-node results.

Shared by both executors. It reads each node's :class:`CapabilityResult` off the
blackboard, merges (and de-dups) their citations, and produces one answer -- either
LLM-composed when an LLM is supplied, or a deterministic concatenation (the offline
default, used in tests). Keeping citations through the merge is what stops the
fan-out from losing the grounding the single-shot path has.

Merging citations is not just concatenation: each node numbered its ``[n]`` markers
over its *own* retrieved chunks, so :meth:`Synthesizer._prepare` re-indexes every
node into the merged list's numbering before a prompt is built. Skip that and the
combined answer contains several colliding ``[1]`` s -- unresolvable for a reader and
uncheckable for the gate.

:meth:`Synthesizer.synthesize` returns the whole :class:`RunResult` (for
``/agents/run``); :meth:`Synthesizer.stream` yields the answer token by token (for
the chat turn), over the same prompt. The difference matters to grounding: only the
non-streamed path can still *replace* an ungrounded answer, so it abstains where
:meth:`stream` -- whose tokens are already gone -- can only append a caveat.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from typing import Any

from industryiq.core.agents.grounding import (
    global_citation_index,
    hits_from_sources,
    renumber_result,
    source_label,
)
from industryiq.core.agents.models import CapabilityResult, Plan, RunResult
from industryiq.core.agents.ports import Blackboard
from industryiq.core.generation import GenerativeLLM
from industryiq.core.grounding import GroundingGate, citation_caveat, verify_citations

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
    """One entry per distinct source label, accumulating each one's grounding text.

    De-duping by label keeps the reader's citation list document-shaped (three chunks
    of one report are one source, not three). But two nodes -- or two chunks -- can
    ground *different* claims in the same document, so their texts are joined rather
    than the first winning: the merged entry has to carry everything that supports a
    citation to it, or a check against it fails answers it should pass.
    """
    at: dict[str, int] = {}
    merged: list[dict[str, Any]] = []
    for sources in source_lists:
        for src in sources:
            key = str(src.get("source"))
            if key not in at:
                at[key] = len(merged)
                merged.append(dict(src))
                continue
            existing = merged[at[key]]
            kept, addition = str(existing.get("text") or ""), str(src.get("text") or "")
            if addition and addition not in kept:
                existing["text"] = "\n\n".join(part for part in (kept, addition) if part)
    return merged


def _ordered(
    plan: Plan, results: Mapping[str, CapabilityResult]
) -> list[tuple[str, CapabilityResult]]:
    """The results that landed, in plan-node order."""
    return [(node.node_id, results[node.node_id]) for node in plan.nodes if node.node_id in results]


class Synthesizer:
    """Turn the per-node results into one answer.

    LLM-composed when an ``llm`` is given; otherwise a deterministic concatenation
    -- so an executor can be unit-tested end to end with no network. An optional
    ``grounding`` gate checks the *composed* answer: the nodes were already checked
    individually at the executor seam, but composition is its own opportunity to
    fabricate, so the root gets its own check. ``None`` (the default) leaves both
    paths byte-for-byte as they were.
    """

    def __init__(
        self, llm: GenerativeLLM | None = None, *, grounding: GroundingGate | None = None
    ) -> None:
        self._llm = llm
        self._grounding = grounding

    def synthesize(self, plan: Plan, results: Mapping[str, CapabilityResult]) -> RunResult:
        ordered, merged = self._prepare(plan, results)
        if not ordered:
            answer = _EMPTY
        else:
            if self._llm is None:
                answer = _concat(ordered)
            else:
                answer = self._llm.generate(_synthesis_prompt(plan.question, ordered, merged))
            # Not streamed, so an ungrounded answer can still be withheld -- this is
            # the one answer path where the gate's abstention works as designed.
            answer = self._gate(plan.question, answer, merged)
        return RunResult(
            run_id=plan.run_id,
            question=plan.question,
            answer=answer,
            sources=merged,
            completed=tuple(node_id for node_id, _result in ordered),
            failed=tuple(node.node_id for node in plan.nodes if node.node_id not in results),
        )

    def stream(self, plan: Plan, results: Mapping[str, CapabilityResult]) -> Iterator[str]:
        """Yield the synthesized answer token by token (chunks concatenate to the
        same text :meth:`synthesize` produces, absent a grounding replacement)."""
        ordered, merged = self._prepare(plan, results)
        if not ordered:
            yield _EMPTY
            return
        if self._llm is None:
            answer = _concat(ordered)
            yield answer
        else:
            parts: list[str] = []
            for token in self._llm.stream(_synthesis_prompt(plan.question, ordered, merged)):
                parts.append(token)
                yield token
            answer = "".join(parts)
        # The tokens are already out, so -- exactly as on the streamed chat retrieve
        # tier -- a bad citation can only be flagged, never replaced.
        if self._grounding is not None:
            invalid = verify_citations(answer, hits_from_sources(merged))
            if invalid:
                yield citation_caveat(invalid)

    def _prepare(
        self, plan: Plan, results: Mapping[str, CapabilityResult]
    ) -> tuple[list[tuple[str, CapabilityResult]], list[dict[str, Any]]]:
        """Order the landed results and re-index their citations into one namespace.

        Returns the results with global ``[n]`` markers plus the merged source list
        those markers point into -- the pair every downstream step needs, computed
        once so the prompt, the answer, and the gate can never disagree about what
        ``[3]`` means.
        """
        ordered = _ordered(plan, results)
        merged = merge_sources(result for _node_id, result in ordered)
        index = global_citation_index(merged)
        return [(node_id, renumber_result(r, index)) for node_id, r in ordered], merged

    def _gate(self, question: str, answer: str, merged: list[dict[str, Any]]) -> str:
        """Abstain from, or caveat, a composed answer that its sources do not support."""
        if self._grounding is None:
            return answer
        verdict = self._grounding.check(question, answer, hits_from_sources(merged))
        if verdict.grounded:
            return answer
        if verdict.abstention is not None:
            return verdict.abstention
        return answer + citation_caveat(verdict.invalid_citations)


def _concat(ordered: list[tuple[str, CapabilityResult]]) -> str:
    return "\n\n".join(f"[{node_id}] {result.summary}" for node_id, result in ordered)


def _synthesis_prompt(
    question: str,
    ordered: list[tuple[str, CapabilityResult]],
    merged: list[dict[str, Any]],
) -> str:
    """Assemble the synthesis prompt (pure; no I/O).

    The findings arrive already re-indexed into ``merged``'s numbering, so the source
    list is spelled out and the model is told to reuse those markers rather than
    "keep" whatever it sees -- the old wording invited it to carry through node-local
    numbers that no longer mean anything here.
    """
    sources = "\n".join(f"[{i}] {source_label(src)}" for i, src in enumerate(merged, start=1))
    parts = "\n\n".join(f"Subtask {node_id}:\n{result.summary}" for node_id, result in ordered)
    return (
        "Combine the subtask findings below into one coherent, cited answer to the "
        "question. Cite only with the [n] markers from the Sources list -- reuse the "
        "markers already present in the findings and never invent a new number.\n\n"
        f"Question: {question}\n\n"
        f"Sources:\n{sources}\n\n"
        f"{parts}\n\nAnswer:"
    )
