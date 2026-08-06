"""Apply the grounding gate to a fan-out's shape.

:mod:`industryiq.core.grounding` defines the check itself -- the
:class:`~industryiq.core.grounding.GroundingGate` port plus pure citation helpers --
over a ``(question, answer, list[Hit])`` triple. A planned run does not have that
shape, for two reasons that are both consequences of fanning out:

* a node returns a :class:`~industryiq.core.agents.models.CapabilityResult` whose
  ``sources`` are JSON dicts, not ``Hit`` s -- they have to survive the task queue and
  the blackboard. :func:`hits_from_sources` rebuilds the triple so a node's answer is
  checkable exactly like a retrieve-tier one.
* every node numbers its citations ``[1]..[k]`` over *its own* retrieved chunks, so in
  a synthesized answer node A's ``[1]`` and node B's ``[1]`` are different documents.
  :func:`renumber_result` rewrites each node's markers into the merged source list's
  numbering, which is what makes a citation check on the final text mean anything.

:func:`check_node_result` is the executor-seam call -- one choke point that both the
in-process executor and the distributed worker make between running a capability and
writing its result to the blackboard. Gating there rather than at synthesis is
deliberate: at the node the *full* retrieved chunk text is still present and the
citation numbering is still local and coherent, so the check has everything it needs;
by synthesis both have been merged and deduped.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Any

from industryiq.core.agents.models import CapabilityResult
from industryiq.core.grounding import GroundingGate, renumber_citations
from industryiq.core.vectorstore import Hit


def source_label(source: Mapping[str, Any]) -> str:
    """The identity a citation dict is merged and de-duplicated on."""
    return str(source.get("source"))


def node_question(inputs: Mapping[str, Any], fallback: str) -> str:
    """The question a node was asked, for the gate's benefit.

    ``inputs`` is opaque to the executor by design (the :class:`Capability` port says
    so), so this reads the conventional keys the built-in capabilities use and falls
    back to the *run's* question rather than guessing. Only an LLM gate reads it -- the
    deterministic one ignores the question entirely -- so an imperfect fallback costs
    context in a judge prompt, never correctness.
    """
    for key in ("question", "query"):
        value = inputs.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return fallback


def hits_from_sources(sources: Iterable[Mapping[str, Any]]) -> list[Hit]:
    """Rebuild the ``list[Hit]`` the gate expects from a result's citation dicts.

    The inverse of ``capabilities._source_of``. Lossy by construction -- a citation
    dict carries what a *citation* needs (label, score, the grounding text) rather
    than a full hit -- but it carries the two fields the gate reads, which is the
    point of retaining ``text`` on the way out.
    """
    return [
        Hit(
            id=source_label(src),
            score=float(src.get("score") or 0.0),
            metadata={"source": src.get("source"), "text": str(src.get("text") or "")},
        )
        for src in sources
    ]


def check_node_result(
    gate: GroundingGate | None, question: str, result: CapabilityResult
) -> CapabilityResult:
    """Gate one node's result on its way to the blackboard.

    ``None`` (the default everywhere) returns the result untouched, so an unwired or
    offline run behaves exactly as before. An ungrounded node is **replaced by its
    abstention** rather than dropped, so synthesis can say "no data on X" instead of
    silently answering a narrower question than the plan asked; either way the verdict
    is recorded in ``data`` so the ledger, the synthesizer, and a future replan loop
    can all see it.

    Fabricated ``[n]`` markers are *not* patched here -- they are dropped structurally
    when :func:`renumber_result` re-indexes the node into the merged namespace -- so
    this only records them.
    """
    if gate is None:
        return result
    if not _is_verifiable(result):
        return result
    verdict = gate.check(question, result.summary, hits_from_sources(result.sources))
    if verdict.grounded:
        return result
    data = dict(result.data or {})
    data["grounded"] = False
    data["grounding_reason"] = verdict.reason
    if verdict.abstention is None:
        return replace(result, data=data)
    return replace(result, summary=verdict.abstention, data=data)


def _is_verifiable(result: CapabilityResult) -> bool:
    """Whether this result carries the grounding text a check would read.

    Cited-but-textless results are a real case, not a bug: ``web_search`` runs
    server-side, so the provider holds the page content and only URLs come back. There
    is nothing local to check such an answer against, and running a gate on empty text
    would not be a strict check -- it would be a **wrong** one, since an LLM judge
    handed no context judges every claim unsupported and abstains from a perfectly good
    answer. So pass those through.

    An empty ``sources`` list is the opposite case and is *not* excused: no citations at
    all is precisely the "nothing grounded this" signal the gate exists to catch.
    """
    return not result.sources or any(str(src.get("text") or "") for src in result.sources)


def global_citation_index(merged: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    """Map each merged source's label to its 1-based position in the merged list."""
    return {source_label(src): i for i, src in enumerate(merged, start=1)}


def renumber_result(result: CapabilityResult, index: Mapping[str, int]) -> CapabilityResult:
    """Rewrite one node's local ``[n]`` markers into the merged global numbering.

    ``result.sources`` is built in hit order, so the node's local marker ``[n]`` is
    ``sources[n - 1]`` -- that positional correspondence is the whole mapping, and it
    is why :func:`hits_from_sources`' ordering must never be disturbed.
    """
    labels = [source_label(src) for src in result.sources]
    return replace(result, summary=renumber_citations(result.summary, labels, index))
