"""Grounding gate: is a generated answer actually supported by its context?

The post-generation faithfulness check -- the mirror image of the *pre*-retrieval
:class:`~industryiq.core.retrieval.ports.RelevanceFilter` ("did we retrieve anything
useful?"). This gate asks "is what we *said* backed by what we retrieved?" and lets a
caller **abstain** (ship a "not in the reports" reply) or **caveat** (flag a fabricated
citation) instead of shipping an ungrounded answer -- the difference between a RAG system
that *scales* and one that is *right*.

Two pieces, deliberately separated for testability (mirrors ``generation.py``):

* pure helpers -- :func:`cited_indices` / :func:`verify_citations` /
  :func:`renumber_citations` / :func:`citation_caveat` -- that parse, re-index, and
  report on the ``[n]`` markers a grounded prompt asks for. No I/O and exactly one
  correct implementation each, so *not* a Protocol (a port there would be
  over-abstraction). They are shape-agnostic (``str`` and ``list[Hit]`` in, no
  knowledge of chat turns or agent runs) so every answer path can share them.
* :class:`GroundingGate` -- the port a caller depends on, with two adapters:
  :class:`DeterministicGroundingGate` (the default -- network-free, so the offline/test
  path stays LLM-free) and :class:`LlmGroundingGate` (an opt-in faithfulness check that
  spends one extra model call). Both return a :class:`GroundingVerdict`.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from industryiq.core.generation import LLM
from industryiq.core.vectorstore import Hit

# Shipped in place of an answer when there is no grounded context to answer from.
DEFAULT_ABSTENTION = (
    "I couldn't find that in the reports I have access to, so I can't answer it "
    "reliably. Try rephrasing, or ask about a topic the reports cover."
)

_CITATION = re.compile(r"\[(\d+)\]")


def cited_indices(answer: str) -> list[int]:
    """Return the distinct ``[n]`` citation markers in ``answer``, in first-seen order."""
    seen: dict[int, None] = {}
    for match in _CITATION.finditer(answer):
        seen.setdefault(int(match.group(1)), None)
    return list(seen)


def verify_citations(answer: str, hits: list[Hit]) -> tuple[int, ...]:
    """Return the cited ``[n]`` markers that fall outside the numbered context.

    The grounded prompts number context ``[1]..[len(hits)]`` (see
    :func:`~industryiq.core.generation.build_prompt`), so a citation is valid iff
    ``1 <= n <= len(hits)``. Anything else is a fabricated reference the model invented.
    """
    upper = len(hits)
    return tuple(index for index in cited_indices(answer) if index < 1 or index > upper)


def citation_caveat(invalid: tuple[int, ...]) -> str:
    """A suffix note flagging citation markers that matched no retrieved source.

    Shared by every *streamed* answer path: once tokens are out, a bad citation can
    only be flagged, not replaced, so each of them appends this same note.
    """
    refs = ", ".join(f"[{index}]" for index in invalid)
    return f"\n\n_Note: {refs} could not be matched to a retrieved source and may be unreliable._"


def renumber_citations(answer: str, labels: Sequence[str], index: Mapping[str, int]) -> str:
    """Rewrite ``answer``'s local ``[n]`` markers into a shared global numbering.

    Needed wherever several independently-grounded answers are combined into one: each
    was numbered ``[1]..[len(labels)]`` over *its own* context, so without this pass
    one text's ``[1]`` and another's ``[1]`` mean different documents and no citation
    check on the combined answer can mean anything.

    ``labels[n - 1]`` is the source label local marker ``[n]`` refers to, and ``index``
    maps a label to its position in the combined list. A marker that resolves to
    neither -- a fabricated reference, or a label missing from ``index`` -- is
    **dropped**: a number that survives into the combined text would silently point at
    whichever document happens to sit at that position, and a wrong citation is worse
    than none. (The gate reports these separately via ``invalid_citations``; this is
    the structural repair, not the report.)
    """

    def _remap(match: re.Match[str]) -> str:
        local = int(match.group(1))
        if 1 <= local <= len(labels):
            target = index.get(labels[local - 1])
            if target is not None:
                return f"[{target}]"
        return ""

    return _tidy(_CITATION.sub(_remap, answer))


def _tidy(text: str) -> str:
    """Close the gaps a dropped ``[n]`` marker leaves behind."""
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]+([,.;:!?])", r"\1", text)
    return re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)


@dataclass(frozen=True)
class GroundingVerdict:
    """The outcome of a grounding check on one answer.

    ``grounded`` is the headline: is the answer supported by its context? When it is not
    and ``abstention`` is set, the caller should ship the abstention text *instead of*
    the answer (a replacement). When ``grounded`` is False only because of
    ``invalid_citations``, the caller can keep the answer but flag/prune the fabricated
    ``[n]`` markers (a caveat). ``reason`` is a short, loggable explanation.
    """

    grounded: bool
    invalid_citations: tuple[int, ...] = ()
    reason: str = ""
    abstention: str | None = None


@runtime_checkable
class GroundingGate(Protocol):
    """Decide whether an answer is grounded in the context it was given.

    Implementations decide how -- deterministic signals, an LLM faithfulness judge, a
    hybrid. The caller depends only on this seam, so the offline/test path can run the
    network-free gate and production can opt into the model-backed one, interchangeably.
    """

    def check(self, question: str, answer: str, hits: list[Hit]) -> GroundingVerdict: ...


class DeterministicGroundingGate(GroundingGate):
    """Offline, network-free grounding gate -- the default.

    Uses only signals it can compute without a model: an answer built on **no** context
    is not grounded (so abstain), and ``[n]`` markers pointing outside the numbered
    context are fabricated references (so flag). Cheap and reproducible; the LLM
    faithfulness check is the separate, opt-in :class:`LlmGroundingGate`.
    """

    def __init__(self, *, abstention: str = DEFAULT_ABSTENTION) -> None:
        self._abstention = abstention

    def check(self, question: str, answer: str, hits: list[Hit]) -> GroundingVerdict:
        if not hits:
            return GroundingVerdict(
                grounded=False, reason="no supporting context", abstention=self._abstention
            )
        invalid = verify_citations(answer, hits)
        if invalid:
            return GroundingVerdict(
                grounded=False,
                invalid_citations=invalid,
                reason=f"citations outside context 1-{len(hits)}: {list(invalid)}",
            )
        return GroundingVerdict(grounded=True)


def _format_context(hits: list[Hit]) -> str:
    """Number the passages ``[1]..`` exactly as the grounded prompts do."""
    return "\n".join(f"[{i}] {hit.metadata.get('text', '')}" for i, hit in enumerate(hits, start=1))


def _grounding_prompt(question: str, answer: str, hits: list[Hit]) -> str:
    """Assemble the faithfulness-judge prompt (pure; no I/O)."""
    return (
        "Decide whether the ANSWER is fully supported by the CONTEXT passages. Reply "
        "with ONE word: GROUNDED if every factual claim in the answer is supported by "
        "the context, or UNSUPPORTED if any claim is not.\n\n"
        f"Question: {question}\n\n"
        f"Context:\n{_format_context(hits)}\n\n"
        f"Answer:\n{answer}\n\n"
        "Verdict (GROUNDED/UNSUPPORTED):"
    )


class LlmGroundingGate(GroundingGate):
    """Faithfulness check via the LLM -- opt-in (one extra ``generate`` call per answer).

    Asks the model whether the context supports every claim, and combines that
    GROUNDED/UNSUPPORTED verdict with the deterministic citation check. Because it spends
    a model call (and latency), it is not the default; wire it where correctness outweighs
    cost. On empty context it short-circuits to abstention without calling the model.
    """

    def __init__(self, llm: LLM, *, abstention: str = DEFAULT_ABSTENTION) -> None:
        self._llm = llm
        self._abstention = abstention

    def check(self, question: str, answer: str, hits: list[Hit]) -> GroundingVerdict:
        if not hits:
            return GroundingVerdict(
                grounded=False, reason="no supporting context", abstention=self._abstention
            )
        invalid = verify_citations(answer, hits)
        raw = self._llm.generate(_grounding_prompt(question, answer, hits)).strip().upper()
        if raw.startswith("UNSUPPORTED"):
            return GroundingVerdict(
                grounded=False,
                invalid_citations=invalid,
                reason="llm judged the answer unsupported by context",
                abstention=self._abstention,
            )
        if invalid:
            return GroundingVerdict(
                grounded=False,
                invalid_citations=invalid,
                reason=f"citations outside context 1-{len(hits)}: {list(invalid)}",
            )
        return GroundingVerdict(grounded=True)
