"""Capabilities: the named units of work an agent dispatches to.

Each capability satisfies the :class:`~industryiq.core.agents.ports.Capability`
port -- a ``name``, a *prescriptive* ``description`` the planner routes on, and a
``run`` that turns an opaque ``inputs`` dict into a uniform
:class:`~industryiq.core.agents.models.CapabilityResult`. Today there is one real
capability (industry analysis = a mini-RAG over the report corpus); web search /
database lookup slot in later behind the same seam.

Also here: the demo ``FailureHook`` -- a tiny injectable seam a worker/executor
calls before running a node, so the "kill a worker mid-run" beat is reproducible
without actually killing an OS process. In production it is :func:`no_failure`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from industryiq.core.agents.models import CapabilityResult
from industryiq.core.agents.ports import Capability
from industryiq.core.generation import LLM, generate_answer
from industryiq.core.retrieval.ports import RetrievalPort
from industryiq.core.vectorstore import Hit


def _source_of(hit: Hit) -> dict[str, Any]:
    """Distil a hit into a citation dict (best-effort over varied metadata keys)."""
    md = hit.metadata
    return {
        "source": md.get("source") or md.get("doc") or md.get("title") or md.get("filename"),
        "score": round(float(hit.score), 4),
    }


class IndustryAnalysisCapability(Capability):
    """Answer a focused question about ONE industry from the report corpus.

    A mini-RAG: retrieve the most relevant chunks for the (industry-scoped)
    question, then ground an answer on them -- reusing the same retrieval + prompt
    machinery (:func:`~industryiq.core.generation.generate_answer`) the chat path
    uses, so a subtask answer is as grounded as a chat answer. Returns the answer
    as ``summary`` and the retrieved chunks as ``sources`` (so the final synthesis
    stays cited).
    """

    name = "industry_analysis"
    description = (
        "Answer a specific question about ONE industry/sector using the internal "
        "report library (market sizes, forecasts, adoption and investment figures). "
        "Use one call per distinct industry the request names. "
        'inputs: {"industry": "<sector>", "question": "<focused question>"}.'
    )

    def __init__(self, retriever: RetrievalPort, llm: LLM, *, k: int = 6) -> None:
        self._retriever = retriever
        self._llm = llm
        self._k = k

    def run(self, inputs: dict[str, Any]) -> CapabilityResult:
        question = str(inputs.get("question") or inputs.get("query") or "").strip()
        industry = str(inputs.get("industry") or "").strip()
        query = f"{industry}: {question}" if industry else question
        hits = self._retriever.retrieve(query, k=self._k)
        answer = generate_answer(query, hits, self._llm)
        return CapabilityResult(
            summary=answer,
            data={"industry": industry, "chunks": len(hits)},
            sources=[_source_of(hit) for hit in hits],
        )


# --- demo failure injection -------------------------------------------------------

# A hook the worker calls with the node id right before running its capability. It
# normally does nothing; a demo hook may raise to simulate a crash.
FailureHook = Callable[[str], None]


class WorkerCrash(RuntimeError):
    """Raised by a demo :data:`FailureHook` to simulate a worker dying on a node."""


def no_failure(node_id: str) -> None:
    """The production hook: never fails (the default everywhere but the demo)."""
    return None


class CrashOnceHook:
    """Fail the *first* execution of each targeted node, then let it pass.

    Simulates a worker that dies mid-node: the first attempt raises (the task is
    left unacked and becomes reclaimable), and the reclaiming worker's attempt
    succeeds -- exactly the crash -> reclaim -> resume path, made deterministic.
    ``nodes=None`` targets every node; pass a set to crash only specific ones.
    """

    def __init__(self, nodes: set[str] | None = None) -> None:
        self._nodes = nodes
        self._crashed: set[str] = set()

    def __call__(self, node_id: str) -> None:
        targeted = self._nodes is None or node_id in self._nodes
        if targeted and node_id not in self._crashed:
            self._crashed.add(node_id)
            raise WorkerCrash(f"simulated crash on node {node_id}")
