"""Capabilities: the named units of work an agent dispatches to.

Each capability satisfies the :class:`~industryiq.core.agents.ports.Capability`
port -- a ``name``, a *prescriptive* ``description`` the planner routes on, and a
``run`` that turns an opaque ``inputs`` dict into a uniform
:class:`~industryiq.core.agents.models.CapabilityResult`. Two real capabilities
today: :class:`IndustryAnalysisCapability` (a mini-RAG over the report corpus) and
:class:`WebSearchCapability` (Anthropic's server-side web search, for current /
external facts the reports don't cover); a database lookup slots in later behind
the same seam.

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
from industryiq.core.retrieval.ports import CorpusRetriever
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
    question, then ground an answer on them. Retrieval goes through the shared
    :class:`~industryiq.core.retrieval.ports.CorpusRetriever` -- the *same* tuned
    core (strategy router + reranker) the chat retrieve-tier uses -- so this tool
    inherits that tuning rather than reimplementing it. Grounding reuses
    :func:`~industryiq.core.generation.generate_answer`, so a subtask answer is as
    grounded as a chat answer. Returns the answer as ``summary`` and the retrieved
    chunks as ``sources`` (so the final synthesis stays cited).
    """

    name = "industry_analysis"
    description = (
        "Answer a specific question about ONE industry/sector using the internal "
        "report library (market sizes, forecasts, adoption and investment figures). "
        "Use one call per distinct industry the request names. "
        'inputs: {"industry": "<sector>", "question": "<focused question>"}.'
    )

    def __init__(self, corpus: CorpusRetriever, llm: LLM, *, k: int = 6) -> None:
        self._corpus = corpus
        self._llm = llm
        self._k = k

    def run(self, inputs: dict[str, Any]) -> CapabilityResult:
        question = str(inputs.get("question") or inputs.get("query") or "").strip()
        industry = str(inputs.get("industry") or "").strip()
        query = f"{industry}: {question}" if industry else question
        hits = self._corpus.retrieve_corpus(query, k=self._k)
        answer = generate_answer(query, hits, self._llm)
        return CapabilityResult(
            summary=answer,
            data={"industry": industry, "chunks": len(hits)},
            sources=[_source_of(hit) for hit in hits],
        )


# --- web search (Anthropic server-side tool) --------------------------------------

# Server-tool type with dynamic filtering (Opus 4.6+/Sonnet 4.6+). Older models
# would need the basic "web_search_20250305"; the app pins a current model.
WEB_SEARCH_TOOL = "web_search_20260209"


def _web_text(content: list[Any]) -> str:
    """Concatenate the answer text blocks of a web-search response."""
    return "".join(
        getattr(block, "text", "") for block in content if getattr(block, "type", None) == "text"
    )


def _web_sources(content: list[Any]) -> list[dict[str, Any]]:
    """Collect cited URLs from a web-search response, de-duped, order preserved.

    Sources come from two places: the ``web_search_tool_result`` blocks (the raw
    results Claude searched) and the ``citations`` Claude attached to its answer
    text. Both are folded into the same ``{source, title}`` envelope other tools use.
    """
    seen: set[str] = set()
    out: list[dict[str, Any]] = []

    def _add(url: str | None, title: str | None) -> None:
        if url and url not in seen:
            seen.add(url)
            out.append({"source": url, "title": title})

    for block in content:
        btype = getattr(block, "type", None)
        if btype == "web_search_tool_result":
            results = getattr(block, "content", None)
            if isinstance(results, list):  # a list of results (an error is a single object)
                for result in results:
                    _add(getattr(result, "url", None), getattr(result, "title", None))
        elif btype == "text":
            for citation in getattr(block, "citations", None) or []:
                _add(getattr(citation, "url", None), getattr(citation, "title", None))
    return out


class WebSearchCapability(Capability):
    """Answer a question with Anthropic's *server-side* web search.

    Declares the ``web_search`` server tool on a single ``messages.create`` call;
    Anthropic runs the searches and returns the grounded answer with citations --
    no client-side tool loop. A long search turn can stop with
    ``stop_reason == "pause_turn"``; we resume by re-sending with the paused
    assistant turn appended. Returns the answer as ``summary`` and the cited URLs
    as ``sources``, so a planned web subtask is as citable as a retrieval subtask.

    The Anthropic client is imported lazily (only when no ``client`` is injected),
    so this module stays import-light and unit-testable with a fake client.
    """

    name = "web_search"
    description = (
        "Search the public web for CURRENT or external information the internal report "
        "library does not cover -- recent news, latest or post-cutoff figures, or facts "
        "about companies and markets outside the ingested reports. Use it for anything "
        "time-sensitive or clearly not in a market-research report. "
        'inputs: {"question": "<what to look up on the web>"}.'
    )

    def __init__(
        self,
        *,
        model_id: str,
        api_key: str | None = None,
        max_tokens: int = 2048,
        max_searches: int = 5,
        tool_type: str = WEB_SEARCH_TOOL,
        client: Any | None = None,
    ) -> None:
        self._model_id = model_id
        self._max_tokens = max_tokens
        self._max_searches = max_searches
        self._tool_type = tool_type
        if client is not None:
            self._client: Any = client
        else:
            # Lazy: only importing the capability with a real client pulls in anthropic.
            from anthropic import Anthropic

            self._client = Anthropic(api_key=api_key)

    def run(self, inputs: dict[str, Any]) -> CapabilityResult:
        question = str(inputs.get("question") or inputs.get("query") or "").strip()
        messages: list[dict[str, Any]] = [{"role": "user", "content": question}]
        tools = [{"type": self._tool_type, "name": "web_search", "max_uses": self._max_searches}]
        message: Any = None
        # Server-tool loop: resume a paused search turn (bounded so a stuck loop ends).
        for _ in range(self._max_searches + 2):
            message = self._client.messages.create(
                model=self._model_id,
                max_tokens=self._max_tokens,
                messages=messages,
                tools=tools,
            )
            if getattr(message, "stop_reason", None) != "pause_turn":
                break
            messages.append({"role": "assistant", "content": message.content})
        content = list(getattr(message, "content", None) or [])
        return CapabilityResult(
            summary=_web_text(content),
            data={"tool": self._tool_type},
            sources=_web_sources(content),
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
