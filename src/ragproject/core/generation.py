"""Generation: turn a question + retrieved chunks into an answer.

Two pieces, deliberately separated for testability:

* :func:`build_prompt` -- a *pure* function that assembles the prompt text. No
  network, fully unit-testable.
* :class:`LLM` -- the interface for the actual text generator, with a
  :class:`FakeLLM` for tests. The real provider (e.g. Bedrock/Claude) is added
  later and only needs to satisfy this interface.
"""

import re
from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from ragproject.core.vectorstore import Hit


@runtime_checkable
class LLM(Protocol):
    """Anything that can turn a prompt string into a completion string."""

    def generate(self, prompt: str) -> str:
        """Return the model's completion for ``prompt``."""
        ...


@runtime_checkable
class StreamingLLM(Protocol):
    """Anything that can stream a completion as ordered text chunks."""

    def stream(self, prompt: str) -> Iterator[str]:
        """Yield the model's completion for ``prompt``, chunk by chunk."""
        ...


@runtime_checkable
class GenerativeLLM(LLM, StreamingLLM, Protocol):
    """An LLM that can both return a full completion and stream it.

    The provider concretes implement both; consumers depend on the narrowest
    port they need -- ``LLM`` for the rewriter/pipeline, ``StreamingLLM`` for the
    chat answer (Interface Segregation).
    """


class FakeLLM(GenerativeLLM):
    """Offline LLM for tests.

    Returns a fixed response and records the last prompt it received, so tests
    can assert both the output and exactly what prompt was sent.
    """

    def __init__(self, response: str = "FAKE_ANSWER") -> None:
        self._response = response
        self.last_prompt: str | None = None

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        return self._response

    def stream(self, prompt: str) -> Iterator[str]:
        # Emit word-sized chunks so tests exercise multi-chunk streaming; the
        # pieces concatenate back to exactly ``response``.
        self.last_prompt = prompt
        yield from re.findall(r"\S+\s*", self._response) or [self._response]


def build_prompt(query: str, hits: list[Hit]) -> str:
    """Assemble a grounded RAG prompt from a question and retrieved chunks.

    Each chunk is numbered so the model can cite it as ``[n]``.
    """
    if hits:
        context = "\n".join(
            f"[{i}] {hit.metadata.get('text', '')}" for i, hit in enumerate(hits, start=1)
        )
    else:
        context = "(no relevant context found)"
    return (
        "Answer the question using only the context below. "
        "Cite sources as [n]. If the context does not contain the answer, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )


def generate_answer(query: str, hits: list[Hit], llm: LLM) -> str:
    """Build the prompt from ``query`` + ``hits`` and return the LLM's answer."""
    return llm.generate(build_prompt(query, hits))
