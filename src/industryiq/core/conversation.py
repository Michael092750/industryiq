"""Neutral conversation primitives shared by the chat and retrieval packages.

:class:`Turn` is the one value object both sides need -- retrieval's query
rewriter condenses against conversation history, and chat orchestrates and
persists turns. It lives here, rather than in :mod:`industryiq.core.chat`, so
:mod:`industryiq.core.retrieval` can depend on it without importing ``chat`` --
keeping the package dependency strictly one-way (``chat -> retrieval``).

:func:`format_history` is the shared turn-rendering helper the prompt builders
on both sides reuse; it depends only on :class:`Turn`.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Turn:
    """One completed exchange in a conversation: a question and its answer."""

    question: str
    answer: str


def format_history(history: list[Turn]) -> str:
    """Render prior turns as alternating ``User:`` / ``Assistant:`` lines."""
    lines: list[str] = []
    for turn in history:
        lines.append(f"User: {turn.question}")
        lines.append(f"Assistant: {turn.answer}")
    return "\n".join(lines)
