"""Pure prompt construction for chat -- no I/O, no provider, fully testable.

Separated from :class:`ChatService` (Single Responsibility): the service decides
*when* to build a prompt; these functions decide *what* it says. Mirrors the
existing :func:`industryiq.core.generation.build_prompt`.
"""

from industryiq.core.chat.models import Turn
from industryiq.core.vectorstore import Hit


def format_history(history: list[Turn]) -> str:
    """Render prior turns as alternating ``User:`` / ``Assistant:`` lines."""
    lines: list[str] = []
    for turn in history:
        lines.append(f"User: {turn.question}")
        lines.append(f"Assistant: {turn.answer}")
    return "\n".join(lines)


def build_condense_prompt(history: list[Turn], question: str) -> str:
    """Prompt that rewrites a follow-up into a standalone question.

    If the question already stands alone, the model is told to return it
    unchanged.
    """
    return (
        "Given the conversation so far and a follow-up question, rewrite the "
        "follow-up as a standalone question that can be understood without the "
        "conversation. If it already stands alone, return it unchanged. "
        "Respond with only the rewritten question.\n\n"
        f"Conversation:\n{format_history(history)}\n\n"
        f"Follow-up: {question}\n\n"
        "Standalone question:"
    )


def build_route_prompt(history: list[Turn], question: str, kb_description: str) -> str:
    """Prompt that asks whether a question needs a knowledge-base lookup.

    ``kb_description`` tells the model what the knowledge base contains, so it can
    judge whether the question is in scope instead of guessing blind.
    """
    history_block = f"Conversation so far:\n{format_history(history)}\n\n" if history else ""
    return (
        f"The knowledge base contains {kb_description}. Decide whether answering "
        "the question needs information looked up from it. Questions about its "
        "subject matter do; greetings, small talk, and questions about this "
        "conversation itself do not. Answer with only 'yes' or 'no'.\n\n"
        f"{history_block}"
        f"Question: {question}\n\n"
        "Needs knowledge base (yes/no):"
    )


def build_chat_prompt(history: list[Turn], question: str, hits: list[Hit]) -> str:
    """Assemble the grounded answer prompt from history, question, and chunks.

    Each chunk is numbered so the model can cite it as ``[n]``.
    """
    if hits:
        context = "\n".join(
            f"[{i}] {hit.metadata.get('text', '')}" for i, hit in enumerate(hits, start=1)
        )
    else:
        context = "(no relevant context found)"
    history_block = f"Conversation so far:\n{format_history(history)}\n\n" if history else ""
    return (
        "You are a helpful assistant. Answer the question using only the context "
        "below and the conversation so far. Cite sources as [n]. If the context "
        "does not contain the answer, say so.\n\n"
        f"{history_block}"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
