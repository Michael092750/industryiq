"""Pure prompt construction for chat -- no I/O, no provider, fully testable.

Separated from :class:`ChatService` (Single Responsibility): the service decides
*when* to build a prompt; these functions decide *what* it says. Mirrors the
existing :func:`industryiq.core.generation.build_prompt`.

Only the routing and answer prompts live here. The query-condense prompt is a
retrieval concern and lives in :mod:`industryiq.core.retrieval.prompting`; the
shared :func:`~industryiq.core.conversation.format_history` helper is imported
from the neutral conversation module.
"""

from industryiq.core.conversation import Turn, format_history
from industryiq.core.vectorstore import Hit


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
