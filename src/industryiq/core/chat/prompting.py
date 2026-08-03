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
    """Prompt that classifies a question into the three answer tiers.

    ``kb_description`` tells the model what the knowledge base contains, so it can
    judge scope instead of guessing blind. The reply is one word:
    ``NONE`` / ``SIMPLE`` / ``COMPLEX``.
    """
    history_block = f"Conversation so far:\n{format_history(history)}\n\n" if history else ""
    return (
        f"The knowledge base contains {kb_description}. Classify how to answer the "
        "question, replying with ONE word:\n"
        "- NONE: greetings, small talk, or questions about this conversation itself "
        "-- no lookup needed.\n"
        "- SIMPLE: one focused question answerable by a single knowledge-base lookup.\n"
        "- COMPLEX: needs several lookups, a comparison across multiple industries or "
        "entities, or multi-step reasoning.\n\n"
        f"{history_block}"
        f"Question: {question}\n\n"
        "Classification (NONE/SIMPLE/COMPLEX):"
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
