"""Pure prompt construction for retrieval -- no I/O, no provider, fully testable.

Currently just the query-condense prompt used by
:class:`~industryiq.core.retrieval.adapters.rewriting.LlmQueryRewriter`. Kept in
the retrieval package (rather than chat) so the rewriter adapter never imports
``chat``. The shared :func:`~industryiq.core.conversation.format_history` helper
comes from the neutral conversation module.
"""

from industryiq.core.conversation import Turn, format_history


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
