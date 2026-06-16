"""Relevance-filtering adapters: implementations of the :class:`RelevanceFilter` port.

The post-retrieval coverage gate as a swappable strategy, symmetric with the
pre-retrieval :class:`RetrievalRouter`. Swap in a reranker or a quorum rule
without touching :class:`ChatService`.
"""

from ragproject.core.chat.ports import RelevanceFilter
from ragproject.core.vectorstore import Hit


class ThresholdFilter(RelevanceFilter):
    """Keep hits whose similarity score clears a threshold; drop the rest."""

    def __init__(self, threshold: float = 0.0) -> None:
        self._threshold = threshold

    def keep(self, hits: list[Hit]) -> list[Hit]:
        return [hit for hit in hits if hit.score >= self._threshold]
