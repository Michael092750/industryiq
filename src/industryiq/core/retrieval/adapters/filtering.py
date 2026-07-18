"""Relevance-filtering adapters: implementations of the :class:`RelevanceFilter` port.

The post-retrieval coverage gate as a swappable strategy. Swap in a reranker or a
quorum rule without touching :class:`~industryiq.core.retrieval.service.RetrievalService`.
"""

from industryiq.core.retrieval.ports import RelevanceFilter
from industryiq.core.vectorstore import Hit, ScoreKind


class ThresholdFilter(RelevanceFilter):
    """Keep hits whose score clears a threshold *for that score's scale*; drop the rest.

    A threshold is only meaningful against a known scale (:class:`ScoreKind`), so
    the cutoff is chosen per hit by its ``score_kind``. Cosine scores are
    cross-query comparable and get the configured ``threshold``. BM25 and
    normalized-blend scores are query-relative -- a cosine cutoff would pass every
    BM25 hit and slice a fixed fraction off every normalized set -- so by default
    they are *not* value-filtered (they still ride the top-``k`` cap from
    retrieval). Pass ``thresholds`` to set a cutoff for such a kind once a
    benchmark has calibrated one; a kind left out of the map is never dropped.
    """

    def __init__(
        self,
        threshold: float = 0.0,
        *,
        thresholds: dict[ScoreKind, float] | None = None,
    ) -> None:
        # ``threshold`` is the cosine cutoff (keeps the single-float constructor
        # working); ``thresholds`` overrides/extends per kind. A kind absent from
        # the map maps to -inf, i.e. "keep everything of this kind".
        self._thresholds: dict[ScoreKind, float] = {
            ScoreKind.COSINE: threshold,
            **(thresholds or {}),
        }

    @classmethod
    def from_settings(
        cls,
        cosine: float = 0.0,
        *,
        bm25: float | None = None,
        normalized: float | None = None,
    ) -> "ThresholdFilter":
        """Build a filter from per-scale cutoffs, dropping the ones left unset.

        This is the one place that maps the optional BM25 / normalized config
        values onto their :class:`ScoreKind`; ``None`` means "no cutoff for that
        kind" (the config default), so the kind stays out of the map and is never
        value-filtered. Keeps the settings-to-kind knowledge here rather than at
        every :class:`ChatService` construction site.
        """
        per_kind = {
            kind: value
            for kind, value in ((ScoreKind.BM25, bm25), (ScoreKind.NORMALIZED, normalized))
            if value is not None
        }
        return cls(cosine, thresholds=per_kind or None)

    def keep(self, hits: list[Hit]) -> list[Hit]:
        return [
            hit for hit in hits if hit.score >= self._thresholds.get(hit.score_kind, float("-inf"))
        ]
