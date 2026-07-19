"""Context-expansion adapters: implementations of the :class:`ContextExpander` port.

Neighbour (context-window) expansion recovers a fact that sits in the chunk *before*
or *after* the matched one: retrieval stays precise on small chunks, but each hit's
``text`` is widened to include its neighbours before generation. ``chunk_index`` is
contiguous per ``source`` (see
:meth:`~industryiq.core.retrieval.retriever.Retriever.index`), so a hit's neighbours
are ``(source, chunk_index ± radius)``, fetched via
:meth:`~industryiq.core.vectorstore.VectorStore.fetch_neighbors`.
"""

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from industryiq.core.retrieval.ports import ContextExpander
from industryiq.core.vectorstore import Hit, VectorStore

_JOIN = "\n\n"


class NoOpExpander(ContextExpander):
    """Return hits unchanged -- the default (expansion off) and a test double."""

    def expand(self, hits: list[Hit]) -> list[Hit]:
        return hits


class NeighborExpander(ContextExpander):
    """Stitch each hit together with its adjacent chunks from the same source.

    Hits are processed in input (rank) order, so a higher-ranked hit claims its
    neighbours first; a lower-ranked hit whose *center* was already stitched into an
    earlier one is dropped (absorbed), and any overlap in its window is skipped -- so
    no chunk's text is ever duplicated and ``k`` stays meaningful. Only ``text`` is
    widened; the emitted hit keeps the matched chunk's id/score/score_kind, so
    citations and scoring are unchanged.

    Hits without a ``source``/``chunk_index`` (e.g. session uploads served from a
    different store) pass through untouched.

    * ``radius`` -- neighbours per side (0 disables; then this is a no-op).
    * ``max_chunks`` -- cap on the odd window width per hit, so ``k`` hits can't blow
      the context window; the effective radius is trimmed to honour it. ``None`` = no cap.
    * ``clamp_section`` -- when set, a window stops at a ``section`` boundary, so an
      unrelated heading's text is never pulled in.
    """

    def __init__(
        self,
        store: VectorStore,
        *,
        radius: int = 1,
        max_chunks: int | None = 5,
        clamp_section: bool = False,
    ) -> None:
        self._store = store
        self._clamp_section = clamp_section
        # Honour the width cap by trimming the effective radius (odd window <= max_chunks).
        self._radius = radius if max_chunks is None else min(radius, (max_chunks - 1) // 2)

    def expand(self, hits: list[Hit]) -> list[Hit]:
        if self._radius <= 0 or not hits:
            return hits
        # position -> (source, center_index) for the hits we can expand.
        expandable: dict[int, tuple[str, int]] = {}
        for pos, hit in enumerate(hits):
            source = hit.metadata.get("source")
            index = hit.metadata.get("chunk_index")
            if isinstance(source, str) and isinstance(index, int):
                expandable[pos] = (source, index)
        if not expandable:
            return hits
        fetched = self._fetch(expandable.values())

        out: list[Hit] = []
        consumed: dict[str, set[int]] = defaultdict(set)
        for pos, hit in enumerate(hits):
            if pos not in expandable:
                out.append(hit)
                continue
            source, center = expandable[pos]
            if center in consumed[source]:
                continue  # absorbed by an earlier, higher-ranked overlapping window
            out.append(self._widen(hit, source, center, fetched[source], consumed[source]))
        return out

    def _fetch(self, centers: Iterable[tuple[str, int]]) -> dict[str, dict[int, dict[str, Any]]]:
        """Fetch every needed neighbour, one round-trip per source."""
        needed: dict[str, set[int]] = defaultdict(set)
        for source, index in centers:
            for i in range(max(0, index - self._radius), index + self._radius + 1):
                needed[source].add(i)
        return {
            source: self._store.fetch_neighbors(source, sorted(indices))
            for source, indices in needed.items()
        }

    def _widen(
        self,
        hit: Hit,
        source: str,
        center: int,
        fetched: dict[int, dict[str, Any]],
        consumed: set[int],
    ) -> Hit:
        """Build the stitched window for one hit and mark its chunks consumed."""
        center_section = fetched.get(center, {}).get("section") or hit.metadata.get("section")
        texts: list[str] = []
        for i in self._window(center, fetched, center_section):
            if i in consumed:
                continue  # already stitched into an earlier hit's text; don't duplicate
            consumed.add(i)
            # The center's text is on the hit itself if the store didn't return it.
            meta = fetched.get(i)
            text = meta.get("text") if meta is not None else None
            if text is None and i == center:
                text = hit.metadata.get("text")
            if text:
                texts.append(text)
        widened = _JOIN.join(texts) if texts else hit.metadata.get("text", "")
        return Hit(
            id=hit.id,
            score=hit.score,
            metadata={**hit.metadata, "text": widened},
            score_kind=hit.score_kind,
        )

    def _window(
        self, center: int, fetched: dict[int, dict[str, Any]], center_section: object
    ) -> list[int]:
        """The contiguous index window around ``center``, in reading order.

        Extends up to ``radius`` each way, stopping at a missing chunk (a gap in the
        store) and -- when ``clamp_section`` is set and the center has a section -- at
        the first neighbour in a different section.
        """
        indices = [center]
        clamp = self._clamp_section and center_section
        for step in (-1, 1):
            i = center + step
            while abs(i - center) <= self._radius and i >= 0 and i in fetched:
                if clamp and fetched[i].get("section") != center_section:
                    break
                indices.append(i)
                i += step
        return sorted(indices)
