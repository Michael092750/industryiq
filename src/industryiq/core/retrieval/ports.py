"""The retrieval package's ports -- the abstractions the retrieval side depends on.

Following the Dependency Inversion Principle, :class:`RetrievalService` is written
against these ``Protocol``s, never against concrete adapters. They were promoted
out of the chat package so retrieval is self-contained and a supervisor/agent can
compose it without pulling in ``chat``.

* :class:`RetrievalPort`, :class:`RelevanceFilter`, :class:`QueryRewriter`,
  :class:`SessionDocumentStore` -- the fine-grained collaborators the service
  composes.
* :class:`ContextRetriever` + :class:`RetrievalResult` -- the *coarse* seam a
  caller (e.g. :class:`~industryiq.core.chat.service.ChatService`) depends on:
  "gather the grounding context for this turn" in, a filtered/merged result out.
"""

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from industryiq.core.conversation import Turn
from industryiq.core.vectorstore import Hit, SearchPlan


@runtime_checkable
class RetrievalPort(Protocol):
    """Find the chunks most relevant to a query.

    Deliberately narrow (Interface Segregation): a retrieval consumer only ever
    *reads*, so it depends on ``retrieve`` alone, not on indexing. The concrete
    :class:`~industryiq.core.retrieval.retriever.Retriever` satisfies this
    structurally.

    ``plan`` selects the search strategy + metadata filter (a
    :class:`~industryiq.core.vectorstore.SearchPlan`); ``None`` means the default
    (hybrid-RRF, no filter), which every store can serve. A non-default plan
    requires a strategy-capable store or the retriever raises.
    """

    def retrieve(self, query: str, k: int = 5, plan: SearchPlan | None = None) -> list[Hit]: ...


@runtime_checkable
class RelevanceFilter(Protocol):
    """Decide which retrieved hits are relevant enough to ground the answer.

    The post-retrieval coverage gate ("did we actually find anything useful?"),
    symmetric with the pre-retrieval routing decision. Implementations decide how
    -- a score threshold, a reranker, a quorum rule.
    """

    def keep(self, hits: list[Hit]) -> list[Hit]: ...


@runtime_checkable
class QueryRewriter(Protocol):
    """Rewrite a follow-up question into a standalone one.

    "What about its pricing?" only makes sense given prior turns, but retrieval
    needs a self-contained query. Implementations decide how (LLM, no-op, ...).
    """

    def condense(self, history: list[Turn], question: str) -> str: ...


@runtime_checkable
class SearchStrategyRouter(Protocol):
    """Choose *how* to search for a (standalone) question -- a :class:`SearchPlan`.

    The pre-retrieval "retrieve how?" decision, downstream of the "which tier?"
    :class:`~industryiq.core.chat.ports.TurnRouter` and run on the already
    condensed query. Implementations decide the strategy (dense / lexical / hybrid),
    any metadata pre-filter, and fusion weights -- an LLM classifier, a heuristic, or
    a fixed default. Returning ``SearchPlan()`` reproduces today's hybrid-RRF path.
    """

    def select(self, question: str) -> SearchPlan: ...


@runtime_checkable
class Reranker(Protocol):
    """Reorder a wide candidate pool by reading each ``(query, chunk_text)`` pair.

    The Stage-2 *precision* step of a two-stage retrieve->rerank pipeline: Stage 1
    (dense + BM25, RRF-fused) casts a wide, cheap net for *recall*; a reranker then
    reads the *text* of each candidate and re-sorts by true relevance, returning the
    top ``k``. Unlike a fusion ranker (``RRFRanker`` / ``WeightedRanker``), which only
    combines the dense and BM25 *rank positions* and never sees the chunk text, a
    cross-encoder scores the ``(query, chunk)`` pair *jointly* -- so it can rescue a
    chunk that one leg ranked poorly (e.g. an exact-figure answer the 384-dim embedder
    blurs and RRF then buries).

    Composes with -- or *replaces* -- a :class:`SearchStrategyRouter`: reading the
    content makes the router's "which signal matters?" guess moot, at ~0 API calls (a
    local model, ~50-100 ms) instead of the router's ~1.4 s LLM round-trip.
    Implementations decide the model;
    :class:`~industryiq.core.retrieval.adapters.reranking.NoOpReranker` is the identity
    default (the pipeline collapses back to plain Stage-1 retrieval).
    """

    def rerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]: ...


@runtime_checkable
class ContextExpander(Protocol):
    """Widen retrieved hits with adjacent context before generation.

    An answer often straddles a chunk boundary -- the matched chunk names the topic
    but the figure/definition sits in the previous or next chunk. An expander stitches
    each hit's neighbours (by ``source`` + ``chunk_index``) into its ``text`` so the
    generator sees the whole passage, while the hit's id/score stay the matched chunk's
    (retrieval precision is unchanged; only the text grows). Implementations decide the
    window; :class:`~industryiq.core.retrieval.adapters.expansion.NoOpExpander` is the
    identity default.
    """

    def expand(self, hits: list[Hit]) -> list[Hit]: ...


@runtime_checkable
class SessionDocumentStore(Protocol):
    """Ephemeral per-conversation document index (in memory, lost on restart).

    Lets a chat search the files uploaded into *that session* only, separate
    from the persistent shared knowledge base. Uploaded documents are the
    primary context: they lead the results, and the shared store only backfills
    (see :func:`industryiq.core.retrieval.service.order_session_first`). Same
    embedder as the shared store, so scores stay comparable within each
    backfilled slot.

    The index lives here (a retrieval construct), but its lifecycle methods
    (``add`` / ``documents`` / ``clear``) are driven by chat -- upload/list
    routes and conversation deletion -- while ``retrieve`` is used by
    :class:`RetrievalService`.
    """

    def add(self, conversation_id: str, filename: str, text: str) -> list[str]: ...

    def retrieve(self, conversation_id: str, query: str, k: int = 5) -> list[Hit]: ...

    def documents(self, conversation_id: str) -> list[str]: ...

    def clear(self, conversation_id: str) -> None: ...


@dataclass(frozen=True)
class RetrievalResult:
    """The grounding context gathered for one turn.

    * ``standalone_question`` -- the condensed, self-contained query actually used
      for retrieval (surfaced for debugging follow-ups).
    * ``hits`` -- the filtered, merged chunks to ground the answer on.
    * ``timings_ms`` -- per-step durations (``"rewrite"``, ``"route_strategy"``,
      ``"retrieve"``) so the caller can fold them into its own turn timings.
    * ``search_plan`` -- the strategy + metadata filter chosen for this turn
      (surfaced for debugging which retrieval path ran); defaults to hybrid-RRF.
    """

    standalone_question: str
    hits: list[Hit]
    timings_ms: dict[str, float]
    search_plan: SearchPlan = field(default_factory=SearchPlan)


@runtime_checkable
class CorpusRetriever(Protocol):
    """The tuned shared-corpus retrieval, minus conversation state.

    ``retrieve_corpus`` runs strategy-route -> retrieve -> relevance-filter ->
    rerank for a *standalone* question -- everything :meth:`ContextRetriever.gather`
    does on the shared corpus, without the follow-up rewrite or the per-conversation
    session documents. It is the reusable core shared by ``gather``'s shared leg and
    a stateless caller (an agent's retrieve tool), so both inherit the same tuning.
    """

    def retrieve_corpus(self, question: str, k: int) -> list[Hit]: ...


@runtime_checkable
class ContextRetriever(Protocol):
    """Gather the grounding context for one conversational turn.

    The coarse retrieval seam: a caller hands over the raw question plus recent
    history and gets back a :class:`RetrievalResult` (rewrite + all doc retrieval
    + relevance filtering + merge already applied). Implementations decide the
    strategy; :class:`RetrievalService` is the default.

    ``plan_override`` forces a specific :class:`SearchPlan` instead of routing --
    used to re-run a turn on a broadened plan (e.g. filter dropped) without paying
    for strategy routing again.
    """

    def gather(
        self,
        conversation_id: str,
        question: str,
        history: list[Turn],
        k: int,
        *,
        plan_override: SearchPlan | None = None,
    ) -> RetrievalResult: ...
