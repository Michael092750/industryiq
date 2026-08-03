"""Unit tests for RetrievalService -- the retrieve job in isolation.

Everything the chat turn used to inline -- rewrite, fan-out to session + shared
sources, per-source relevance filter, and merge -- now lives here and is tested
directly against ``gather``, with in-memory doubles and zero network.
"""

from industryiq.core.conversation import Turn
from industryiq.core.embeddings import FakeEmbedder
from industryiq.core.retrieval import Retriever
from industryiq.core.retrieval.adapters.filtering import ThresholdFilter
from industryiq.core.retrieval.service import RetrievalService, order_session_first
from industryiq.core.vectorstore import (
    Hit,
    InMemoryVectorStore,
    MetadataFilter,
    SearchPlan,
    SearchStrategy,
)


class RecordingRewriter:
    """A QueryRewriter double that records calls and returns a canned query."""

    def __init__(self, rewritten: str = "STANDALONE") -> None:
        self._rewritten = rewritten
        self.calls: list[tuple[list[Turn], str]] = []

    def condense(self, history: list[Turn], question: str) -> str:
        self.calls.append((history, question))
        return self._rewritten


class RecordingRetriever:
    """A RetrievalPort double that records the queries + plans it is asked to retrieve."""

    def __init__(self, hits: list[Hit] | None = None) -> None:
        self._hits = hits or []
        self.queries: list[str] = []
        self.plans: list[SearchPlan] = []

    def retrieve(self, query: str, k: int = 5, plan: SearchPlan | None = None) -> list[Hit]:
        self.queries.append(query)
        self.plans.append(plan if plan is not None else SearchPlan())
        return self._hits


class RecordingStrategyRouter:
    """A SearchStrategyRouter double: records questions, returns a canned plan."""

    def __init__(self, plan: SearchPlan | None = None) -> None:
        self._plan = plan if plan is not None else SearchPlan()
        self.questions: list[str] = []

    def select(self, question: str) -> SearchPlan:
        self.questions.append(question)
        return self._plan


class RecordingExpander:
    """A ContextExpander double: records the hits it is handed, returns them unchanged."""

    def __init__(self) -> None:
        self.calls: list[list[Hit]] = []

    def expand(self, hits: list[Hit]) -> list[Hit]:
        self.calls.append(hits)
        return hits


class RecordingReranker:
    """A Reranker double: records calls, reverses the hits (so a reorder is visible)."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[Hit], int]] = []

    def rerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]:
        self.calls.append((query, hits, k))
        return list(reversed(hits))[:k]


class StubSessionDocuments:
    """A SessionDocumentStore double returning fixed per-session hits."""

    def __init__(self, hits: list[Hit] | None = None) -> None:
        self._hits = hits or []

    def add(self, conversation_id: str, filename: str, text: str) -> list[str]:
        return []

    def retrieve(self, conversation_id: str, query: str, k: int = 5) -> list[Hit]:
        return self._hits

    def documents(self, conversation_id: str) -> list[str]:
        return []

    def clear(self, conversation_id: str) -> None:
        return None


def _service(
    retriever: RecordingRetriever | None = None,
    rewriter: RecordingRewriter | None = None,
    relevance_filter: ThresholdFilter | None = None,
    session_documents: StubSessionDocuments | None = None,
    strategy_router: RecordingStrategyRouter | None = None,
    expander: RecordingExpander | None = None,
    reranker: RecordingReranker | None = None,
    clock=None,
) -> RetrievalService:
    kwargs = {} if clock is None else {"clock": clock}
    return RetrievalService(
        retriever=retriever or RecordingRetriever(),
        rewriter=rewriter or RecordingRewriter(),
        relevance_filter=relevance_filter or ThresholdFilter(),
        strategy_router=strategy_router,
        reranker=reranker,
        expander=expander,
        session_documents=session_documents,
        **kwargs,
    )


def test_gather_retrieves_with_the_rewritten_query() -> None:
    retriever = RecordingRetriever()
    rewriter = RecordingRewriter(rewritten="standalone query")
    result = _service(retriever=retriever, rewriter=rewriter).gather("c", "follow up?", [], k=5)
    assert retriever.queries == ["standalone query"]
    assert result.standalone_question == "standalone query"


def test_gather_passes_history_to_the_rewriter() -> None:
    rewriter = RecordingRewriter()
    history = [Turn("q1", "a1")]
    _service(rewriter=rewriter).gather("c", "q2", history, k=5)
    seen_history, question = rewriter.calls[0]
    assert seen_history == history
    assert question == "q2"


def test_gather_grounds_hits_from_the_shared_corpus() -> None:
    retriever = Retriever(FakeEmbedder(dim=16), InMemoryVectorStore())
    retriever.index(["the sky is blue"])
    result = _service(
        retriever=retriever, rewriter=RecordingRewriter(rewritten="the sky is blue")
    ).gather("c", "the sky is blue", [], k=5)
    assert result.hits
    assert result.hits[0].metadata["text"] == "the sky is blue"


def test_gather_relevance_filter_drops_hits_below_threshold() -> None:
    retriever = RecordingRetriever(hits=[Hit("c1", 0.2, {"text": "x"})])
    result = _service(retriever=retriever, relevance_filter=ThresholdFilter(threshold=0.5)).gather(
        "c", "q", [], k=5
    )
    assert retriever.queries == ["STANDALONE"]  # it DID search
    assert result.hits == []  # but the low-score hit was filtered out


def test_gather_relevance_filter_keeps_hits_at_or_above_threshold() -> None:
    retriever = RecordingRetriever(hits=[Hit("c1", 0.9, {"text": "x"})])
    result = _service(retriever=retriever, relevance_filter=ThresholdFilter(threshold=0.5)).gather(
        "c", "q", [], k=5
    )
    assert result.hits[0].id == "c1"


def test_gather_puts_session_uploads_first() -> None:
    shared = RecordingRetriever(hits=[Hit("g", 0.9, {"text": "global"})])
    session = StubSessionDocuments(hits=[Hit("s", 0.2, {"text": "session"})])
    result = _service(retriever=shared, session_documents=session).gather("c", "q", [], k=5)
    # Upload leads despite a higher-scoring corpus hit; the corpus still backfills.
    assert [hit.id for hit in result.hits] == ["s", "g"]
    assert shared.queries == ["STANDALONE"]  # corpus consulted to fill the leftover slots


def test_gather_skips_shared_when_session_fills_the_budget() -> None:
    shared = RecordingRetriever(hits=[Hit("g", 0.9, {"text": "global"})])
    session = StubSessionDocuments(hits=[Hit("s", 0.5, {"text": "session"})])
    result = _service(retriever=shared, session_documents=session).gather("c", "q", [], k=1)
    assert [hit.id for hit in result.hits] == ["s"]  # only the upload
    assert shared.queries == []  # corpus never consulted -- the upload already filled k


def test_gather_without_session_documents_uses_shared_only() -> None:
    shared = RecordingRetriever(hits=[Hit("g", 0.9, {"text": "global"})])
    result = _service(retriever=shared).gather("c", "q", [], k=5)
    assert [hit.id for hit in result.hits] == ["g"]


def test_gather_reports_rewrite_route_and_retrieve_timings(fake_clock) -> None:
    result = _service(clock=fake_clock(step=0.001)).gather("c", "q", [], k=5)
    assert {"rewrite", "route_strategy", "retrieve"} <= set(result.timings_ms)
    assert all(value >= 0 for value in result.timings_ms.values())


def test_gather_without_router_retrieves_with_the_default_plan() -> None:
    retriever = RecordingRetriever()
    _service(retriever=retriever).gather("c", "q", [], k=5)
    # No strategy router configured -> today's behaviour (hybrid-RRF, no filter).
    assert retriever.plans[0].is_default()


def test_gather_routes_strategy_on_the_standalone_query() -> None:
    plan = SearchPlan(strategy=SearchStrategy.LEXICAL, filter=MetadataFilter(publisher="McKinsey"))
    router = RecordingStrategyRouter(plan=plan)
    retriever = RecordingRetriever()
    result = _service(
        retriever=retriever,
        rewriter=RecordingRewriter(rewritten="standalone"),
        strategy_router=router,
    ).gather("c", "follow up?", [], k=5)
    # The router classifies the *condensed* query, and its plan reaches the retriever.
    assert router.questions == ["standalone"]
    assert retriever.plans[0] is plan
    # ...and is surfaced on the result for debugging which path ran.
    assert result.search_plan is plan


def test_gather_surfaces_default_plan_when_no_router() -> None:
    result = _service().gather("c", "q", [], k=5)
    assert result.search_plan.is_default()


def test_gather_plan_override_bypasses_the_strategy_router() -> None:
    router = RecordingStrategyRouter(plan=SearchPlan(strategy=SearchStrategy.LEXICAL))
    retriever = RecordingRetriever()
    override = SearchPlan(strategy=SearchStrategy.SEMANTIC)
    result = _service(retriever=retriever, strategy_router=router).gather(
        "c", "q", [], k=5, plan_override=override
    )
    assert router.questions == []  # router not consulted on an override
    assert retriever.plans[0] is override  # the override drove retrieval verbatim
    assert result.search_plan is override


def test_gather_expands_context_when_an_expander_is_configured() -> None:
    retriever = RecordingRetriever(hits=[Hit("g", 0.9, {"text": "global"})])
    expander = RecordingExpander()
    result = _service(retriever=retriever, expander=expander).gather("c", "q", [], k=5)
    assert expander.calls == [result.hits]  # the merged hits were handed to the expander
    assert "expand" in result.timings_ms


def test_gather_skips_expansion_when_no_expander() -> None:
    result = _service().gather("c", "q", [], k=5)
    assert "expand" not in result.timings_ms


def test_order_session_first_leads_with_uploads_over_higher_scoring_shared() -> None:
    session = [Hit("s", 0.2, {})]
    shared = [Hit("g", 0.9, {})]
    # The uploaded session hit leads even though the shared hit scores higher.
    assert [hit.id for hit in order_session_first(session, shared, k=5)] == ["s", "g"]


def test_order_session_first_backfills_remaining_slots_from_shared() -> None:
    session = [Hit("s", 0.4, {})]
    shared = [Hit("g1", 0.9, {}), Hit("g2", 0.1, {})]
    # k=2: one session slot, one shared slot -- the higher-scoring shared hit fills it.
    assert [hit.id for hit in order_session_first(session, shared, k=2)] == ["s", "g1"]


def test_order_session_first_dedups_by_id_keeping_the_session_copy() -> None:
    ordered = order_session_first([Hit("x", 0.4, {})], [Hit("x", 0.8, {})], k=5)
    assert [hit.id for hit in ordered] == ["x"]
    assert ordered[0].score == 0.4  # the session copy leads; the shared duplicate is dropped


# --- retrieve_corpus: the shared tuned core the agent retrieve tool reuses --------


def test_retrieve_corpus_uses_the_tuned_core_without_rewrite_or_session() -> None:
    plan = SearchPlan(strategy=SearchStrategy.LEXICAL)
    router = RecordingStrategyRouter(plan=plan)
    retriever = RecordingRetriever(hits=[Hit("g", 0.9, {"text": "corpus"})])
    rewriter = RecordingRewriter(rewritten="SHOULD_NOT_BE_USED")
    session = StubSessionDocuments(hits=[Hit("s", 0.9, {"text": "upload"})])
    service = _service(
        retriever=retriever,
        rewriter=rewriter,
        strategy_router=router,
        session_documents=session,
    )
    hits = service.retrieve_corpus("plain question", k=5)
    assert [hit.id for hit in hits] == ["g"]  # shared corpus only -- no session upload
    assert router.questions == ["plain question"]  # routed on the raw question
    assert retriever.plans[0] is plan  # the tuned strategy plan reached the retriever
    assert rewriter.calls == []  # no follow-up rewrite


def test_retrieve_corpus_applies_the_reranker() -> None:
    reranker = RecordingReranker()
    retriever = RecordingRetriever(
        hits=[Hit("a", 0.9, {"text": "a"}), Hit("b", 0.8, {"text": "b"})]
    )
    hits = _service(retriever=retriever, reranker=reranker).retrieve_corpus("q", k=2)
    assert reranker.calls and reranker.calls[0][0] == "q"
    assert [hit.id for hit in hits] == ["b", "a"]  # reranker reordered -> planner inherits it


def test_gather_shared_leg_matches_retrieve_corpus() -> None:
    # gather rewrites "follow up?" -> "standalone"; retrieve_corpus of that standalone
    # returns the identical shared hits -- one shared implementation, no drift.
    service = _service(
        retriever=RecordingRetriever(hits=[Hit("g", 0.9, {"text": "x"})]),
        rewriter=RecordingRewriter(rewritten="standalone"),
    )
    gathered = service.gather("c", "follow up?", [], k=5).hits
    corpus = service.retrieve_corpus("standalone", k=5)
    assert [hit.id for hit in gathered] == [hit.id for hit in corpus] == ["g"]
