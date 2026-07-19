"""Retrieval: the retrieve job as a self-contained, ports-and-adapters package.

Public surface:

* :class:`Retriever` -- the low-level embed-and-search mechanism over one store.
* :class:`RetrievalService` -- composes rewrite + fan-out (session + shared
  corpus) + relevance filter + merge into a single :meth:`gather` call, returning
  a :class:`RetrievalResult`. This is the coarse :class:`ContextRetriever` seam a
  caller (e.g. ``ChatService``) depends on.
* Ports (:mod:`industryiq.core.retrieval.ports`) and adapters
  (:mod:`industryiq.core.retrieval.adapters`) -- the abstractions and their
  concrete implementations. The Redis session-doc store is imported only where it
  is wired (:mod:`industryiq.api.deps`), to keep this package import light.

The package depends only on ``embeddings``, ``vectorstore``, ``generation``, and
the neutral ``conversation`` / ``timing`` helpers -- never on ``chat`` -- so the
dependency runs one way (``chat -> retrieval``).
"""

from industryiq.core.retrieval.adapters.expansion import NeighborExpander, NoOpExpander
from industryiq.core.retrieval.adapters.filtering import ThresholdFilter
from industryiq.core.retrieval.adapters.rewriting import LlmQueryRewriter, NoOpQueryRewriter
from industryiq.core.retrieval.adapters.session_documents import SessionDocuments
from industryiq.core.retrieval.adapters.strategy import FixedStrategyRouter, LlmStrategyRouter
from industryiq.core.retrieval.ports import (
    ContextExpander,
    ContextRetriever,
    QueryRewriter,
    RelevanceFilter,
    RetrievalPort,
    RetrievalResult,
    SearchStrategyRouter,
    SessionDocumentStore,
)
from industryiq.core.retrieval.retriever import Retriever
from industryiq.core.retrieval.service import RetrievalService, order_session_first

__all__ = [
    "ContextExpander",
    "ContextRetriever",
    "FixedStrategyRouter",
    "LlmQueryRewriter",
    "LlmStrategyRouter",
    "NeighborExpander",
    "NoOpExpander",
    "NoOpQueryRewriter",
    "QueryRewriter",
    "RelevanceFilter",
    "RetrievalPort",
    "RetrievalResult",
    "RetrievalService",
    "Retriever",
    "SearchStrategyRouter",
    "SessionDocumentStore",
    "SessionDocuments",
    "ThresholdFilter",
    "order_session_first",
]
