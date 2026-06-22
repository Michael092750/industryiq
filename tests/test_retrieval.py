from industryiq.core.embeddings import FakeEmbedder
from industryiq.core.retrieval import Retriever
from industryiq.core.vectorstore import InMemoryVectorStore


def _retriever() -> Retriever:
    return Retriever(FakeEmbedder(dim=16), InMemoryVectorStore())


def test_retrieve_returns_most_relevant_first() -> None:
    retriever = _retriever()
    retriever.index(["cats are great", "dogs are loyal", "fish can swim"])
    # FakeEmbedder is exact-match: querying a stored chunk verbatim ranks it top.
    hits = retriever.retrieve("dogs are loyal", k=3)
    assert hits[0].metadata["text"] == "dogs are loyal"


def test_retrieve_respects_k() -> None:
    retriever = _retriever()
    retriever.index(["a", "b", "c", "d", "e"])
    assert len(retriever.retrieve("a", k=2)) == 2


def test_retrieve_on_empty_index_returns_empty() -> None:
    assert _retriever().retrieve("anything", k=5) == []


def test_index_stores_text_in_metadata() -> None:
    retriever = _retriever()
    retriever.index(["hello world"])
    hit = retriever.retrieve("hello world", k=1)[0]
    assert hit.metadata["text"] == "hello world"


def test_index_merges_caller_metadata_with_text() -> None:
    retriever = _retriever()
    retriever.index(["doc one"], metadatas=[{"source": "file.txt"}])
    hit = retriever.retrieve("doc one", k=1)[0]
    assert hit.metadata == {"text": "doc one", "source": "file.txt"}


def test_index_returns_assigned_ids() -> None:
    retriever = _retriever()
    ids = retriever.index(["x", "y"], ids=["id-x", "id-y"])
    assert ids == ["id-x", "id-y"]


def test_index_empty_texts_is_noop() -> None:
    retriever = _retriever()
    assert retriever.index([]) == []
    assert retriever.retrieve("x", k=5) == []
