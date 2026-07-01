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
    assert hit.metadata["text"] == "doc one"
    assert hit.metadata["source"] == "file.txt"
    # Derived metadata is added at index time.
    assert hit.metadata["chunk_index"] == 0
    assert "content_hash" in hit.metadata


def test_index_assigns_sequential_chunk_index() -> None:
    retriever = _retriever()
    retriever.index(["first", "second", "third"])
    indices = sorted(meta["chunk_index"] for _id, meta in retriever.all_chunks())
    assert indices == [0, 1, 2]


def test_index_returns_assigned_ids() -> None:
    retriever = _retriever()
    ids = retriever.index(["x", "y"], ids=["id-x", "id-y"])
    assert ids == ["id-x", "id-y"]


def test_index_empty_texts_is_noop() -> None:
    retriever = _retriever()
    assert retriever.index([]) == []
    assert retriever.retrieve("x", k=5) == []


def test_retrieve_keeps_short_chunks_by_default() -> None:
    # min_chunk_chars defaults to 0, so the length filter is off.
    retriever = _retriever()
    retriever.index(["hi", "x " * 150])
    assert retriever.retrieve("hi", k=1)[0].metadata["text"] == "hi"


def test_retrieve_drops_short_chunks_when_min_chunk_chars_set() -> None:
    long_chunk = "x " * 150  # 300 chars, well over the threshold
    retriever = Retriever(FakeEmbedder(dim=16), InMemoryVectorStore(), min_chunk_chars=200)
    retriever.index(["hi", long_chunk])
    # Exact-match would rank the short "hi" top, but it's below the threshold and dropped.
    hits = retriever.retrieve("hi", k=1)
    assert len(hits) == 1
    assert hits[0].metadata["text"] == long_chunk


def test_retrieve_falls_back_when_all_chunks_short() -> None:
    # If every candidate is below the threshold, return the raw hits rather than
    # nothing -- a short answer beats no answer.
    retriever = Retriever(FakeEmbedder(dim=16), InMemoryVectorStore(), min_chunk_chars=200)
    retriever.index(["hi", "yo"])
    assert len(retriever.retrieve("hi", k=1)) == 1
