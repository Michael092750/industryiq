from industryiq.core.chat.adapters.session_documents import SessionDocuments
from industryiq.core.embeddings import FakeEmbedder


def _docs() -> SessionDocuments:
    return SessionDocuments(FakeEmbedder(dim=16))


def test_add_then_retrieve_finds_the_document() -> None:
    docs = _docs()
    docs.add("c1", "facts.txt", "the sky is blue")
    hits = docs.retrieve("c1", "the sky is blue", k=3)
    assert hits[0].metadata["text"] == "the sky is blue"
    assert hits[0].metadata["source"] == "facts.txt"


def test_retrieve_unknown_conversation_is_empty() -> None:
    assert _docs().retrieve("nope", "anything", k=3) == []


def test_documents_lists_uploaded_filenames() -> None:
    docs = _docs()
    docs.add("c1", "a.txt", "alpha")
    docs.add("c1", "b.txt", "beta")
    assert set(docs.documents("c1")) == {"a.txt", "b.txt"}


def test_sessions_are_isolated() -> None:
    docs = _docs()
    docs.add("c1", "a.txt", "alpha beta gamma")
    assert docs.retrieve("c2", "alpha beta gamma", k=3) == []
    assert docs.documents("c2") == []
