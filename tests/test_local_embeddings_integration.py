"""Integration test for LocalEmbedder (fastembed).

Run with the ``local`` extra installed (`pip install -e ".[dev,local]"`):

    pytest -m integration

Skipped from the default unit run. Downloads a small model on first use, then
asserts LocalEmbedder honors the same Embedder contract as FakeEmbedder.
"""

import pytest

from industryiq.core.embeddings import Embedder

pytestmark = pytest.mark.integration


@pytest.fixture
def embedder() -> Embedder:
    pytest.importorskip("fastembed")
    from industryiq.core.local_embeddings import LocalEmbedder

    return LocalEmbedder()


def test_satisfies_interface(embedder: Embedder) -> None:
    assert isinstance(embedder, Embedder)


def test_output_shape_matches_dim(embedder: Embedder) -> None:
    vectors = embedder.embed(["hello world", "a second piece of text"])
    assert len(vectors) == 2
    assert all(len(v) == embedder.dim for v in vectors)


def test_is_deterministic(embedder: Embedder) -> None:
    assert embedder.embed(["hello"]) == embedder.embed(["hello"])


def test_different_texts_produce_different_vectors(embedder: Embedder) -> None:
    [vec_a], [vec_b] = embedder.embed(["alpha"]), embedder.embed(["beta"])
    assert vec_a != vec_b


def test_empty_input_returns_empty(embedder: Embedder) -> None:
    assert embedder.embed([]) == []
