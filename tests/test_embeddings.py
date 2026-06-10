import pytest

from ragproject.core.embeddings import Embedder, FakeEmbedder


def test_fake_embedder_satisfies_interface() -> None:
    # FakeEmbedder must be usable anywhere an Embedder is expected.
    assert isinstance(FakeEmbedder(), Embedder)


def test_output_shape_matches_dim() -> None:
    embedder = FakeEmbedder(dim=8)
    vectors = embedder.embed(["a", "b", "c"])
    assert len(vectors) == 3
    assert all(len(v) == 8 for v in vectors)


def test_is_deterministic() -> None:
    embedder = FakeEmbedder(dim=8)
    assert embedder.embed(["hello"]) == embedder.embed(["hello"])


def test_different_texts_produce_different_vectors() -> None:
    embedder = FakeEmbedder(dim=8)
    [vec_a], [vec_b] = embedder.embed(["alpha"]), embedder.embed(["beta"])
    assert vec_a != vec_b


def test_empty_input_returns_empty_list() -> None:
    assert FakeEmbedder().embed([]) == []


def test_dim_property() -> None:
    assert FakeEmbedder(dim=16).dim == 16


def test_invalid_dim_raises() -> None:
    with pytest.raises(ValueError):
        FakeEmbedder(dim=0)
