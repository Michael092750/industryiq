"""Embeddings: turn text into vectors.

This module defines the :class:`Embedder` *interface* (a ``Protocol``) plus a
:class:`FakeEmbedder` for tests. The real provider-backed implementation
(e.g. Amazon Bedrock) is added later and only needs to satisfy the same
interface, so retrieval/pipeline code never depends on a specific provider.
"""

import hashlib
from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    """Anything that can turn texts into fixed-length vectors."""

    @property
    def dim(self) -> int:
        """The length of every vector this embedder produces."""
        ...

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, each of length :attr:`dim`."""
        ...


class FakeEmbedder:
    """Deterministic, offline embedder for tests.

    Vectors are derived from a hash of the text, so the same text always maps
    to the same vector (and different texts almost always differ). It carries
    no semantic meaning -- it exists to test plumbing, not retrieval quality.
    """

    def __init__(self, dim: int = 8) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return [digest[i % len(digest)] / 255.0 for i in range(self._dim)]
