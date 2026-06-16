"""A local, offline embedder backed by fastembed (no API key, no AWS).

The local-dev counterpart to :class:`ragproject.core.bedrock.BedrockEmbedder`.
Selected by ``RAG_PROVIDER=anthropic``, where Anthropic's API has no embeddings
endpoint and real Bedrock Titan credentials are inconvenient. Downloads a small
ONNX model on first use and runs entirely on the CPU, so retrieval is
*semantically meaningful* locally without any external service.

Satisfies the :class:`~ragproject.core.embeddings.Embedder` protocol, so it pairs
with any LLM behind the same composition root. ``fastembed`` is an optional
dependency (the ``local`` extra), imported lazily so production -- which uses
Bedrock and never installs it -- is unaffected.
"""

from ragproject.core.embeddings import Embedder

# bge-small-en-v1.5: 384-dim, a good speed/quality tradeoff for local dev. The
# dim is pinned to the model; changing the model means changing the dim.
_DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
_DEFAULT_DIM = 384


class LocalEmbedder(Embedder):
    """CPU embeddings from a small sentence-transformer via fastembed."""

    def __init__(self, model_name: str = _DEFAULT_MODEL, dim: int = _DEFAULT_DIM) -> None:
        # Lazy import: only the local extra needs fastembed installed.
        from fastembed import TextEmbedding

        self._dim = dim
        self._model = TextEmbedding(model_name=model_name)

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        # fastembed yields numpy arrays; convert to plain lists for the store.
        return [[float(x) for x in vector] for vector in self._model.embed(texts)]
