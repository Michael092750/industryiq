"""Amazon Bedrock implementations of the Embedder and LLM interfaces.

* :class:`BedrockEmbedder` -- Amazon Titan Text Embeddings v2 (via boto3).
* :class:`BedrockLLM` -- Anthropic Claude (via the Anthropic Bedrock SDK).

Both satisfy the same Protocols as the offline fakes, so they are drop-in
replacements selected by configuration in ``api.deps``.
"""

import json
from collections.abc import Iterator

import boto3
from anthropic import AnthropicBedrock

from industryiq.core.embeddings import Embedder
from industryiq.core.generation import GenerativeLLM


class BedrockEmbedder(Embedder):
    """Text embeddings from Amazon Titan Text Embeddings v2."""

    def __init__(
        self,
        model_id: str = "amazon.titan-embed-text-v2:0",
        region: str = "us-east-1",
        dim: int = 1024,
    ) -> None:
        self._model_id = model_id
        self._dim = dim
        self._client = boto3.client("bedrock-runtime", region_name=region)

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        # Titan embeds one text per call; loop over the batch.
        vectors: list[list[float]] = []
        for text in texts:
            response = self._client.invoke_model(
                modelId=self._model_id,
                body=json.dumps({"inputText": text, "dimensions": self._dim, "normalize": True}),
            )
            payload = json.loads(response["body"].read())
            vectors.append(payload["embedding"])
        return vectors


class BedrockLLM(GenerativeLLM):
    """Text generation from Anthropic Claude on Amazon Bedrock."""

    def __init__(
        self,
        model_id: str = "us.anthropic.claude-sonnet-4-6",
        region: str = "us-east-1",
        max_tokens: int = 1024,
    ) -> None:
        self._model_id = model_id
        self._max_tokens = max_tokens
        self._client = AnthropicBedrock(aws_region=region)

    def generate(self, prompt: str) -> str:
        message = self._client.messages.create(
            model=self._model_id,
            max_tokens=self._max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in message.content if block.type == "text")

    def stream(self, prompt: str) -> Iterator[str]:
        with self._client.messages.stream(
            model=self._model_id,
            max_tokens=self._max_tokens,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            yield from stream.text_stream
