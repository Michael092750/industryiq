"""Direct Anthropic API implementation of the LLM interface (local/dev).

The cloud-API counterpart to :class:`ragproject.core.bedrock.BedrockLLM`: the same
Claude models, but authenticated with an ``ANTHROPIC_API_KEY`` instead of an AWS
IAM role. Selected by ``RAG_PROVIDER=anthropic`` for local testing, where real
Bedrock credentials are inconvenient.

Satisfies the same :class:`~ragproject.core.generation.GenerativeLLM` protocol as
the fakes and the Bedrock provider, so it is a drop-in alternative chosen in
:mod:`ragproject.api.deps`.
"""

from collections.abc import Iterator

from anthropic import Anthropic

from ragproject.core.generation import GenerativeLLM


class AnthropicLLM(GenerativeLLM):
    """Text generation from Anthropic's Claude API (api-key auth)."""

    def __init__(
        self,
        model_id: str = "claude-sonnet-4-6",
        api_key: str | None = None,
        max_tokens: int = 1024,
    ) -> None:
        self._model_id = model_id
        self._max_tokens = max_tokens
        # api_key=None lets the SDK fall back to the ANTHROPIC_API_KEY env var.
        self._client = Anthropic(api_key=api_key)

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
