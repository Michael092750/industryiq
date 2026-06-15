"""Domain types for multi-round chat.

Small, immutable value objects shared across the chat module. They carry no
behavior and no I/O, so every other chat component -- ports, service, stores,
prompt builders -- can depend on them without coupling to a provider.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import TypeAlias

from ragproject.core.vectorstore import Hit


@dataclass(frozen=True)
class Conversation:
    """A single chat thread: an id, a human label, and when it began."""

    id: str
    title: str
    created_at: datetime


@dataclass(frozen=True)
class Turn:
    """One completed exchange in a conversation: a question and its answer."""

    question: str
    answer: str


@dataclass(frozen=True)
class RouteDecision:
    """Whether a turn should consult the knowledge base.

    A named type (rather than a bare ``bool``) so the router port can grow
    richer verdicts later without changing its signature.
    """

    should_retrieve: bool


@dataclass(frozen=True)
class ChatResult:
    """The outcome of one :meth:`ChatService.reply` call.

    Bundles the answer with the chunks it was grounded in, the standalone
    question actually used for retrieval (handy for debugging follow-ups), and
    the per-step timings of the turn (milliseconds, keyed by step name).
    """

    answer: str
    hits: list[Hit]
    standalone_question: str
    timings_ms: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class StreamStatus:
    """A phase indicator for the UI: 'thinking' | 'retrieving' | 'generating'.

    ``label`` is human-facing chrome (e.g. "Checking local knowledge base...");
    it is canned UI text, never model output.
    """

    phase: str
    label: str = ""


@dataclass(frozen=True)
class StreamStart:
    """Emitted once after routing: the retrieval result, before answer tokens."""

    standalone_question: str
    hits: list[Hit]


@dataclass(frozen=True)
class StreamToken:
    """A chunk of the streamed answer."""

    text: str


@dataclass(frozen=True)
class StreamEnd:
    """Final streamed event: the complete answer and per-step timings."""

    answer: str
    timings_ms: dict[str, float]


# One conversational turn, streamed, is a sequence of these.
StreamEvent: TypeAlias = StreamStatus | StreamStart | StreamToken | StreamEnd
