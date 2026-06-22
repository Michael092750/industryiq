"""Domain types for multi-round chat.

Small, immutable value objects shared across the chat module. They carry no
behavior and no I/O, so every other chat component -- ports, service, stores,
prompt builders -- can depend on them without coupling to a provider.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import TypeAlias

from industryiq.core.vectorstore import Hit


@dataclass(frozen=True)
class Conversation:
    """A single chat thread: an id, a human label, when it began, and its owner.

    ``owner_id`` is the id of the user who created it; ``None`` means unowned
    (the pre-auth default, still used by stores called without a user). Routes
    pass the authenticated user so a thread is only ever served to its owner.
    """

    id: str
    title: str
    created_at: datetime
    owner_id: str | None = None


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
class ChatPolicy:
    """Tuning knobs for a chat turn, grouped so they don't crowd the constructor.

    * ``k`` -- how many chunks to retrieve.
    * ``history_limit`` -- how many recent turns to feed into the prompt.
    """

    k: int = 5
    history_limit: int = 6


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
    """A phase marker for the UI: 'thinking' | 'retrieving' | 'generating'.

    Carries only the semantic phase. The user-facing wording (and its
    translation) is the frontend's concern, not the domain's -- so no display
    text lives here.
    """

    phase: str


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
