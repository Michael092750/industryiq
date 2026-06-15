"""HTTP routes for multi-round chat.

A thin layer over :class:`ChatService`: validate input, call the service,
serialize the result. No chat logic lives here -- that is all in
``ragproject.core.chat``.
"""

import json
from collections.abc import Iterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ragproject.api.deps import get_chat_service
from ragproject.core.chat import (
    ChatService,
    ConversationNotFound,
    StreamEnd,
    StreamStart,
    StreamStatus,
    StreamToken,
)

Service = Annotated[ChatService, Depends(get_chat_service)]

router = APIRouter(prefix="/conversations", tags=["chat"])


class CreateConversationRequest(BaseModel):
    title: str = Field(default="New conversation", min_length=1, max_length=255)


class ConversationResponse(BaseModel):
    id: str
    title: str


class MessageRequest(BaseModel):
    question: str = Field(min_length=1)


class Source(BaseModel):
    text: str
    score: float
    document: str | None = None  # originating document name, if known


class MessageResponse(BaseModel):
    answer: str
    standalone_question: str
    sources: list[Source]
    timings_ms: dict[str, float]


class TurnResponse(BaseModel):
    question: str
    answer: str


class HistoryResponse(BaseModel):
    conversation_id: str
    turns: list[TurnResponse]


@router.post("", response_model=ConversationResponse)
def create_conversation(
    request: CreateConversationRequest, service: Service
) -> ConversationResponse:
    conversation = service.start(request.title)
    return ConversationResponse(id=conversation.id, title=conversation.title)


@router.post("/{conversation_id}/messages", response_model=MessageResponse)
def post_message(
    conversation_id: str, request: MessageRequest, service: Service
) -> MessageResponse:
    try:
        result = service.reply(conversation_id, request.question)
    except ConversationNotFound:
        raise HTTPException(status_code=404, detail="Conversation not found") from None
    return MessageResponse(
        answer=result.answer,
        standalone_question=result.standalone_question,
        sources=[
            Source(
                text=hit.metadata.get("text", ""),
                score=hit.score,
                document=hit.metadata.get("source"),
            )
            for hit in result.hits
        ],
        timings_ms=result.timings_ms,
    )


def _sse(event: str, data: dict[str, object]) -> str:
    """Format one Server-Sent Event frame."""
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.post("/{conversation_id}/messages/stream")
def stream_message(
    conversation_id: str, request: MessageRequest, service: Service
) -> StreamingResponse:
    # Validate existence up front so a missing conversation is a real 404,
    # before the streaming response (and its 200 status) has begun.
    if service.get_conversation(conversation_id) is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    def event_stream() -> Iterator[str]:
        for event in service.reply_stream(conversation_id, request.question):
            if isinstance(event, StreamStatus):
                yield _sse("status", {"phase": event.phase})
            elif isinstance(event, StreamStart):
                sources = [
                    Source(
                        text=hit.metadata.get("text", ""),
                        score=hit.score,
                        document=hit.metadata.get("source"),
                    )
                    for hit in event.hits
                ]
                yield _sse(
                    "sources",
                    {
                        "standalone_question": event.standalone_question,
                        "sources": [source.model_dump() for source in sources],
                    },
                )
            elif isinstance(event, StreamToken):
                yield _sse("token", {"text": event.text})
            elif isinstance(event, StreamEnd):
                yield _sse("done", {"answer": event.answer, "timings_ms": event.timings_ms})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.get("/{conversation_id}/messages", response_model=HistoryResponse)
def get_messages(conversation_id: str, service: Service) -> HistoryResponse:
    if service.get_conversation(conversation_id) is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    turns = service.get_history(conversation_id)
    return HistoryResponse(
        conversation_id=conversation_id,
        turns=[TurnResponse(question=turn.question, answer=turn.answer) for turn in turns],
    )
