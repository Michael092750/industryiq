"""HTTP routes for multi-round chat.

A thin layer over :class:`ChatService`: validate input, call the service,
serialize the result. No chat logic lives here -- that is all in
``industryiq.core.chat``.
"""

import json
from collections.abc import Iterator
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from industryiq.api.deps import get_chat_service, get_current_user, get_session_documents
from industryiq.api.file_loading import load_upload
from industryiq.core.auth import User
from industryiq.core.chat import (
    ChatService,
    Conversation,
    ConversationNotFound,
    SessionDocumentStore,
    StreamEnd,
    StreamStart,
    StreamStatus,
    StreamToken,
)

Service = Annotated[ChatService, Depends(get_chat_service)]
SessionDocs = Annotated[SessionDocumentStore, Depends(get_session_documents)]
CurrentUser = Annotated[User, Depends(get_current_user)]

router = APIRouter(prefix="/conversations", tags=["chat"])


def _require_owned(service: ChatService, conversation_id: str, user: User) -> Conversation:
    """Return the conversation only if it exists and belongs to ``user``.

    A thread owned by someone else is reported as missing (404, not 403) so the
    API never reveals that another user's conversation id exists.
    """
    conversation = service.get_conversation(conversation_id)
    if conversation is None or conversation.owner_id != user.id:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conversation


class CreateConversationRequest(BaseModel):
    title: str = Field(default="New conversation", min_length=1, max_length=255)


class RenameConversationRequest(BaseModel):
    title: str = Field(min_length=1, max_length=255)


class ConversationResponse(BaseModel):
    id: str
    title: str


class ConversationsResponse(BaseModel):
    conversations: list[ConversationResponse]


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


class UploadDocumentResponse(BaseModel):
    filename: str
    chunks: int


class SessionDocumentsResponse(BaseModel):
    conversation_id: str
    documents: list[str]


@router.post("", response_model=ConversationResponse)
def create_conversation(
    request: CreateConversationRequest, service: Service, user: CurrentUser
) -> ConversationResponse:
    conversation = service.start(request.title, owner_id=user.id)
    return ConversationResponse(id=conversation.id, title=conversation.title)


@router.get("", response_model=ConversationsResponse)
def list_conversations(service: Service, user: CurrentUser) -> ConversationsResponse:
    conversations = service.list_conversations(owner_id=user.id)
    return ConversationsResponse(
        conversations=[ConversationResponse(id=c.id, title=c.title) for c in conversations]
    )


@router.patch("/{conversation_id}", response_model=ConversationResponse)
def rename_conversation(
    conversation_id: str, request: RenameConversationRequest, service: Service, user: CurrentUser
) -> ConversationResponse:
    _require_owned(service, conversation_id, user)
    service.rename_conversation(conversation_id, request.title)
    return ConversationResponse(id=conversation_id, title=request.title)


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_conversation(conversation_id: str, service: Service, user: CurrentUser) -> None:
    _require_owned(service, conversation_id, user)
    service.delete_conversation(conversation_id)


@router.post("/{conversation_id}/messages", response_model=MessageResponse)
def post_message(
    conversation_id: str, request: MessageRequest, service: Service, user: CurrentUser
) -> MessageResponse:
    _require_owned(service, conversation_id, user)
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
    conversation_id: str, request: MessageRequest, service: Service, user: CurrentUser
) -> StreamingResponse:
    # Validate ownership up front so a missing/foreign conversation is a real
    # 404, before the streaming response (and its 200 status) has begun.
    _require_owned(service, conversation_id, user)

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
def get_messages(conversation_id: str, service: Service, user: CurrentUser) -> HistoryResponse:
    _require_owned(service, conversation_id, user)
    turns = service.get_history(conversation_id)
    return HistoryResponse(
        conversation_id=conversation_id,
        turns=[TurnResponse(question=turn.question, answer=turn.answer) for turn in turns],
    )


@router.post("/{conversation_id}/documents", response_model=UploadDocumentResponse)
def upload_document(
    conversation_id: str,
    service: Service,
    session_documents: SessionDocs,
    user: CurrentUser,
    file: UploadFile,
) -> UploadDocumentResponse:
    _require_owned(service, conversation_id, user)
    text = load_upload(file)
    chunk_ids = session_documents.add(conversation_id, file.filename or "upload", text)
    return UploadDocumentResponse(filename=file.filename or "upload", chunks=len(chunk_ids))


@router.get("/{conversation_id}/documents", response_model=SessionDocumentsResponse)
def list_session_documents(
    conversation_id: str, service: Service, session_documents: SessionDocs, user: CurrentUser
) -> SessionDocumentsResponse:
    _require_owned(service, conversation_id, user)
    return SessionDocumentsResponse(
        conversation_id=conversation_id,
        documents=session_documents.documents(conversation_id),
    )
