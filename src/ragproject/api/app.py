"""FastAPI application: thin HTTP layer over :class:`RagPipeline`.

Routes validate input, call the pipeline, and serialize the result. They contain
no RAG logic of their own -- that all lives in ``ragproject.core``.
"""

import os
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, UploadFile
from pydantic import BaseModel, Field

from ragproject.api.deps import get_pipeline
from ragproject.core.loaders import SUPPORTED_EXTENSIONS, load
from ragproject.core.pipeline import RagPipeline

Pipeline = Annotated[RagPipeline, Depends(get_pipeline)]


class IngestRequest(BaseModel):
    text: str = Field(min_length=1)
    source: str | None = None


class IngestResponse(BaseModel):
    chunk_ids: list[str]


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    k: int = Field(default=5, ge=1)


class Source(BaseModel):
    text: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]


class Chunk(BaseModel):
    id: str
    text: str
    source: str | None = None


class ChunksResponse(BaseModel):
    count: int
    chunks: list[Chunk]


app = FastAPI(title="ragproject")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest")
def ingest(request: IngestRequest, pipeline: Pipeline) -> IngestResponse:
    chunk_ids = pipeline.ingest_text(request.text, source=request.source)
    return IngestResponse(chunk_ids=chunk_ids)


@app.post("/ingest/file")
def ingest_file(file: UploadFile, pipeline: Pipeline) -> IngestResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type {suffix!r}; supported: {sorted(SUPPORTED_EXTENSIONS)}",
        )
    # Write the upload to a temp file so the path-based loaders can read it.
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name
    try:
        text = load(tmp_path)
    finally:
        os.unlink(tmp_path)
    chunk_ids = pipeline.ingest_text(text, source=file.filename)
    return IngestResponse(chunk_ids=chunk_ids)


@app.get("/chunks")
def list_chunks(pipeline: Pipeline, limit: int = 100) -> ChunksResponse:
    items = pipeline.list_chunks(limit=limit)
    chunks = [
        Chunk(id=chunk_id, text=metadata.get("text", ""), source=metadata.get("source"))
        for chunk_id, metadata in items
    ]
    return ChunksResponse(count=len(chunks), chunks=chunks)


@app.post("/query")
def query(request: QueryRequest, pipeline: Pipeline) -> QueryResponse:
    result = pipeline.query(request.question, k=request.k)
    sources = [Source(text=hit.metadata.get("text", ""), score=hit.score) for hit in result.hits]
    return QueryResponse(answer=result.answer, sources=sources)
