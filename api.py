"""
FastAPI service — exposes the indexing and query pipelines as REST endpoints.

Run:
    uvicorn rag_pipeline.api:app --host 0.0.0.0 --port 8000 --workers 1

Endpoints:
    POST  /index/documents          Index a list of text documents
    POST  /index/url                Fetch a URL and index its content
    DELETE /index/source            Delete all chunks for a source
    GET   /index/sources            List all indexed sources
    POST  /query                    Query the knowledge base (non-streaming)
    POST  /query/stream             Stream the answer token by token (SSE)
    GET   /health                   Liveness check
    GET   /health/ready             Readiness check (pings Qdrant + Redis)
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from rag_pipeline.config.settings import settings
from rag_pipeline.core.models import Document
from rag_pipeline.indexing.pipeline import IndexingPipeline
from rag_pipeline.query.pipeline import QueryPipeline

log = logging.getLogger(__name__)
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
)

# Singletons shared across requests
_indexer: IndexingPipeline | None = None
_querier: QueryPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    global _indexer, _querier
    log.info("Starting RAG service")
    _indexer = IndexingPipeline()
    _querier = QueryPipeline()
    await _indexer.__aenter__()
    await _querier.__aenter__()
    log.info("Pipelines ready")
    yield
    log.info("Shutting down")
    await _indexer.__aexit__(None, None, None)
    await _querier.__aexit__(None, None, None)


app = FastAPI(
    title="RAG Pipeline API",
    version="1.0.0",
    lifespan=lifespan,
)


# ------------------------------------------------------------------
# Request / Response schemas
# ------------------------------------------------------------------

class IndexDocumentRequest(BaseModel):
    content: str = Field(..., min_length=1, description="Raw document text")
    source: str = Field(..., description="Unique identifier, e.g. file path or URL")
    doc_type: str = Field("text", description="text | pdf | html | markdown")
    title: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class IndexResponse(BaseModel):
    doc_id: str
    source: str
    chunks_created: int
    status: str
    duration_ms: float
    error: str | None = None


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int | None = None
    filters: dict[str, Any] | None = None
    use_query_rewrite: bool = False
    use_hyde: bool = False
    use_mmr: bool = False


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]
    chunks_used: int
    model: str
    prompt_tokens: int
    completion_tokens: int
    retrieval_ms: float
    generation_ms: float


class ChatRequest(BaseModel):
    question: str
    history: list[dict[str, str]] = Field(default_factory=list)
    filters: dict[str, Any] | None = None


# ------------------------------------------------------------------
# Middleware — request timing
# ------------------------------------------------------------------

@app.middleware("http")
async def add_timing_header(request: Request, call_next):
    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.1f}"
    return response


# ------------------------------------------------------------------
# Indexing endpoints
# ------------------------------------------------------------------

@app.post("/index/documents", response_model=list[IndexResponse], tags=["indexing"])
async def index_documents(requests: list[IndexDocumentRequest]):
    """Index one or more documents. Idempotent — same source can be re-indexed."""
    docs = [
        Document(
            content=r.content,
            source=r.source,
            doc_type=r.doc_type,
            title=r.title,
            metadata=r.metadata,
        )
        for r in requests
    ]
    results = await _indexer.index_documents(docs)
    return [
        IndexResponse(
            doc_id=res.doc_id,
            source=res.source,
            chunks_created=res.chunks_created,
            status=res.status.value,
            duration_ms=res.duration_ms,
            error=res.error,
        )
        for res in results
    ]


@app.delete("/index/source", tags=["indexing"])
async def delete_source(source: str, doc_id: str):
    """Delete all indexed chunks for a given source document."""
    await _indexer._store.delete_by_doc_id(doc_id)
    return {"deleted": True, "source": source, "doc_id": doc_id}


@app.get("/index/sources", tags=["indexing"])
async def list_sources():
    """Return all unique source identifiers in the index."""
    sources = await _indexer._store.scroll_sources()
    return {"sources": sources, "count": len(sources)}


@app.get("/index/info", tags=["indexing"])
async def index_info():
    """Return collection statistics."""
    return await _indexer._store.collection_info()


# ------------------------------------------------------------------
# Query endpoints
# ------------------------------------------------------------------

@app.post("/query", response_model=QueryResponse, tags=["query"])
async def query(req: QueryRequest):
    """Query the knowledge base. Returns a complete answer with source citations."""
    result = await _querier.query(
        req.question,
        top_k=req.top_k,
        filters=req.filters,
        use_query_rewrite=req.use_query_rewrite,
        use_hyde=req.use_hyde,
        use_mmr=req.use_mmr,
    )
    return QueryResponse(
        question=result.question,
        answer=result.answer,
        sources=result.sources,
        chunks_used=len(result.chunks),
        model=result.model,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        retrieval_ms=result.retrieval_ms,
        generation_ms=result.generation_ms,
    )


@app.post("/query/stream", tags=["query"])
async def query_stream(req: QueryRequest):
    """
    Stream the answer as Server-Sent Events.
    Each event is a raw text token. The stream ends when the LLM finishes.
    """
    async def token_generator() -> AsyncIterator[bytes]:
        async for token in _querier.stream(
            req.question,
            top_k=req.top_k,
            filters=req.filters,
            use_query_rewrite=req.use_query_rewrite,
        ):
            yield f"data: {token}\n\n".encode()
        yield b"data: [DONE]\n\n"

    return StreamingResponse(
        token_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",   # disable nginx buffering
        },
    )


@app.post("/query/chat", response_model=QueryResponse, tags=["query"])
async def chat(req: ChatRequest):
    """Multi-turn conversation endpoint with history condensation."""
    result = await _querier.chat(
        req.question,
        history=req.history,
        filters=req.filters,
    )
    return QueryResponse(
        question=result.question,
        answer=result.answer,
        sources=result.sources,
        chunks_used=len(result.chunks),
        model=result.model,
        prompt_tokens=result.prompt_tokens,
        completion_tokens=result.completion_tokens,
        retrieval_ms=result.retrieval_ms,
        generation_ms=result.generation_ms,
    )


# ------------------------------------------------------------------
# Health checks
# ------------------------------------------------------------------

@app.get("/health", tags=["ops"])
async def health():
    return {"status": "ok"}


@app.get("/health/ready", tags=["ops"])
async def ready():
    """Deep readiness check — verifies Qdrant and Redis connectivity."""
    checks: dict[str, str] = {}
    try:
        info = await _indexer._store.collection_info()
        checks["qdrant"] = f"ok ({info['vectors_count']} vectors)"
    except Exception as exc:
        checks["qdrant"] = f"error: {exc}"

    try:
        if _indexer._embed._redis:
            await _indexer._embed._redis.ping()
            checks["redis"] = "ok"
        else:
            checks["redis"] = "not connected"
    except Exception as exc:
        checks["redis"] = f"error: {exc}"

    all_ok = all("error" not in v for v in checks.values())
    status_code = 200 if all_ok else 503
    return {"status": "ready" if all_ok else "degraded", "checks": checks}
