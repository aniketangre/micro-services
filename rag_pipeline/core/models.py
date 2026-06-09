"""
Core data models shared across the indexing and query pipelines.
All dataclasses are frozen where appropriate to prevent mutation bugs.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class ChunkStrategy(str, Enum):
    """How a document is split into chunks before embedding.

    RECURSIVE: split on paragraph → sentence → word boundaries (default, best quality).
    SEMANTIC:  split at topic boundaries detected by embedding similarity.
    FIXED:     split every N characters regardless of content (fastest, lowest quality).
    """

    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    FIXED = "fixed"


class IndexStatus(str, Enum):
    """Lifecycle state of an indexed document.

    PENDING:  queued but not yet written to the vector store.
    INDEXED:  all chunks are stored and searchable.
    FAILED:   indexing threw an exception; see IndexingResult.error.
    STALE:    source content changed but the index has not been updated yet.
    """

    PENDING = "pending"
    INDEXED = "indexed"
    FAILED = "failed"
    STALE = "stale"


@dataclass
class Document:
    """Raw input document before chunking."""

    content: str
    source: str                          # file path, URL, DB row id, etc.
    doc_type: str = "text"               # text | pdf | html | markdown
    title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def doc_id(self) -> str:
        """Stable deterministic ID based on source + content hash."""
        payload = f"{self.source}:{hashlib.sha256(self.content.encode()).hexdigest()[:16]}"
        return hashlib.md5(payload.encode()).hexdigest()


@dataclass
class Chunk:
    """
    A sub-document unit that gets embedded and stored in the vector DB.

    chunk_id is deterministic: changing the content changes the id,
    allowing upserts to be idempotent.
    """

    text: str
    doc_id: str
    source: str
    chunk_index: int                     # position within the parent document
    token_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    @property
    def chunk_id(self) -> str:
        """Deterministic ID derived from doc_id, position, and content hash.
        Re-indexing the same text at the same position produces the same ID,
        making upserts idempotent without a separate dedup lookup.
        """
        payload = f"{self.doc_id}:{self.chunk_index}:{hashlib.sha256(self.text.encode()).hexdigest()[:16]}"
        return hashlib.md5(payload.encode()).hexdigest()

    def to_payload(self) -> dict[str, Any]:
        """Serialise to the flat dict stored alongside the vector in Qdrant."""
        return {
            "text": self.text,
            "doc_id": self.doc_id,
            "source": self.source,
            "chunk_index": self.chunk_index,
            "token_count": self.token_count,
            "created_at": self.created_at.isoformat(),
            **self.metadata,
        }


@dataclass
class RetrievedChunk:
    """A chunk returned by the vector search, augmented with its similarity score."""

    chunk_id: str
    text: str
    score: float
    source: str
    doc_id: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.text[:80].replace("\n", " ")
        return f"RetrievedChunk(score={self.score:.4f}, source={self.source!r}, text={preview!r})"


@dataclass
class IndexingResult:
    """Outcome of indexing a single document."""

    doc_id: str
    source: str
    chunks_created: int
    status: IndexStatus
    error: str | None = None
    duration_ms: float = 0.0


@dataclass
class QueryResult:
    """Complete response from the query pipeline."""

    question: str
    answer: str
    chunks: list[RetrievedChunk]
    model: str
    prompt_tokens: int
    completion_tokens: int
    retrieval_ms: float
    generation_ms: float

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def sources(self) -> list[str]:
        """Deduplicated source list in retrieval order."""
        seen: set[str] = set()
        out: list[str] = []
        for c in self.chunks:
            if c.source not in seen:
                seen.add(c.source)
                out.append(c.source)
        return out
