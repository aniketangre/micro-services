"""
Indexing pipeline — takes raw documents and produces searchable vectors.

Pipeline stages:
  1. Load     — accept str / file path / URL / list of Documents
  2. Chunk    — RecursiveCharacterTextSplitter with tiktoken length function
  3. Embed    — EmbeddingService (cached, batched, with retry)
  4. Upsert   — VectorStore (idempotent, content-hash IDs)

Concurrency model:
  - Documents are processed in parallel (asyncio.gather with semaphore)
  - Each document's chunks are embedded in one batched API call
  - Qdrant upsert is fire-and-wait (wait=True) per document

Usage:
    pipeline = IndexingPipeline()
    async with pipeline:
        results = await pipeline.index_documents(docs)
        await pipeline.index_file("./data/policy.pdf")
        await pipeline.index_directory("./data", glob="**/*.md")
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import AsyncIterator, Iterable

from langchain.text_splitter import RecursiveCharacterTextSplitter

from rag_pipeline.config.settings import settings
from rag_pipeline.core.embeddings import EmbeddingService
from rag_pipeline.core.models import (
    Chunk,
    Document,
    IndexingResult,
    IndexStatus,
)
from rag_pipeline.core.vector_store import VectorStore

log = logging.getLogger(__name__)

# Semaphore: max concurrent document embedding calls.
# Prevents memory spikes when indexing thousands of documents.
_MAX_CONCURRENT = 8


class IndexingPipeline:
    def __init__(self) -> None:
        self._embed = EmbeddingService()
        self._store = VectorStore()
        self._splitter = _build_splitter()
        self._sem = asyncio.Semaphore(_MAX_CONCURRENT)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "IndexingPipeline":
        await self._embed.__aenter__()
        await self._store.__aenter__()
        return self

    async def __aexit__(self, *args) -> None:
        await self._embed.__aexit__(*args)
        await self._store.__aexit__(*args)

    # ------------------------------------------------------------------
    # High-level entry points
    # ------------------------------------------------------------------

    async def index_documents(
        self, documents: Iterable[Document]
    ) -> list[IndexingResult]:
        """Index an iterable of Document objects concurrently."""
        docs = list(documents)
        if not docs:
            return []
        log.info("Indexing %d document(s)", len(docs))
        tasks = [self._index_one(doc) for doc in docs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        out: list[IndexingResult] = []
        for doc, res in zip(docs, results):
            if isinstance(res, Exception):
                log.error("Failed to index %r: %s", doc.source, res)
                out.append(
                    IndexingResult(
                        doc_id=doc.doc_id,
                        source=doc.source,
                        chunks_created=0,
                        status=IndexStatus.FAILED,
                        error=str(res),
                    )
                )
            else:
                out.append(res)
        return out

    async def index_file(
        self, path: str | Path, metadata: dict | None = None
    ) -> IndexingResult:
        """Load a single file and index it."""
        doc = await self._load_file(Path(path), metadata or {})
        return await self._index_one(doc)

    async def index_directory(
        self,
        directory: str | Path,
        glob: str = "**/*.txt",
        metadata: dict | None = None,
        max_files: int | None = None,
    ) -> list[IndexingResult]:
        """Recursively index all files matching *glob* in *directory*."""
        root = Path(directory)
        paths = list(root.glob(glob))
        if max_files:
            paths = paths[:max_files]
        log.info("Found %d files matching %r in %s", len(paths), glob, root)
        docs = [await self._load_file(p, metadata or {}) for p in paths]
        return await self.index_documents(docs)

    async def reindex_source(self, source: str, document: Document) -> IndexingResult:
        """
        Delete all existing chunks for *source* then re-index.
        Use this when a document's content changes.
        """
        await self._store.delete_by_doc_id(document.doc_id)
        return await self._index_one(document)

    # ------------------------------------------------------------------
    # Streaming indexing (for large corpora)
    # ------------------------------------------------------------------

    async def index_stream(
        self, stream: AsyncIterator[Document], batch_size: int = 50
    ) -> AsyncIterator[list[IndexingResult]]:
        """
        Process a stream of documents in batches.
        Yields result batches as they complete — memory-efficient for large corpora.

        Usage:
            async for batch in pipeline.index_stream(doc_stream):
                for result in batch:
                    print(result.status, result.source)
        """
        batch: list[Document] = []
        async for doc in stream:
            batch.append(doc)
            if len(batch) >= batch_size:
                yield await self.index_documents(batch)
                batch.clear()
        if batch:
            yield await self.index_documents(batch)

    # ------------------------------------------------------------------
    # Core indexing logic
    # ------------------------------------------------------------------

    async def _index_one(self, doc: Document) -> IndexingResult:
        async with self._sem:
            t0 = time.perf_counter()
            try:
                chunks = self._chunk(doc)
                if not chunks:
                    log.warning("Document %r produced 0 chunks — skipping", doc.source)
                    return IndexingResult(
                        doc_id=doc.doc_id,
                        source=doc.source,
                        chunks_created=0,
                        status=IndexStatus.INDEXED,
                    )

                texts = [c.text for c in chunks]
                vectors = await self._embed.embed(texts)

                n = await self._store.upsert_chunks(chunks, vectors)
                duration_ms = (time.perf_counter() - t0) * 1000
                log.info(
                    "Indexed %r → %d chunks in %.0f ms",
                    doc.source,
                    n,
                    duration_ms,
                )
                return IndexingResult(
                    doc_id=doc.doc_id,
                    source=doc.source,
                    chunks_created=n,
                    status=IndexStatus.INDEXED,
                    duration_ms=duration_ms,
                )

            except Exception as exc:
                log.exception("Error indexing %r", doc.source)
                return IndexingResult(
                    doc_id=doc.doc_id,
                    source=doc.source,
                    chunks_created=0,
                    status=IndexStatus.FAILED,
                    error=str(exc),
                    duration_ms=(time.perf_counter() - t0) * 1000,
                )

    def _chunk(self, doc: Document) -> list[Chunk]:
        """Split a Document into Chunks using recursive character splitting."""
        raw_chunks: list[str] = self._splitter.split_text(doc.content)
        chunks: list[Chunk] = []
        for i, text in enumerate(raw_chunks):
            text = text.strip()
            if not text:
                continue
            chunk = Chunk(
                text=text,
                doc_id=doc.doc_id,
                source=doc.source,
                chunk_index=i,
                token_count=_approx_tokens(text),
                metadata={
                    "doc_type": doc.doc_type,
                    "title": doc.title,
                    **doc.metadata,
                },
            )
            chunks.append(chunk)
        return chunks

    # ------------------------------------------------------------------
    # File loaders
    # ------------------------------------------------------------------

    @staticmethod
    async def _load_file(path: Path, metadata: dict) -> Document:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            content = await _load_pdf(path)
            doc_type = "pdf"
        elif suffix in {".html", ".htm"}:
            content = await _load_html(path)
            doc_type = "html"
        elif suffix == ".md":
            content = path.read_text(encoding="utf-8", errors="replace")
            doc_type = "markdown"
        else:
            content = path.read_text(encoding="utf-8", errors="replace")
            doc_type = "text"

        return Document(
            content=content,
            source=str(path),
            doc_type=doc_type,
            title=path.stem,
            metadata=metadata,
        )


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _build_splitter() -> RecursiveCharacterTextSplitter:
    """
    RecursiveCharacterTextSplitter with tiktoken token counting.
    Falls back to character counting if tiktoken is unavailable.
    """
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model("gpt-4o")

        def _len_fn(text: str) -> int:
            return len(enc.encode(text, disallowed_special=()))

    except ImportError:
        log.warning("tiktoken not installed; falling back to character-based length")
        _len_fn = len  # type: ignore[assignment]

    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=_len_fn,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator=False,
    )


def _approx_tokens(text: str) -> int:
    """Fast approximation: 1 token ≈ 4 characters for English text."""
    return max(1, len(text) // 4)


async def _load_pdf(path: Path) -> str:
    """Extract text from PDF using pdfminer (pure-Python, no binary deps)."""
    try:
        from pdfminer.high_level import extract_text
        return extract_text(str(path))
    except ImportError:
        log.warning("pdfminer not installed; reading PDF as raw bytes (poor quality)")
        return path.read_bytes().decode("utf-8", errors="replace")


async def _load_html(path: Path) -> str:
    """Strip HTML tags using html2text for clean plain-text extraction."""
    try:
        import html2text
        converter = html2text.HTML2Text()
        converter.ignore_links = False
        converter.ignore_images = True
        return converter.handle(path.read_text(encoding="utf-8"))
    except ImportError:
        import html
        import re
        raw = path.read_text(encoding="utf-8")
        clean = re.sub(r"<[^>]+>", " ", raw)
        return html.unescape(clean)
