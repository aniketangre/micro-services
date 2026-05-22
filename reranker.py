"""
Reranker — second-stage cross-encoder that re-scores retrieved chunks.

Why: bi-encoder (dense retrieval) optimises for fast ANN search but its
scores are not calibrated for fine-grained relevance. A cross-encoder
sees (query, chunk) together, producing much more accurate relevance
scores at the cost of O(k) forward passes.

Typical improvement: 15–30% NDCG@3 over dense-only retrieval.
Latency overhead: ~40–80ms for k=10 on a CPU.

Models (in order of quality vs speed):
  cross-encoder/ms-marco-MiniLM-L-6-v2   best balance (default)
  cross-encoder/ms-marco-MiniLM-L-2-v2   fastest
  cross-encoder/ms-marco-electra-base     highest quality

The model is lazy-loaded on first use and cached for the process lifetime.
Set USE_RERANKER=false to skip this stage entirely.
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from typing import Sequence

from rag_pipeline.core.models import RetrievedChunk

log = logging.getLogger(__name__)

_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="reranker")


@lru_cache(maxsize=1)
def _load_cross_encoder():
    """Lazy-load and cache the cross-encoder. Thread-safe via lru_cache."""
    try:
        from sentence_transformers import CrossEncoder
        log.info("Loading cross-encoder %r", _RERANKER_MODEL)
        model = CrossEncoder(_RERANKER_MODEL, max_length=512)
        log.info("Cross-encoder loaded")
        return model
    except ImportError as e:
        raise ImportError(
            "sentence-transformers is required for reranking. "
            "Install with: pip install sentence-transformers"
        ) from e


def _rerank_sync(
    query: str, chunks: list[RetrievedChunk], top_n: int
) -> list[RetrievedChunk]:
    """CPU-bound reranking — runs in a thread pool."""
    model = _load_cross_encoder()
    pairs = [(query, c.text) for c in chunks]

    t0 = time.perf_counter()
    scores = model.predict(pairs, show_progress_bar=False)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    reranked = sorted(
        zip(chunks, scores),
        key=lambda x: x[1],
        reverse=True,
    )

    top = reranked[:top_n]
    log.debug(
        "Reranked %d → %d chunks in %.1f ms (top score: %.4f)",
        len(chunks),
        top_n,
        elapsed_ms,
        top[0][1] if top else 0.0,
    )

    # Replace the bi-encoder score with the cross-encoder score
    return [
        RetrievedChunk(
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            score=float(score),
            source=chunk.source,
            doc_id=chunk.doc_id,
            chunk_index=chunk.chunk_index,
            metadata={**chunk.metadata, "bi_encoder_score": chunk.score},
        )
        for chunk, score in top
    ]


async def rerank(
    query: str,
    chunks: list[RetrievedChunk],
    top_n: int,
) -> list[RetrievedChunk]:
    """
    Async wrapper — offloads the CPU-bound inference to a thread pool
    so it doesn't block the event loop.
    """
    if not chunks:
        return []
    if len(chunks) <= top_n:
        return chunks  # nothing to rerank

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _executor, _rerank_sync, query, chunks, top_n
    )
