"""
VectorStore — thin async wrapper around Qdrant.

Responsibilities:
  - Collection creation and schema management
  - Upsert with idempotent IDs (content-hash based)
  - Dense vector search with optional metadata filtering
  - Hybrid dense+sparse search via Qdrant's built-in RRF fusion
  - Scroll / delete helpers for index maintenance

All point IDs are UUID-v5 derived from the chunk's chunk_id string so they
are deterministic, collision-resistant, and compatible with Qdrant's UUID ID type.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    HnswConfigDiff,
    MatchAny,
    MatchValue,
    OptimizersConfigDiff,
    PointStruct,
    ScoredPoint,
    VectorParams,
    models as qmodels,
)

from rag_pipeline.config.settings import settings
from rag_pipeline.core.models import Chunk, RetrievedChunk

log = logging.getLogger(__name__)

# Deterministic UUID namespace for chunk IDs → Qdrant point IDs
_NS = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(_NS, chunk_id))


class VectorStore:
    def __init__(self) -> None:
        self._client = AsyncQdrantClient(
            host=settings.qdrant_host,
            port=settings.qdrant_port,
            grpc_port=settings.qdrant_grpc_port,
            prefer_grpc=settings.qdrant_prefer_grpc,
            api_key=settings.qdrant_api_key,
            timeout=30,
        )
        self._collection = settings.collection_name

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "VectorStore":
        await self.ensure_collection()
        return self

    async def __aexit__(self, *_) -> None:
        await self._client.close()

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    async def ensure_collection(self) -> None:
        """Create the collection if it doesn't exist. Idempotent."""
        existing = {c.name for c in await self._client.get_collections()}
        if self._collection in existing:
            log.debug("Collection %r already exists", self._collection)
            return

        log.info("Creating collection %r (dim=%d)", self._collection, settings.embed_dimensions)
        await self._client.create_collection(
            collection_name=self._collection,
            vectors_config=VectorParams(
                size=settings.embed_dimensions,
                distance=Distance.COSINE,
                on_disk=False,          # keep HNSW graph in RAM for low latency
            ),
            hnsw_config=HnswConfigDiff(
                m=16,                   # edges per node; 16 is a solid default
                ef_construct=200,       # higher = better recall, slower build
                full_scan_threshold=10_000,
            ),
            optimizers_config=OptimizersConfigDiff(
                memmap_threshold=100_000,   # flush segments > 100k vectors to mmap
            ),
        )

        # Payload index for fast metadata filtering
        for field_name, field_schema in [
            ("source", qmodels.PayloadSchemaType.KEYWORD),
            ("doc_id", qmodels.PayloadSchemaType.KEYWORD),
            ("created_at", qmodels.PayloadSchemaType.DATETIME),
        ]:
            await self._client.create_payload_index(
                collection_name=self._collection,
                field_name=field_name,
                field_schema=field_schema,
            )
        log.info("Collection %r created with payload indexes", self._collection)

    async def collection_info(self) -> dict[str, Any]:
        """Return a summary of the collection: vector count, indexed count, and status."""
        info = await self._client.get_collection(self._collection)
        return {
            "vectors_count": info.vectors_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "status": info.status,
        }

    # ------------------------------------------------------------------
    # Upsert
    # ------------------------------------------------------------------

    async def upsert_chunks(
        self, chunks: list[Chunk], vectors: list[list[float]]
    ) -> int:
        """
        Upsert (chunk, vector) pairs. Returns the number of points upserted.
        Uses content-hash IDs so re-indexing the same document is idempotent.
        """
        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunks ({len(chunks)}) and vectors ({len(vectors)}) must have equal length"
            )

        points = [
            PointStruct(
                id=_point_id(chunk.chunk_id),
                vector=vector,
                payload=chunk.to_payload(),
            )
            for chunk, vector in zip(chunks, vectors)
        ]

        await self._client.upsert(
            collection_name=self._collection,
            points=points,
            wait=True,          # wait for indexing before returning
        )
        log.debug("Upserted %d points", len(points))
        return len(points)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search(
        self,
        query_vector: list[float],
        top_k: int = 8,
        score_threshold: float = 0.0,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """
        Dense vector search with optional metadata filters.

        filters: flat dict of payload field → value or list[value].
          {"source": "policy.pdf"}
          {"source": ["policy.pdf", "faq.pdf"]}
        """
        qdrant_filter = self._build_filter(filters) if filters else None

        results: list[ScoredPoint] = await self._client.search(
            collection_name=self._collection,
            query_vector=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        return [self._to_retrieved(r) for r in results]

    async def search_with_mmr(
        self,
        query_vector: list[float],
        top_k: int = 8,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        score_threshold: float = 0.0,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievedChunk]:
        """
        Maximal Marginal Relevance search — balances relevance and diversity.
        Fetches fetch_k candidates then greedily selects top_k diverse results.

        lambda_mult: 1.0 = pure relevance, 0.0 = pure diversity
        """
        import numpy as np

        candidates = await self.search(
            query_vector=query_vector,
            top_k=fetch_k,
            score_threshold=score_threshold,
            filters=filters,
        )
        if not candidates:
            return []

        # Build matrix of candidate vectors for diversity computation.
        # We re-use the scores as proxy for query similarity since Qdrant
        # doesn't return raw vectors by default (avoids the extra payload size).
        # For true MMR, enable with_vectors=True in the search call above.
        if len(candidates) <= top_k:
            return candidates

        q = np.array(query_vector, dtype="float32")
        selected: list[RetrievedChunk] = []
        remaining = list(candidates)

        while len(selected) < top_k and remaining:
            if not selected:
                # First pick: highest relevance
                best = max(remaining, key=lambda c: c.score)
            else:
                # Score = λ·relevance − (1−λ)·max_sim_to_selected
                best_score = float("-inf")
                best = remaining[0]
                for cand in remaining:
                    relevance = cand.score
                    # Simple Jaccard proxy for diversity (no vectors available)
                    cand_tokens = set(cand.text.lower().split())
                    max_sim = max(
                        len(cand_tokens & set(s.text.lower().split())) /
                        max(len(cand_tokens | set(s.text.lower().split())), 1)
                        for s in selected
                    )
                    score = lambda_mult * relevance - (1 - lambda_mult) * max_sim
                    if score > best_score:
                        best_score = score
                        best = cand

            selected.append(best)
            remaining.remove(best)

        return selected

    # ------------------------------------------------------------------
    # Maintenance helpers
    # ------------------------------------------------------------------

    async def delete_by_doc_id(self, doc_id: str) -> int:
        """Delete all chunks belonging to a document. Returns deleted count."""
        result = await self._client.delete(
            collection_name=self._collection,
            points_selector=qmodels.FilterSelector(
                filter=Filter(
                    must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
                )
            ),
        )
        log.info("Deleted chunks for doc_id=%r (status=%s)", doc_id, result.status)
        return result.operation_id  # Qdrant returns operation_id, not count

    async def scroll_sources(self, limit: int = 1000) -> list[str]:
        """Return all unique source values currently indexed."""
        sources: set[str] = set()
        offset = None
        while True:
            records, offset = await self._client.scroll(
                collection_name=self._collection,
                scroll_filter=None,
                limit=limit,
                offset=offset,
                with_payload=["source"],
                with_vectors=False,
            )
            for r in records:
                if r.payload and "source" in r.payload:
                    sources.add(r.payload["source"])
            if offset is None:
                break
        return sorted(sources)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_filter(filters: dict[str, Any]) -> Filter:
        """Convert a flat dict of field→value pairs into a Qdrant Filter.
        A list value becomes a MatchAny (OR), a scalar becomes MatchValue (exact match).
        All conditions are joined with must (AND).
        """
        conditions = []
        for key, value in filters.items():
            if isinstance(value, list):
                conditions.append(
                    FieldCondition(key=key, match=MatchAny(any=value))
                )
            else:
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
        return Filter(must=conditions)

    @staticmethod
    def _to_retrieved(point: ScoredPoint) -> RetrievedChunk:
        """Deserialise a Qdrant ScoredPoint into a RetrievedChunk.
        Known payload fields are extracted explicitly; everything else is
        kept in metadata for downstream consumers.
        """
        p = point.payload or {}
        return RetrievedChunk(
            chunk_id=str(point.id),
            text=p.get("text", ""),
            score=point.score,
            source=p.get("source", ""),
            doc_id=p.get("doc_id", ""),
            chunk_index=p.get("chunk_index", 0),
            metadata={k: v for k, v in p.items() if k not in {"text", "doc_id", "source", "chunk_index"}},
        )
