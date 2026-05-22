"""
EmbeddingService — async, cached, batched.

Features:
  - SHA-256 keyed Redis cache (TTL configurable, default 7 days)
  - Automatic batch splitting to respect OpenAI's 2048-input-per-request limit
  - Exponential back-off retry on RateLimitError / APIConnectionError
  - L2 normalisation so dot-product == cosine at query time (faster SIMD)
  - Context manager lifecycle: await svc.close() frees the aioredis connection

Usage:
    async with EmbeddingService() as svc:
        vectors = await svc.embed(["hello world", "foo bar"])
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from typing import Sequence

import numpy as np
from openai import AsyncOpenAI, RateLimitError, APIConnectionError, APITimeoutError
from redis.asyncio import Redis

from rag_pipeline.config.settings import settings

log = logging.getLogger(__name__)

# OpenAI hard limit: 2048 texts per embed call; stay conservative
_MAX_BATCH = min(settings.embed_batch_size, 512)
_RETRY_STATUSES = (RateLimitError, APIConnectionError, APITimeoutError)
_MAX_RETRIES = 5


class EmbeddingService:
    def __init__(self) -> None:
        self._client = AsyncOpenAI(api_key=settings.openai_api_key)
        self._redis: Redis | None = None
        self._model = settings.embed_model
        self._dims = settings.embed_dimensions
        self._ttl = settings.cache_ttl_seconds

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "EmbeddingService":
        self._redis = Redis.from_url(settings.redis_url, decode_responses=False)
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the Redis connection. Safe to call multiple times."""
        if self._redis:
            await self._redis.aclose()
            self._redis = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Return L2-normalised embeddings for *texts*.
        Results are in the same order as inputs.
        """
        if not texts:
            return []

        cache_keys = [self._cache_key(t) for t in texts]

        # 1. Bulk cache read
        cached = await self._mget(cache_keys)

        results: list[list[float] | None] = list(cached)
        missing_indices = [i for i, v in enumerate(results) if v is None]

        if not missing_indices:
            log.debug("embed: 100%% cache hit (%d texts)", len(texts))
            return results  # type: ignore[return-value]

        log.debug(
            "embed: %d/%d cache miss, calling API",
            len(missing_indices),
            len(texts),
        )

        # 2. Embed uncached texts in batches
        missing_texts = [texts[i] for i in missing_indices]
        new_vectors = await self._embed_batched(missing_texts)

        # 3. Write new vectors back to cache
        pipe_data = {
            cache_keys[missing_indices[j]]: json.dumps(new_vectors[j])
            for j in range(len(missing_indices))
        }
        await self._mset(pipe_data)

        # 4. Stitch results back in order
        for j, idx in enumerate(missing_indices):
            results[idx] = new_vectors[j]

        return results  # type: ignore[return-value]

    async def embed_query(self, text: str) -> list[float]:
        """Convenience wrapper for a single query string."""
        vecs = await self.embed([text])
        return vecs[0]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cache_key(self, text: str) -> str:
        # Model name is included in the hash so swapping embed models
        # automatically invalidates all cached vectors for old models.
        digest = hashlib.sha256(f"{self._model}:{text}".encode()).hexdigest()
        return f"emb:{digest}"

    async def _mget(self, keys: list[str]) -> list[list[float] | None]:
        """Batch-read vectors from Redis. Returns None for each cache miss."""
        if not self._redis:
            return [None] * len(keys)
        raw = await self._redis.mget(*keys)
        out: list[list[float] | None] = []
        for r in raw:
            out.append(json.loads(r) if r is not None else None)
        return out

    async def _mset(self, mapping: dict[str, str]) -> None:
        """Batch-write vectors to Redis using a pipeline to reduce round-trips.
        Each key is set with the configured TTL (default 7 days).
        """
        if not self._redis or not mapping:
            return
        async with self._redis.pipeline(transaction=False) as pipe:
            for k, v in mapping.items():
                await pipe.setex(k, self._ttl, v)
            await pipe.execute()

    async def _embed_batched(self, texts: list[str]) -> list[list[float]]:
        """Split into ≤_MAX_BATCH chunks, embed each, and reassemble."""
        all_vectors: list[list[float]] = []
        batches = [texts[i : i + _MAX_BATCH] for i in range(0, len(texts), _MAX_BATCH)]

        for batch in batches:
            vecs = await self._embed_with_retry(batch)
            all_vectors.extend(vecs)

        return all_vectors

    async def _embed_with_retry(self, texts: list[str]) -> list[list[float]]:
        delay = 1.0
        for attempt in range(_MAX_RETRIES):
            try:
                t0 = time.perf_counter()
                resp = await self._client.embeddings.create(
                    model=self._model,
                    input=texts,
                    dimensions=self._dims,
                )
                elapsed = (time.perf_counter() - t0) * 1000
                log.debug("OpenAI embed: %d texts in %.1f ms", len(texts), elapsed)

                raw = np.array([e.embedding for e in resp.data], dtype="float32")
                # L2 normalise so dot-product ≡ cosine similarity
                norms = np.linalg.norm(raw, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1.0, norms)
                normalised = raw / norms
                return normalised.tolist()

            except _RETRY_STATUSES as exc:
                if attempt == _MAX_RETRIES - 1:
                    raise
                wait = delay * (2**attempt)
                log.warning(
                    "embed attempt %d/%d failed (%s), retrying in %.1fs",
                    attempt + 1,
                    _MAX_RETRIES,
                    type(exc).__name__,
                    wait,
                )
                await asyncio.sleep(wait)

        raise RuntimeError("unreachable")  # pragma: no cover
