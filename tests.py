"""
Tests for both pipelines using pytest-asyncio + unittest.mock.

Run:
    pytest tests/ -v --asyncio-mode=auto

The tests use patched clients so no live API keys or services are needed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from rag_pipeline.core.models import Chunk, Document, IndexStatus
from rag_pipeline.indexing.pipeline import IndexingPipeline
from rag_pipeline.query.pipeline import QueryPipeline, _build_prompt


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_document() -> Document:
    return Document(
        content="The refund policy allows returns within 30 days of purchase. "
                "Items must be in original condition. Digital downloads are non-refundable.",
        source="policy.txt",
        doc_type="text",
        title="Refund Policy",
    )


@pytest.fixture
def sample_chunk() -> Chunk:
    doc = Document(content="The refund policy allows returns within 30 days.", source="policy.txt")
    return Chunk(
        text="The refund policy allows returns within 30 days.",
        doc_id=doc.doc_id,
        source="policy.txt",
        chunk_index=0,
        token_count=12,
    )


# ---------------------------------------------------------------------------
# Model tests (no I/O)
# ---------------------------------------------------------------------------

class TestDocumentModel:
    """Verify that Document ID generation is stable and collision-resistant."""

    def test_doc_id_is_deterministic(self):
        doc1 = Document(content="hello", source="a.txt")
        doc2 = Document(content="hello", source="a.txt")
        assert doc1.doc_id == doc2.doc_id

    def test_doc_id_changes_with_content(self):
        doc1 = Document(content="hello", source="a.txt")
        doc2 = Document(content="world", source="a.txt")
        assert doc1.doc_id != doc2.doc_id

    def test_doc_id_changes_with_source(self):
        doc1 = Document(content="hello", source="a.txt")
        doc2 = Document(content="hello", source="b.txt")
        assert doc1.doc_id != doc2.doc_id


class TestChunkModel:
    """Verify Chunk serialisation and deterministic ID behaviour."""

    def test_chunk_id_is_deterministic(self, sample_chunk):
        chunk2 = Chunk(
            text=sample_chunk.text,
            doc_id=sample_chunk.doc_id,
            source=sample_chunk.source,
            chunk_index=sample_chunk.chunk_index,
        )
        assert sample_chunk.chunk_id == chunk2.chunk_id

    def test_to_payload_includes_text(self, sample_chunk):
        payload = sample_chunk.to_payload()
        assert payload["text"] == sample_chunk.text
        assert payload["source"] == sample_chunk.source
        assert payload["chunk_index"] == 0


# ---------------------------------------------------------------------------
# Prompt builder tests (no I/O)
# ---------------------------------------------------------------------------

class TestPromptBuilder:
    """Unit tests for _build_prompt — no I/O, purely string construction."""

    def test_empty_chunks_shows_no_context_message(self):
        prompt = _build_prompt("What?", [])
        assert "No relevant context" in prompt

    def test_prompt_includes_question(self):
        prompt = _build_prompt("What is the refund policy?", [])
        assert "What is the refund policy?" in prompt

    def test_prompt_includes_source(self):
        from rag_pipeline.core.models import RetrievedChunk
        chunks = [
            RetrievedChunk(
                chunk_id="abc",
                text="Returns within 30 days.",
                score=0.92,
                source="policy.txt",
                doc_id="doc1",
                chunk_index=0,
            )
        ]
        prompt = _build_prompt("What is the refund policy?", chunks)
        assert "policy.txt" in prompt
        assert "Returns within 30 days." in prompt
        assert "0.920" in prompt


# ---------------------------------------------------------------------------
# Embedding service tests (mocked Redis + OpenAI)
# ---------------------------------------------------------------------------

class TestEmbeddingService:
    """Tests for EmbeddingService with Redis and OpenAI mocked out.
    Covers L2 normalisation correctness and cache hit/miss paths.
    """

    @pytest.fixture
    def mock_redis(self):
        r = AsyncMock()
        r.mget = AsyncMock(return_value=[None, None])
        r.pipeline = MagicMock(return_value=AsyncMock())
        return r

    @pytest.fixture
    def mock_openai_embed(self):
        @dataclass
        class FakeEmbed:
            embedding: list[float]

        @dataclass
        class FakeResp:
            data: list

        resp = FakeResp(data=[FakeEmbed(embedding=[0.1] * 1536), FakeEmbed(embedding=[0.2] * 1536)])
        client = AsyncMock()
        client.embeddings.create = AsyncMock(return_value=resp)
        return client

    @pytest.mark.asyncio
    async def test_embed_returns_normalised_vectors(self, mock_redis, mock_openai_embed):
        import numpy as np
        from rag_pipeline.core.embeddings import EmbeddingService

        svc = EmbeddingService()
        svc._redis = mock_redis
        svc._client = mock_openai_embed

        results = await svc.embed(["hello", "world"])
        assert len(results) == 2
        # Check L2 normalisation
        for vec in results:
            norm = np.linalg.norm(vec)
            assert abs(norm - 1.0) < 1e-5

    @pytest.mark.asyncio
    async def test_embed_uses_cache_on_hit(self, mock_redis, mock_openai_embed):
        from rag_pipeline.core.embeddings import EmbeddingService

        cached_vec = [0.5] * 1536
        mock_redis.mget = AsyncMock(return_value=[json.dumps(cached_vec)])

        svc = EmbeddingService()
        svc._redis = mock_redis
        svc._client = mock_openai_embed

        results = await svc.embed(["cached text"])
        assert results[0] == cached_vec
        # OpenAI should NOT have been called
        mock_openai_embed.embeddings.create.assert_not_called()


# ---------------------------------------------------------------------------
# Indexing pipeline tests (mocked embedding + vector store)
# ---------------------------------------------------------------------------

class TestIndexingPipeline:
    """Integration-style tests for IndexingPipeline with all external I/O mocked.
    Verifies the happy path and that embedding errors produce FAILED results.
    """

    @pytest.mark.asyncio
    async def test_index_document_success(self, sample_document):
        with (
            patch("rag_pipeline.indexing.pipeline.EmbeddingService") as MockEmbed,
            patch("rag_pipeline.indexing.pipeline.VectorStore") as MockStore,
        ):
            embed_instance = AsyncMock()
            embed_instance.embed = AsyncMock(return_value=[[0.1] * 1536] * 10)
            embed_instance.__aenter__ = AsyncMock(return_value=embed_instance)
            embed_instance.__aexit__ = AsyncMock(return_value=None)
            MockEmbed.return_value = embed_instance

            store_instance = AsyncMock()
            store_instance.upsert_chunks = AsyncMock(return_value=3)
            store_instance.__aenter__ = AsyncMock(return_value=store_instance)
            store_instance.__aexit__ = AsyncMock(return_value=None)
            MockStore.return_value = store_instance

            async with IndexingPipeline() as pipeline:
                results = await pipeline.index_documents([sample_document])

            assert len(results) == 1
            result = results[0]
            assert result.status == IndexStatus.INDEXED
            assert result.source == "policy.txt"
            assert result.error is None

    @pytest.mark.asyncio
    async def test_index_document_handles_error_gracefully(self):
        with (
            patch("rag_pipeline.indexing.pipeline.EmbeddingService") as MockEmbed,
            patch("rag_pipeline.indexing.pipeline.VectorStore") as MockStore,
        ):
            embed_instance = AsyncMock()
            embed_instance.embed = AsyncMock(side_effect=RuntimeError("API down"))
            embed_instance.__aenter__ = AsyncMock(return_value=embed_instance)
            embed_instance.__aexit__ = AsyncMock(return_value=None)
            MockEmbed.return_value = embed_instance

            store_instance = AsyncMock()
            store_instance.__aenter__ = AsyncMock(return_value=store_instance)
            store_instance.__aexit__ = AsyncMock(return_value=None)
            MockStore.return_value = store_instance

            doc = Document(content="test", source="test.txt")
            async with IndexingPipeline() as pipeline:
                results = await pipeline.index_documents([doc])

            assert results[0].status == IndexStatus.FAILED
            assert "API down" in results[0].error


# ---------------------------------------------------------------------------
# Query pipeline tests (mocked embedding + vector store + LLM)
# ---------------------------------------------------------------------------

class TestQueryPipeline:
    """Integration-style tests for QueryPipeline with embedding, vector store, and LLM mocked.
    Verifies answer content, source attribution, and graceful handling of empty retrieval.
    """

    @pytest.fixture
    def mock_chunks(self):
        from rag_pipeline.core.models import RetrievedChunk
        return [
            RetrievedChunk(
                chunk_id="c1",
                text="Refunds are accepted within 30 days.",
                score=0.92,
                source="policy.txt",
                doc_id="d1",
                chunk_index=0,
            )
        ]

    @pytest.mark.asyncio
    async def test_query_returns_answer(self, mock_chunks):
        with (
            patch("rag_pipeline.query.pipeline.EmbeddingService") as MockEmbed,
            patch("rag_pipeline.query.pipeline.VectorStore") as MockStore,
            patch("rag_pipeline.query.pipeline.AsyncOpenAI") as MockLLM,
            patch("rag_pipeline.query.pipeline.rerank", new_callable=AsyncMock) as mock_rerank,
        ):
            embed_instance = AsyncMock()
            embed_instance.embed_query = AsyncMock(return_value=[0.1] * 1536)
            embed_instance.__aenter__ = AsyncMock(return_value=embed_instance)
            embed_instance.__aexit__ = AsyncMock(return_value=None)
            MockEmbed.return_value = embed_instance

            store_instance = AsyncMock()
            store_instance.search = AsyncMock(return_value=mock_chunks)
            store_instance.__aenter__ = AsyncMock(return_value=store_instance)
            store_instance.__aexit__ = AsyncMock(return_value=None)
            MockStore.return_value = store_instance

            mock_rerank.return_value = mock_chunks

            # Mock LLM response
            fake_msg = MagicMock()
            fake_msg.content = "Refunds are accepted within 30 days of purchase."
            fake_choice = MagicMock()
            fake_choice.message = fake_msg
            fake_usage = MagicMock()
            fake_usage.prompt_tokens = 100
            fake_usage.completion_tokens = 20
            fake_response = MagicMock()
            fake_response.choices = [fake_choice]
            fake_response.usage = fake_usage
            fake_response.model = "gpt-4o-mini"

            llm_instance = MagicMock()
            llm_instance.chat.completions.create = AsyncMock(return_value=fake_response)
            MockLLM.return_value = llm_instance

            async with QueryPipeline() as pipeline:
                result = await pipeline.query("What is the refund policy?")

            assert "30 days" in result.answer
            assert result.sources == ["policy.txt"]
            assert result.prompt_tokens == 100
            assert result.completion_tokens == 20

    @pytest.mark.asyncio
    async def test_query_returns_no_context_when_empty(self):
        with (
            patch("rag_pipeline.query.pipeline.EmbeddingService") as MockEmbed,
            patch("rag_pipeline.query.pipeline.VectorStore") as MockStore,
            patch("rag_pipeline.query.pipeline.AsyncOpenAI") as MockLLM,
            patch("rag_pipeline.query.pipeline.rerank", new_callable=AsyncMock) as mock_rerank,
        ):
            embed_instance = AsyncMock()
            embed_instance.embed_query = AsyncMock(return_value=[0.1] * 1536)
            embed_instance.__aenter__ = AsyncMock(return_value=embed_instance)
            embed_instance.__aexit__ = AsyncMock(return_value=None)
            MockEmbed.return_value = embed_instance

            store_instance = AsyncMock()
            store_instance.search = AsyncMock(return_value=[])      # empty result
            store_instance.__aenter__ = AsyncMock(return_value=store_instance)
            store_instance.__aexit__ = AsyncMock(return_value=None)
            MockStore.return_value = store_instance

            mock_rerank.return_value = []

            fake_msg = MagicMock()
            fake_msg.content = "I don't have enough information."
            fake_choice = MagicMock()
            fake_choice.message = fake_msg
            fake_usage = MagicMock(prompt_tokens=50, completion_tokens=10)
            fake_response = MagicMock(choices=[fake_choice], usage=fake_usage, model="gpt-4o-mini")

            llm_instance = MagicMock()
            llm_instance.chat.completions.create = AsyncMock(return_value=fake_response)
            MockLLM.return_value = llm_instance

            async with QueryPipeline() as pipeline:
                result = await pipeline.query("unknown topic")

            assert result.chunks == []
            assert result.sources == []
