"""
Query pipeline — handles the online (user-facing) side of RAG.

Stages:
  1. Embed query        — same model as indexing (cache hit typical)
  2. Retrieve           — ANN search in Qdrant, with optional metadata filter
  3. Rerank             — cross-encoder re-scores top-k (optional)
  4. Build prompt       — structured context injection
  5. Generate           — streaming or non-streaming LLM call

Advanced features:
  - Query rewriting: LLM expands/corrects the user query before embedding
  - HyDE: generates a hypothetical answer then embeds that instead of the query
  - Streaming: async generator yields tokens as they arrive
  - Source attribution: every answer includes cited chunk sources

Usage:
    pipeline = QueryPipeline()
    async with pipeline:
        # Simple
        result = await pipeline.query("What is the refund policy?")
        print(result.answer)

        # With filters
        result = await pipeline.query(
            "What changed in v2?",
            filters={"source": "changelog.md"},
        )

        # Streaming
        async for token in pipeline.stream("Summarise the pricing tiers"):
            print(token, end="", flush=True)

        # With query rewriting
        result = await pipeline.query(
            "rfnd plcy?",
            use_query_rewrite=True,
        )
"""

from __future__ import annotations

import logging
import time
from typing import Any, AsyncIterator

from openai import AsyncOpenAI

from rag_pipeline.config.settings import settings
from rag_pipeline.core.embeddings import EmbeddingService
from rag_pipeline.core.models import QueryResult, RetrievedChunk
from rag_pipeline.core.vector_store import VectorStore
from rag_pipeline.query.reranker import rerank

log = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a precise question-answering assistant.
Answer the user's question using ONLY the context passages provided below.
Rules:
- If the answer is not found in the context, reply: "I don't have enough information in the provided context to answer this question."
- Do not use prior knowledge beyond what is in the context.
- Cite the source file(s) in parentheses at the end of each factual claim, e.g. (policy.pdf, section 3).
- Be concise. Prefer bullet points for multi-part answers.
"""

_CONTEXT_TEMPLATE = """\
=== Context passages ===
{context}
=== End context ===

Question: {question}
"""

_REWRITE_PROMPT = """\
Rewrite the following user query to be clearer and more specific for a document search.
Expand abbreviations, fix typos, add relevant synonyms.
Return ONLY the rewritten query — no explanation, no quotes.

Original query: {query}
Rewritten query:"""

_HYDE_PROMPT = """\
Write a short, factual paragraph that would directly answer the following question.
This is used to improve document retrieval, so be specific and use domain terminology.
Return ONLY the paragraph — no preamble.

Question: {question}
Hypothetical answer:"""


class QueryPipeline:
    def __init__(self) -> None:
        self._embed = EmbeddingService()
        self._store = VectorStore()
        self._llm = AsyncOpenAI(api_key=settings.openai_api_key)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "QueryPipeline":
        await self._embed.__aenter__()
        await self._store.__aenter__()
        return self

    async def __aexit__(self, *args) -> None:
        await self._embed.__aexit__(*args)
        await self._store.__aexit__(*args)

    # ------------------------------------------------------------------
    # Main query entry point
    # ------------------------------------------------------------------

    async def query(
        self,
        question: str,
        *,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        use_query_rewrite: bool = False,
        use_hyde: bool = False,
        use_mmr: bool = False,
    ) -> QueryResult:
        """
        Full RAG pipeline. Returns a QueryResult with answer, sources, and metrics.

        top_k:              override the default retrieval_top_k from settings
        filters:            Qdrant payload filters, e.g. {"source": "manual.pdf"}
        use_query_rewrite:  LLM rewrites the query before embedding (fixes typos, expands abbreviations)
        use_hyde:           Embed a hypothetical answer instead of the raw query (improves recall on keyword queries)
        use_mmr:            Use Maximal Marginal Relevance for diverse retrieval
        """
        k = top_k or settings.retrieval_top_k

        # --- Stage 1: Optional query transformation ---
        search_text = question
        if use_hyde:
            search_text = await self._generate_hypothesis(question)
            log.debug("HyDE hypothesis: %r", search_text[:120])
        elif use_query_rewrite:
            search_text = await self._rewrite_query(question)
            log.debug("Rewritten query: %r", search_text)

        # --- Stage 2: Embed ---
        t_retrieve_start = time.perf_counter()
        query_vector = await self._embed.embed_query(search_text)

        # --- Stage 3: Retrieve ---
        if use_mmr:
            chunks = await self._store.search_with_mmr(
                query_vector=query_vector,
                top_k=k,
                fetch_k=k * 3,
                score_threshold=settings.score_threshold,
                filters=filters,
            )
        else:
            chunks = await self._store.search(
                query_vector=query_vector,
                top_k=k,
                score_threshold=settings.score_threshold,
                filters=filters,
            )

        # --- Stage 4: Rerank ---
        if settings.use_reranker and len(chunks) > settings.rerank_top_n:
            chunks = await rerank(question, chunks, top_n=settings.rerank_top_n)
        elif len(chunks) > settings.rerank_top_n:
            chunks = chunks[: settings.rerank_top_n]

        retrieval_ms = (time.perf_counter() - t_retrieve_start) * 1000

        log.info(
            "Retrieved %d chunks in %.0f ms (query=%r)",
            len(chunks),
            retrieval_ms,
            question[:60],
        )

        # --- Stage 5: Generate ---
        t_gen_start = time.perf_counter()
        prompt = _build_prompt(question, chunks)
        response = await self._llm.chat.completions.create(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        generation_ms = (time.perf_counter() - t_gen_start) * 1000
        answer = response.choices[0].message.content or ""

        return QueryResult(
            question=question,
            answer=answer,
            chunks=chunks,
            model=response.model,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
        )

    # ------------------------------------------------------------------
    # Streaming entry point
    # ------------------------------------------------------------------

    async def stream(
        self,
        question: str,
        *,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        use_query_rewrite: bool = False,
    ) -> AsyncIterator[str]:
        """
        Stream answer tokens as they arrive from the LLM.
        Retrieval and reranking complete before streaming starts.

        Usage:
            async for token in pipeline.stream("What is the return policy?"):
                print(token, end="", flush=True)
        """
        k = top_k or settings.retrieval_top_k

        search_text = question
        if use_query_rewrite:
            search_text = await self._rewrite_query(question)

        query_vector = await self._embed.embed_query(search_text)
        chunks = await self._store.search(
            query_vector=query_vector,
            top_k=k,
            score_threshold=settings.score_threshold,
            filters=filters,
        )

        if settings.use_reranker and len(chunks) > settings.rerank_top_n:
            chunks = await rerank(question, chunks, top_n=settings.rerank_top_n)

        prompt = _build_prompt(question, chunks)

        stream = await self._llm.chat.completions.create(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout,
            stream=True,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )

        async for event in stream:
            delta = event.choices[0].delta.content
            if delta:
                yield delta

    # ------------------------------------------------------------------
    # Multi-turn conversation
    # ------------------------------------------------------------------

    async def chat(
        self,
        question: str,
        history: list[dict[str, str]],
        *,
        filters: dict[str, Any] | None = None,
    ) -> QueryResult:
        """
        Multi-turn RAG. *history* is a list of {"role": ..., "content": ...} dicts.
        The question is condensed with history context before retrieval.
        """
        if history:
            condensed = await self._condense_question(question, history)
            log.debug("Condensed question: %r", condensed)
        else:
            condensed = question

        result = await self.query(condensed, filters=filters)
        return result

    # ------------------------------------------------------------------
    # Query transformation helpers
    # ------------------------------------------------------------------

    async def _rewrite_query(self, query: str) -> str:
        """Expand and clean the raw user query using an LLM.
        Fixes typos, expands abbreviations, and adds synonyms so the
        embedding better covers the intended search intent.
        Falls back to the original query if the LLM returns an empty response.
        """
        response = await self._llm.chat.completions.create(
            model="gpt-4o-mini",        # use a fast/cheap model for rewriting
            temperature=0.0,
            max_tokens=128,
            messages=[
                {"role": "user", "content": _REWRITE_PROMPT.format(query=query)}
            ],
        )
        return (response.choices[0].message.content or query).strip()

    async def _generate_hypothesis(self, question: str) -> str:
        """HyDE: generate a hypothetical document passage for embedding."""
        response = await self._llm.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.3,
            max_tokens=256,
            messages=[
                {"role": "user", "content": _HYDE_PROMPT.format(question=question)}
            ],
        )
        return (response.choices[0].message.content or question).strip()

    async def _condense_question(
        self, question: str, history: list[dict[str, str]]
    ) -> str:
        """Rephrase the question to be standalone given conversation history."""
        # Limit to the last 6 messages (3 user/assistant pairs) to keep the
        # condensation prompt short and avoid exceeding the model's context window.
        history_text = "\n".join(
            f"{m['role'].capitalize()}: {m['content']}" for m in history[-6:]
        )
        prompt = (
            f"Given the conversation history below, rephrase the follow-up question "
            f"as a standalone question that can be understood without the history.\n\n"
            f"History:\n{history_text}\n\n"
            f"Follow-up question: {question}\n"
            f"Standalone question:"
        )
        response = await self._llm.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.0,
            max_tokens=128,
            messages=[{"role": "user", "content": prompt}],
        )
        return (response.choices[0].message.content or question).strip()


# ------------------------------------------------------------------
# Prompt construction
# ------------------------------------------------------------------

def _build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    """
    Assemble the context block from retrieved chunks.
    Each chunk is labelled with its source for citation.
    """
    if not chunks:
        context = "(No relevant context was found in the knowledge base.)"
    else:
        parts: list[str] = []
        for i, chunk in enumerate(chunks, 1):
            source_label = f"[{i}] Source: {chunk.source} (score: {chunk.score:.3f})"
            parts.append(f"{source_label}\n{chunk.text}")
        context = "\n\n---\n\n".join(parts)

    return _CONTEXT_TEMPLATE.format(context=context, question=question)
