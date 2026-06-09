# RAG Pipeline — Production-Grade Python Implementation

## Architecture

```
rag_pipeline/
├── config/
│   └── settings.py              # Pydantic settings from env vars
├── core/
│   └── models.py                # Document, Chunk, QueryResult (shared dataclasses)
├── services/                    # External system wrappers — each is a standalone service
│   ├── embedding_service.py     # EmbeddingService — OpenAI embeddings + Redis cache
│   ├── vector_store.py          # VectorStore — Qdrant client with HNSW index mgmt
│   └── reranker.py              # Reranker — CrossEncoder (sentence-transformers)
├── pipelines/                   # Business orchestration — wire the services together
│   ├── indexing_pipeline.py     # IndexingPipeline — load → chunk → embed → upsert
│   └── query_pipeline.py        # QueryPipeline — embed → retrieve → rerank → generate
└── api/
    └── app.py                   # FastAPI service
tests/
└── test_pipelines.py            # pytest-asyncio test suite
```

### Layer responsibilities

| Layer | What lives here | Rule |
|---|---|---|
| `config/` | Env-driven settings singleton | No business logic |
| `core/` | Shared dataclasses (Document, Chunk, QueryResult) | No I/O, no imports from other layers |
| `services/` | One class per external system (OpenAI, Qdrant, Redis) | Knows nothing about pipelines |
| `pipelines/` | Orchestration — composes services into multi-step flows | Imports services, not the API |
| `api/` | HTTP transport layer | Thin — delegates immediately to pipelines |

## Quick start

```bash
# 1. Start infrastructure
docker compose up qdrant redis -d

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure
export OPENAI_API_KEY=sk-...

# 4. Run the API
uvicorn rag_pipeline.api.app:app --reload

# 5. Index a document
curl -X POST http://localhost:8000/index/documents \
  -H "Content-Type: application/json" \
  -d '[{"content": "Refunds are accepted within 30 days.", "source": "policy.txt"}]'

# 6. Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the refund policy?"}'
```

## Using the pipelines directly

```python
import asyncio
from rag_pipeline.core.models import Document
from rag_pipeline.pipelines.indexing_pipeline import IndexingPipeline
from rag_pipeline.pipelines.query_pipeline import QueryPipeline

async def main():
    # --- Indexing ---
    async with IndexingPipeline() as indexer:
        results = await indexer.index_documents([
            Document(
                content="Refunds are accepted within 30 days of purchase.",
                source="policy.txt",
            )
        ])
        print(results)

    # --- Querying ---
    async with QueryPipeline() as querier:
        result = await querier.query("What is the refund period?")
        print(result.answer)
        print(result.sources)

        # With streaming
        async for token in querier.stream("Summarise the policy"):
            print(token, end="", flush=True)

asyncio.run(main())
```

## Key design decisions

| Decision | Choice | Rationale |
|---|---|---|
| Async everywhere | asyncio | Embedding and LLM calls are I/O-bound; async avoids thread overhead |
| Content-hash IDs | MD5(source:content_hash) | Idempotent upserts — re-indexing same doc is a no-op |
| L2 normalisation | Applied at embed time | dot-product == cosine; faster SIMD in Qdrant |
| Redis TTL cache | 7 days | Query embeddings are deterministic; cache cuts API calls 60-80% |
| Semaphore | MAX_CONCURRENT=8 | Prevents OOM when indexing thousands of docs in parallel |
| Reranker in thread pool | ThreadPoolExecutor | CPU-bound inference doesn't block the async event loop |
| Score threshold | 0.70 default | Suppresses irrelevant retrievals before they reach the LLM |

## Environment variables

See `rag_pipeline/config/settings.py` for full list. Minimum required:

```
OPENAI_API_KEY=sk-...
```

All others have sensible defaults for local development.

## Running tests

```bash
pytest tests/ -v --asyncio-mode=auto
```
