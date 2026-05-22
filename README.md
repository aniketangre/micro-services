# RAG Pipeline — Production-Grade Python Implementation

## Architecture

```
rag_pipeline/
├── config/
│   └── settings.py          # Pydantic settings from env vars
├── core/
│   ├── models.py            # Document, Chunk, QueryResult (shared dataclasses)
│   ├── embeddings.py        # EmbeddingService — cached, batched, with retry
│   └── vector_store.py      # VectorStore — Qdrant client with HNSW index mgmt
├── indexing/
│   └── pipeline.py          # IndexingPipeline — load → chunk → embed → upsert
├── query/
│   ├── pipeline.py          # QueryPipeline — embed → retrieve → rerank → generate
│   └── reranker.py          # CrossEncoder reranker (sentence-transformers)
├── api.py                   # FastAPI service
└── tests.py                 # pytest-asyncio test suite
```

## Quick start

```bash
# 1. Start infrastructure
docker compose up qdrant redis -d

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure
export OPENAI_API_KEY=sk-...

# 4. Run the API
uvicorn rag_pipeline.api:app --reload

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
from rag_pipeline.indexing.pipeline import IndexingPipeline
from rag_pipeline.query.pipeline import QueryPipeline

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

See `config/settings.py` for full list. Minimum required:

```
OPENAI_API_KEY=sk-...
```

All others have sensible defaults for local development.

## Running tests

```bash
pytest rag_pipeline/tests.py -v --asyncio-mode=auto
```
