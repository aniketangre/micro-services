"""
All configuration in one place, loaded from environment variables.
Usage:
    from rag_pipeline.config.settings import settings
    print(settings.qdrant_host)

Environment variables:
    OPENAI_API_KEY          Required
    QDRANT_HOST             default: localhost
    QDRANT_PORT             default: 6333
    QDRANT_GRPC_PORT        default: 6334
    QDRANT_API_KEY          Optional (for Qdrant Cloud)
    REDIS_URL               default: redis://localhost:6379/0
    COLLECTION_NAME         default: rag_docs
    EMBED_MODEL             default: text-embedding-3-small
    EMBED_DIMENSIONS        default: 1536
    EMBED_BATCH_SIZE        default: 100
    LLM_MODEL               default: gpt-4o-mini
    LLM_TEMPERATURE         default: 0.0
    CHUNK_SIZE              default: 512
    CHUNK_OVERLAP           default: 64
    RETRIEVAL_TOP_K         default: 8
    RERANK_TOP_N            default: 3
    SCORE_THRESHOLD         default: 0.70
    CACHE_TTL_SECONDS       default: 604800 (7 days)
    USE_RERANKER            default: true
    LOG_LEVEL               default: INFO
"""

import os
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- OpenAI ---
    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")

    # --- Qdrant ---
    qdrant_host: str = Field("localhost", alias="QDRANT_HOST")
    qdrant_port: int = Field(6333, alias="QDRANT_PORT")
    qdrant_grpc_port: int = Field(6334, alias="QDRANT_GRPC_PORT")
    qdrant_api_key: str | None = Field(None, alias="QDRANT_API_KEY")
    qdrant_prefer_grpc: bool = Field(True, alias="QDRANT_PREFER_GRPC")

    # --- Redis (embedding cache) ---
    redis_url: str = Field("redis://localhost:6379/0", alias="REDIS_URL")

    # --- Collection ---
    collection_name: str = Field("rag_docs", alias="COLLECTION_NAME")

    # --- Embedding ---
    embed_model: str = Field("text-embedding-3-small", alias="EMBED_MODEL")
    embed_dimensions: int = Field(1536, alias="EMBED_DIMENSIONS")
    embed_batch_size: int = Field(100, alias="EMBED_BATCH_SIZE")

    # --- LLM ---
    llm_model: str = Field("gpt-4o-mini", alias="LLM_MODEL")
    llm_temperature: float = Field(0.0, alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(1024, alias="LLM_MAX_TOKENS")
    llm_timeout: float = Field(30.0, alias="LLM_TIMEOUT")

    # --- Chunking ---
    chunk_size: int = Field(512, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(64, alias="CHUNK_OVERLAP")

    # --- Retrieval ---
    retrieval_top_k: int = Field(8, alias="RETRIEVAL_TOP_K")
    rerank_top_n: int = Field(3, alias="RERANK_TOP_N")
    score_threshold: float = Field(0.70, alias="SCORE_THRESHOLD")
    use_reranker: bool = Field(True, alias="USE_RERANKER")

    # --- Cache ---
    cache_ttl_seconds: int = Field(604_800, alias="CACHE_TTL_SECONDS")  # 7 days

    # --- Observability ---
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    enable_tracing: bool = Field(False, alias="ENABLE_TRACING")

    model_config = {"populate_by_name": True, "env_file": ".env"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
