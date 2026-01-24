"""Application Configuration.

This module provides a structured configuration system using Pydantic v2 settings.
Settings are organized into nested groups for better organization and type safety.

Simplified for LM Studio only with multi-agent LangGraph architecture.
"""

from typing import List, Optional

from pydantic import BaseModel, SecretStr, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LMStudioSettings(BaseModel):
    """LM Studio provider settings."""
    
    base_url: str = "http://localhost:1234/v1"
    model: str = "Qwen3-VL-30B-Instruct"


class LLMSettings(BaseModel):
    """All LLM-related settings (LM Studio only)."""
    
    # Active provider (only lmstudio supported now)
    provider: str = "lmstudio"
    
    # LM Studio settings
    lmstudio: LMStudioSettings = LMStudioSettings()
    
    # Common settings
    temperature: float = 0.35
    max_tokens: int = 4096
    timeout: float = 120.0
    context_window: int = 32000
    
    def get_active_provider(self) -> LMStudioSettings:
        """Get settings for the active provider (LM Studio)."""
        return self.lmstudio


class RAGSettings(BaseModel):
    """All RAG/Vector Store settings."""
    
    # Vector store provider (chromadb)
    provider: str = "chromadb"
    
    # ChromaDB settings
    chroma_db_path: str = "./data/chroma"
    url: Optional[str] = None  # For remote ChromaDB
    
    # Embedding settings
    embedding_model: str = "text-embedding-nomic-embed-text-v1.5"
    embedding_provider: str = "lmstudio"
    embedding_dimension: int = 768
    embedding_normalize: bool = True
    
    # Collection settings
    collection_prefix: str = ""
    
    # ==========================================================================
    # Chunking Configuration
    # ==========================================================================
    chunk_size: int = 1200         # Target characters per chunk
    chunk_overlap: int = 200       # Overlap characters between chunks
    chunk_token_size: int = 512    # Target tokens per chunk (for token mode)
    chunk_token_overlap: int = 50  # Overlap tokens between chunks
    min_chunk_tokens: int = 50     # Minimum chunk size in tokens
    chunk_method: str = "char"     # "char" or "token" based chunking
    respect_headings: bool = True  # Try to break at markdown headers
    respect_sentences: bool = True # Try to break at sentence boundaries
    use_content_hash_ids: bool = True  # Use content-hash for deterministic IDs
    
    # ==========================================================================
    # Reranker settings
    # ==========================================================================
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_enabled: bool = True
    
    # Retrieval settings
    initial_fetch_k: int = 30  # Fetch more candidates for reranking
    rerank_top_k: int = 5      # Keep top K after reranking
    mmr_lambda: float = 0.5    # MMR diversity parameter
    
    # ==========================================================================
    # Hybrid Search settings (BM25 + Vector)
    # ==========================================================================
    bm25_enabled: bool = True
    bm25_weight: float = 0.4      # Weight for BM25 in RRF fusion
    vector_weight: float = 0.6    # Weight for vector search in RRF fusion
    bm25_persist_path: str = "./data/bm25"
    
    # ==========================================================================
    # Knowledge Graph settings (prepared for future use)
    # ==========================================================================
    knowledge_graph_enabled: bool = False
    kg_expansion_depth: int = 2
    kg_persist_path: str = "./data/knowledge_graphs"


class RedisSettings(BaseModel):
    """Redis settings for sessions, caching, and memory persistence."""
    
    url: str = "redis://localhost:6379"
    session_ttl: int = 86400  # 24 hours
    cache_ttl: int = 3600  # 1 hour


class MemorySettings(BaseModel):
    """Memory and auto-summarization settings."""
    
    # Context thresholds for auto-summarization
    max_context_tokens: int = 4000  # Trigger summarization when exceeded
    summary_target_tokens: int = 1000  # Target size after summarization
    sliding_window_size: int = 10  # Recent messages to keep
    
    # Redis key prefixes
    redis_key_prefix: str = "memory:"
    summary_key_prefix: str = "summary:"


class AgentSettings(BaseModel):
    """LangGraph multi-agent settings."""
    
    enabled: bool = True
    max_steps: int = 10
    
    # LangSmith tracing
    langsmith_tracing: bool = True
    langsmith_project: str = "agentic-rag"


class SessionSettings(BaseModel):
    """Session settings (legacy compatibility)."""
    
    max_tokens: int = 2500
    max_messages: int = 15
    sliding_window_turns: int = 3
    episodic_memory_enabled: bool = True
    running_summary_enabled: bool = True


class ServerSettings(BaseModel):
    """HTTP server settings."""
    
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: List[str] = [
        "http://localhost",
        "http://localhost:80",
        "http://127.0.0.1",
        "http://127.0.0.1:80",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ]


class JWTSettings(BaseModel):
    """JWT authentication settings."""
    
    secret_key: SecretStr = SecretStr("change-me-in-production-use-strong-secret")
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60


class LoggingSettings(BaseModel):
    """Logging settings."""
    
    level: str = "INFO"
    log_dir: str = "./logs"
    log_file: str = "app.log"
    max_bytes: int = 10 * 1024 * 1024  # 10MB per file
    backup_count: int = 30
    json_format: bool = True


class Settings(BaseSettings):
    """Root application settings.
    
    Combines all configuration groups into a single settings object.
    Supports loading from environment variables with nested prefixes.
    
    Usage:
        from app.config import settings
        
        # Access LLM settings
        provider = settings.llm.get_active_provider()
        
        # Access RAG settings
        path = settings.rag.chroma_db_path
        
        # Access Memory settings for auto-summarization
        max_tokens = settings.memory.max_context_tokens
    """
    
    # Grouped settings
    llm: LLMSettings = LLMSettings()
    rag: RAGSettings = RAGSettings()
    redis: RedisSettings = RedisSettings()
    memory: MemorySettings = MemorySettings()
    agent: AgentSettings = AgentSettings()
    session: SessionSettings = SessionSettings()
    server: ServerSettings = ServerSettings()
    jwt: JWTSettings = JWTSettings()
    logging: LoggingSettings = LoggingSettings()
    
    # ============================================================
    # LEGACY ALIASES - For backwards compatibility
    # ============================================================
    
    # LLM provider aliases
    @property
    def llm_provider(self) -> str:
        return self.llm.provider
    
    @property
    def lmstudio_base_url(self) -> str:
        return self.llm.lmstudio.base_url
    
    @property
    def lmstudio_model(self) -> str:
        return self.llm.lmstudio.model
    
    @property
    def llm_temperature(self) -> float:
        return self.llm.temperature
    
    @property
    def llm_max_tokens(self) -> int:
        return self.llm.max_tokens
    
    @property
    def llm_timeout(self) -> float:
        return self.llm.timeout
    
    @property
    def context_window(self) -> int:
        return self.llm.context_window
    
    @property
    def lmstudio_temperature(self) -> float:
        return self.llm.temperature
    
    @property
    def lmstudio_max_tokens(self) -> int:
        return self.llm.max_tokens
    
    # RAG aliases
    @property
    def chroma_db_path(self) -> str:
        return self.rag.chroma_db_path
    
    @property
    def embedding_model(self) -> str:
        return self.rag.embedding_model
    
    @property
    def vector_store_provider(self) -> str:
        return self.rag.provider
    
    @property
    def vector_store_url(self) -> Optional[str]:
        return self.rag.url
    
    @property
    def vector_store_prefix(self) -> str:
        return self.rag.collection_prefix
    
    # Session aliases
    @property
    def session_max_tokens(self) -> int:
        return self.session.max_tokens
    
    @property
    def session_max_messages(self) -> int:
        return self.session.max_messages
    
    # Server aliases
    @property
    def backend_host(self) -> str:
        return self.server.host
    
    @property
    def backend_port(self) -> int:
        return self.server.port
    
    @property
    def backend_cors_origins(self) -> List[str]:
        return self.server.cors_origins
    
    # JWT aliases
    @property
    def jwt_secret_key(self) -> SecretStr:
        return self.jwt.secret_key
    
    @property
    def jwt_algorithm(self) -> str:
        return self.jwt.algorithm
    
    @property
    def jwt_access_token_expire_minutes(self) -> int:
        return self.jwt.access_token_expire_minutes
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",
        env_nested_delimiter="__",
        extra="ignore",
    )


settings = Settings()
