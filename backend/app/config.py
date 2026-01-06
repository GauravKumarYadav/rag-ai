"""Application Configuration.

This module provides a structured configuration system using Pydantic v2 settings.
Settings are organized into nested groups for better organization and type safety.
"""

from typing import List, Optional

from pydantic import BaseModel, SecretStr, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProviderSettings(BaseModel):
    """Settings for a specific LLM provider."""
    
    base_url: str
    model: str
    api_key: Optional[str] = None


class LLMSettings(BaseModel):
    """All LLM-related settings."""
    
    # Active provider: "lmstudio", "ollama", "openai", "custom"
    provider: str = "lmstudio"
    
    # Provider-specific settings
    lmstudio: LLMProviderSettings = LLMProviderSettings(
        base_url="http://localhost:1234/v1",
        model="Qwen3-VL-30B-Instruct",
    )
    ollama: LLMProviderSettings = LLMProviderSettings(
        base_url="http://localhost:11434",
        model="llama3",
    )
    openai: LLMProviderSettings = LLMProviderSettings(
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
    )
    custom: LLMProviderSettings = LLMProviderSettings(
        base_url="http://localhost:8080/v1",
        model="default",
    )
    
    # Common settings applied to all providers
    temperature: float = 0.25
    max_tokens: int = 1024
    timeout: float = 60.0
    context_window: int = 16000
    
    def get_active_provider(self) -> LLMProviderSettings:
        """Get settings for the currently active provider."""
        return getattr(self, self.provider)


class RAGSettings(BaseModel):
    """All RAG/Vector Store settings."""
    
    # Vector store provider: "chromadb", "pinecone", "weaviate", "qdrant", "milvus"
    provider: str = "chromadb"
    
    # Local ChromaDB settings
    chroma_db_path: str = "./data/chroma"
    
    # Embedding model (for Ollama: nomic-embed-text, for LMStudio: text-embedding-nomic-embed-text-v1.5)
    embedding_model: str = "nomic-embed-text"
    
    # Cloud/remote store settings
    url: Optional[str] = None
    api_key: Optional[str] = None
    namespace: Optional[str] = None
    collection_prefix: str = ""
    
    # Embedding service URL (microservices mode)
    embedding_service_url: Optional[str] = None


class RedisSettings(BaseModel):
    """Redis settings for sessions, caching, and queues."""
    
    url: str = "redis://localhost:6379"
    session_ttl: int = 86400  # 24 hours
    cache_ttl: int = 3600  # 1 hour


class SessionSettings(BaseModel):
    """Session and memory settings."""
    
    max_tokens: int = 4000
    max_messages: int = 20


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
    access_token_expire_minutes: int = 30


class MySQLSettings(BaseModel):
    """MySQL audit logging database settings."""
    
    host: str = "localhost"
    port: int = 3306
    database: str = "audit_logs"
    user: str = "root"
    password: SecretStr = SecretStr("Sarita1!@2024_4")
    pool_size: int = 5


class LoggingSettings(BaseModel):
    """Structured logging settings."""
    
    # Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL
    level: str = "INFO"
    
    # Log file settings
    log_dir: str = "./logs"
    log_file: str = "app.log"
    
    # File rotation settings
    max_bytes: int = 10 * 1024 * 1024  # 10MB per file
    backup_count: int = 30  # 30 days retention
    
    # Format settings
    json_format: bool = True  # JSON for Loki/ELK, False for human-readable


class EvaluationSettings(BaseModel):
    """RAG evaluation framework settings."""
    
    # Default sample size for Q&A generation
    default_sample_size: int = 50
    
    # Cron schedule for daily evaluation (default: 2 AM UTC)
    cron_schedule: str = "0 2 * * *"
    cron_timezone: str = "UTC"
    
    # Enable/disable scheduled evaluations
    scheduler_enabled: bool = True
    
    # Metrics to compute
    compute_precision: bool = True
    compute_recall: bool = True
    compute_mrr: bool = True
    compute_faithfulness: bool = True


class MonitoringSettings(BaseModel):
    """Prometheus/Grafana monitoring settings."""
    
    # Enable Prometheus metrics endpoint
    metrics_enabled: bool = True
    
    # Loki settings
    loki_retention_days: int = 7


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
        
        # Access JWT settings
        secret = settings.jwt.secret_key.get_secret_value()
    """
    
    # Grouped settings
    llm: LLMSettings = LLMSettings()
    rag: RAGSettings = RAGSettings()
    redis: RedisSettings = RedisSettings()
    session: SessionSettings = SessionSettings()
    server: ServerSettings = ServerSettings()
    jwt: JWTSettings = JWTSettings()
    mysql: MySQLSettings = MySQLSettings()
    logging: LoggingSettings = LoggingSettings()
    evaluation: EvaluationSettings = EvaluationSettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    
    # ============================================================
    # LEGACY ALIASES - For backwards compatibility
    # These properties map old flat config names to new nested structure
    # TODO: Remove these after migrating all code to new nested access
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
    def ollama_base_url(self) -> str:
        return self.llm.ollama.base_url
    
    @property
    def ollama_model(self) -> str:
        return self.llm.ollama.model
    
    @property
    def openai_base_url(self) -> str:
        return self.llm.openai.base_url
    
    @property
    def openai_api_key(self) -> Optional[str]:
        return self.llm.openai.api_key
    
    @property
    def openai_model(self) -> str:
        return self.llm.openai.model
    
    @property
    def custom_base_url(self) -> str:
        return self.llm.custom.base_url
    
    @property
    def custom_api_key(self) -> Optional[str]:
        return self.llm.custom.api_key
    
    @property
    def custom_model(self) -> str:
        return self.llm.custom.model
    
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
    def vector_store_api_key(self) -> Optional[str]:
        return self.rag.api_key
    
    @property
    def vector_store_namespace(self) -> Optional[str]:
        return self.rag.namespace
    
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
    
    # MySQL aliases
    @property
    def mysql_host(self) -> str:
        return self.mysql.host
    
    @property
    def mysql_port(self) -> int:
        return self.mysql.port
    
    @property
    def mysql_database(self) -> str:
        return self.mysql.database
    
    @property
    def mysql_user(self) -> str:
        return self.mysql.user
    
    @property
    def mysql_password(self) -> SecretStr:
        return self.mysql.password
    
    @property
    def mysql_pool_size(self) -> int:
        return self.mysql.pool_size
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="",
        env_nested_delimiter="__",
        extra="ignore",
    )


settings = Settings()

