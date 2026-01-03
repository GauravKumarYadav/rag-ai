from typing import List, Optional

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # LLM Provider Configuration
    # Options: "lmstudio", "ollama", "openai", "custom"
    llm_provider: str = "lmstudio"
    
    # LMStudio settings (default)
    lmstudio_base_url: str = "http://localhost:1234/v1"
    lmstudio_model: str = "Qwen3-VL-30B-Instruct"
    
    # Ollama settings
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    
    # OpenAI settings
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"
    
    # Custom endpoint settings
    custom_base_url: str = "http://localhost:8080/v1"
    custom_api_key: Optional[str] = None
    custom_model: str = "default"
    
    # Common LLM settings
    llm_temperature: float = 0.25
    llm_max_tokens: int = 1024
    llm_timeout: float = 60.0
    context_window: int = 16000

    # Legacy aliases (for backwards compatibility)
    @property
    def lmstudio_temperature(self) -> float:
        return self.llm_temperature
    
    @property
    def lmstudio_max_tokens(self) -> int:
        return self.llm_max_tokens

    # RAG settings
    chroma_db_path: str = "./data/chroma"
    embedding_model: str = "text-embedding-nomic-embed-text-v1.5"
    
    # Vector Store Configuration (modular - can switch providers)
    # Options: "chromadb", "pinecone", "weaviate", "qdrant", "milvus"
    vector_store_provider: str = "chromadb"
    vector_store_url: Optional[str] = None  # For cloud/remote stores
    vector_store_api_key: Optional[str] = None
    vector_store_namespace: Optional[str] = None  # Index/namespace name
    vector_store_prefix: str = ""  # Collection name prefix

    # Session settings
    session_max_tokens: int = 4000
    session_max_messages: int = 20

    # Server settings
    backend_host: str = "0.0.0.0"
    backend_port: int = 8000
    backend_cors_origins: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ]

    # JWT Authentication settings
    jwt_secret_key: SecretStr = SecretStr("change-me-in-production-use-strong-secret")
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30

    # MySQL Audit Logging settings
    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_database: str = "audit_logs"
    mysql_user: str = "root"
    mysql_password: SecretStr = SecretStr("Sarita1!@2024_4")
    mysql_pool_size: int = 5

    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")


settings = Settings()

