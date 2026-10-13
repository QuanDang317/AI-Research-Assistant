from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_NAME: str = "AI Research Assistant"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    GEMINI_API_KEY: str = ""

    # Database
    DATABASE_URL: str = "postgresql://user:password123@localhost:5432/research_db"

    # Qdrant Vector Database
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION_NAME: str = "papers"

    # Embedding Models
    # Default to bge-small (384 dim) for fast CPU inference, configurable to bge-large (1024 dim)
    DENSE_EMBEDDING_MODEL: str = "BAAI/bge-small-en-v1.5"
    DENSE_EMBEDDING_DIM: int = 384
    SPARSE_EMBEDDING_MODEL: str = "Qdrant/bm25"

    # Reranker Model
    RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    UPLOAD_DIR: str = "./data/papers"

settings = Settings()
