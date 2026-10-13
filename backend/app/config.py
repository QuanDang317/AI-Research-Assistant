from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_NAME: str = "AI Research Assistant"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    GEMINI_API_KEY: str = ""

    DATABASE_URL: str = "postgresql://user:password123@localhost:5432/research_db"

    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION_NAME: str = "papers"

    UPLOAD_DIR: str = "./data/papers"

settings = Settings()
