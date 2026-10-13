import os
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.config import settings

logger = logging.getLogger(__name__)

Base = declarative_base()

def get_engine():
    try:
        # Try configured database (e.g. PostgreSQL)
        eng = create_engine(settings.DATABASE_URL, pool_pre_ping=True)
        # Test connection
        with eng.connect() as conn:
            pass
        return eng
    except Exception as e:
        logger.warning(
            f"Could not connect to PostgreSQL ({settings.DATABASE_URL}): {e}. "
            "Falling back to local SQLite at ./data/research.db"
        )
        os.makedirs("./data", exist_ok=True)
        sqlite_url = "sqlite:///./data/research.db"
        return create_engine(sqlite_url, connect_args={"check_same_thread": False})

engine = get_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
