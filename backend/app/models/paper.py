import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, Text, DateTime, JSON
from app.db.session import Base


def utcnow():
    return datetime.now(timezone.utc)


class Paper(Base):
    __tablename__ = "papers"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(Text, nullable=False)
    authors = Column(JSON, default=list, nullable=False)
    year = Column(Integer, nullable=True)
    venue = Column(String(255), nullable=True)
    abstract = Column(Text, nullable=True)
    field = Column(String(100), nullable=True)
    file_path = Column(String(500), nullable=False)
    file_hash = Column(String(64), unique=True, index=True, nullable=True)
    total_pages = Column(Integer, default=0)
    status = Column(String(20), default="ready")
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)
