from pydantic import BaseModel, ConfigDict
from typing import List, Optional
from datetime import datetime


class PaperBase(BaseModel):
    title: str
    authors: List[str] = []
    year: Optional[int] = None
    venue: Optional[str] = None
    abstract: Optional[str] = None
    field: Optional[str] = None


class PaperCreate(PaperBase):
    file_path: str
    file_hash: Optional[str] = None
    total_pages: int = 0


class PaperResponse(PaperBase):
    id: str
    file_path: str
    total_pages: int
    status: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ChunkResponse(BaseModel):
    id: str
    paper_id: str
    chunk_index: int
    section: Optional[str] = None
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    text: str
    token_count: int

    model_config = ConfigDict(from_attributes=True)
