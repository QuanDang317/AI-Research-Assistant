from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class ChatFilters(BaseModel):
    year_min: Optional[int] = None
    year_max: Optional[int] = None
    fields: Optional[List[str]] = None
    authors: Optional[List[str]] = None


class ChatRequest(BaseModel):
    conversation_id: Optional[str] = None
    query: str
    filters: Optional[ChatFilters] = None
    paper_ids: Optional[List[str]] = None
    top_k: int = Field(default=5, ge=1, le=20)


class Citation(BaseModel):
    paper_id: str
    paper_title: str
    section: Optional[str] = None
    page: Optional[int] = None
    chunk_text: str
    relevance_score: Optional[float] = None


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    citations: List[Citation] = []
    metadata: Dict[str, Any] = {}
