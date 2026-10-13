from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class CompareRequest(BaseModel):
    paper_ids: List[str]
    focus_aspects: Optional[List[str]] = ["method", "dataset", "metrics", "results"]


class ComparisonRow(BaseModel):
    paper_id: str
    paper_title: str
    method: str
    dataset: str
    metric: str
    result: str
    citations: List[str] = []


class CompareResponse(BaseModel):
    summary: str
    table: List[ComparisonRow]
