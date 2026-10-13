import logging
from typing import List, Dict, Any, Optional
from collections import defaultdict
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    Range,
    SparseVector
)
from app.config import settings
from app.db.qdrant import get_qdrant_client
from app.services.embedder import Embedder
from app.services.reranker import Reranker

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Hybrid Retriever combining Dense Semantic Search and Sparse BM25 Search
    using Reciprocal Rank Fusion (RRF) and Cross-Encoder Reranking.
    """

    def __init__(
        self,
        client: Optional[QdrantClient] = None,
        embedder: Optional[Embedder] = None,
        reranker: Optional[Reranker] = None,
        rrf_k: int = 60
    ):
        self.client = client or get_qdrant_client()
        self.embedder = embedder or Embedder()
        self.reranker = reranker or Reranker()
        self.rrf_k = rrf_k

    def build_filter(self, filter_params: Optional[Dict[str, Any]]) -> Optional[Filter]:
        """
        Builds Qdrant Filter object from dictionary criteria.
        """
        if not filter_params:
            return None

        conditions = []

        # Filter by specific paper IDs
        if filter_params.get("paper_ids"):
            paper_ids = filter_params["paper_ids"]
            if isinstance(paper_ids, list):
                if len(paper_ids) == 1:
                    conditions.append(
                        FieldCondition(key="paper_id", match=MatchValue(value=paper_ids[0]))
                    )
                elif len(paper_ids) > 1:
                    conditions.append(
                        FieldCondition(key="paper_id", match=MatchAny(any=paper_ids))
                    )
            elif isinstance(paper_ids, str):
                conditions.append(
                    FieldCondition(key="paper_id", match=MatchValue(value=paper_ids))
                )

        # Filter by Publication Year
        if filter_params.get("year_min") is not None or filter_params.get("year_max") is not None:
            range_args = {}
            if filter_params.get("year_min") is not None:
                range_args["gte"] = filter_params["year_min"]
            if filter_params.get("year_max") is not None:
                range_args["lte"] = filter_params["year_max"]
            conditions.append(
                FieldCondition(key="year", range=Range(**range_args))
            )

        # Filter by Research Field
        if filter_params.get("field"):
            conditions.append(
                FieldCondition(key="field", match=MatchValue(value=filter_params["field"]))
            )

        return Filter(must=conditions) if conditions else None

    def retrieve(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 5,
        candidates_k: int = 20,
        apply_rerank: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Executes full Hybrid Retrieval Pipeline:
        1. Query Embedding (Dense + Sparse)
        2. Qdrant Dense Search (candidates_k)
        3. Qdrant Sparse BM25 Search (candidates_k)
        4. Reciprocal Rank Fusion (RRF) to merge candidate lists
        5. Cross-Encoder Reranking to select Top-K most relevant chunks
        """
        collection_name = settings.QDRANT_COLLECTION_NAME
        query_filter = self.build_filter(filters)

        # 1. Embed query
        embedded = self.embedder.embed_query(query)
        dense_vec = embedded.get("dense", [])
        sparse_data = embedded.get("sparse", {})

        dense_points = []
        sparse_points = []

        # 2. Dense Search
        if dense_vec:
            try:
                res = self.client.query_points(
                    collection_name=collection_name,
                    query=dense_vec,
                    using="dense",
                    query_filter=query_filter,
                    limit=candidates_k
                )
                dense_points = getattr(res, "points", [])
            except Exception as e:
                logger.warning(f"Dense search encountered error: {e}")

        # 3. Sparse Search
        if sparse_data.get("indices") and sparse_data.get("values"):
            try:
                res = self.client.query_points(
                    collection_name=collection_name,
                    query=SparseVector(
                        indices=sparse_data["indices"],
                        values=sparse_data["values"]
                    ),
                    using="sparse",
                    query_filter=query_filter,
                    limit=candidates_k
                )
                sparse_points = getattr(res, "points", [])
            except Exception as e:
                logger.warning(f"Sparse search encountered error: {e}")

        # 4. Reciprocal Rank Fusion (RRF)
        rrf_scores = defaultdict(float)
        points_map = {}

        for rank, point in enumerate(dense_points, start=1):
            rrf_scores[point.id] += 1.0 / (self.rrf_k + rank)
            points_map[point.id] = point

        for rank, point in enumerate(sparse_points, start=1):
            rrf_scores[point.id] += 1.0 / (self.rrf_k + rank)
            points_map[point.id] = point

        # Rank all unique candidates by RRF score descending
        sorted_ids = sorted(rrf_scores.keys(), key=lambda pid: rrf_scores[pid], reverse=True)

        candidates = []
        for pid in sorted_ids:
            pt = points_map[pid]
            payload = pt.payload or {}
            candidates.append({
                "id": str(pt.id),
                "paper_id": payload.get("paper_id"),
                "paper_title": payload.get("paper_title"),
                "authors": payload.get("authors", []),
                "year": payload.get("year"),
                "section": payload.get("section"),
                "page_start": payload.get("page_start"),
                "page_end": payload.get("page_end"),
                "text": payload.get("text", ""),
                "rrf_score": rrf_scores[pid]
            })

        # 5. Cross-Encoder Rerank
        if apply_rerank and candidates:
            top_candidates = candidates[:candidates_k]
            return self.reranker.rerank(query=query, candidates=top_candidates, top_k=limit)

        return candidates[:limit]
