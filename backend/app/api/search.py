from fastapi import APIRouter
from app.schemas.chat import SearchRequest, SearchResponse, SearchResultChunk
from app.services.retriever import HybridRetriever
from app.services.query_processor import QueryProcessor

router = APIRouter(prefix="/api/v1/search", tags=["search"])

retriever = HybridRetriever()
query_processor = QueryProcessor(retriever=retriever)


@router.post("", response_model=SearchResponse)
def search_papers(request: SearchRequest):
    """
    Search indexed research papers using Hybrid Search (Dense + Sparse BM25),
    Reciprocal Rank Fusion (RRF), and Cross-Encoder Reranking.
    """
    filters_dict = {}
    if request.filters:
        if request.filters.year_min is not None:
            filters_dict["year_min"] = request.filters.year_min
        if request.filters.year_max is not None:
            filters_dict["year_max"] = request.filters.year_max
        if request.filters.fields:
            filters_dict["field"] = request.filters.fields[0]

    if request.paper_ids:
        filters_dict["paper_ids"] = request.paper_ids

    if request.use_multi_query:
        chunks = query_processor.multi_query_retrieve(
            query=request.query,
            filters=filters_dict if filters_dict else None,
            limit=request.limit
        )
    else:
        chunks = retriever.retrieve(
            query=request.query,
            filters=filters_dict if filters_dict else None,
            limit=request.limit,
            apply_rerank=True
        )

    results = []
    for c in chunks:
        results.append(
            SearchResultChunk(
                id=c["id"],
                paper_id=c.get("paper_id") or "",
                paper_title=c.get("paper_title") or "Unknown Paper",
                section=c.get("section"),
                page_start=c.get("page_start"),
                page_end=c.get("page_end"),
                text=c.get("text") or "",
                score=c.get("rerank_score") or c.get("rrf_score")
            )
        )

    return SearchResponse(
        query=request.query,
        total_results=len(results),
        results=results
    )
