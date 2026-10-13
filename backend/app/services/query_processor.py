import re
import logging
from typing import List, Dict, Any, Optional
from app.config import settings
from app.services.retriever import HybridRetriever

logger = logging.getLogger(__name__)


class QueryProcessor:
    """
    Handles Query Expansion and Multi-Query Retrieval to maximize recall
    across varied scientific terminologies.
    """

    def __init__(self, retriever: Optional[HybridRetriever] = None):
        self.retriever = retriever or HybridRetriever()
        self._gemini_client = None

    @property
    def gemini_client(self):
        if self._gemini_client is None and settings.GEMINI_API_KEY:
            try:
                from google import genai
                self._gemini_client = genai.Client(api_key=settings.GEMINI_API_KEY)
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini Client for query expansion: {e}")
        return self._gemini_client

    def expand_query(self, query: str, num_queries: int = 3) -> List[str]:
        """
        Expands an academic query into alternative formulations.
        Falls back to rule-based variations if LLM is unavailable.
        """
        queries = [query]

        # 1. Try LLM expansion if Gemini API Key is available
        if self.gemini_client:
            try:
                prompt = (
                    f"You are a scientific research assistant. "
                    f"Given the user research question below, generate {num_queries} alternative search queries "
                    f"or sub-queries covering synonyms, technical terminology, and English translations if needed.\n"
                    f"Return ONLY the search queries, one per line. Do not include numbering or bullet points.\n\n"
                    f"Question: {query}"
                )
                response = self.gemini_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt
                )
                lines = [line.strip() for line in response.text.strip().split("\n") if line.strip()]
                for l in lines:
                    # Clean numbering like "1. ", "- "
                    cleaned = re.sub(r"^[\d\.\-\*\s]+", "", l).strip()
                    if cleaned and cleaned.lower() != query.lower() and cleaned not in queries:
                        queries.append(cleaned)
                return queries[:num_queries + 1]
            except Exception as e:
                logger.warning(f"LLM query expansion failed: {e}. Falling back to heuristic expansion.")

        # 2. Heuristic fallback expansion
        tokens = query.split()
        if len(tokens) > 2:
            # Add focused keyword variant
            queries.append(" ".join(tokens[:4]))
        return queries

    def multi_query_retrieve(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 5,
        candidates_k: int = 15
    ) -> List[Dict[str, Any]]:
        """
        Executes Multi-Query Retrieval:
        1. Expands original query into 3-4 distinct queries
        2. Retrieves candidate pools for each query without reranking
        3. Merges & deduplicates candidate chunks
        4. Re-ranks the combined candidates against the original query
        """
        expanded_queries = self.expand_query(query)
        logger.info(f"Expanded queries: {expanded_queries}")

        aggregated_candidates: Dict[str, Dict[str, Any]] = {}

        for q in expanded_queries:
            chunks = self.retriever.retrieve(
                query=q,
                filters=filters,
                limit=candidates_k,
                candidates_k=candidates_k,
                apply_rerank=False
            )
            for chunk in chunks:
                cid = chunk["id"]
                if cid not in aggregated_candidates:
                    aggregated_candidates[cid] = chunk
                else:
                    # Boost RRF score for chunks found by multiple query formulations
                    aggregated_candidates[cid]["rrf_score"] += chunk.get("rrf_score", 0.0)

        merged_list = list(aggregated_candidates.values())
        if not merged_list:
            return []

        # Sort by accumulated score
        merged_list.sort(key=lambda c: c.get("rrf_score", 0.0), reverse=True)

        # Final Cross-Encoder Rerank against original user query
        return self.retriever.reranker.rerank(
            query=query,
            candidates=merged_list[:candidates_k * 2],
            top_k=limit
        )
