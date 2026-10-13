import logging
from typing import List, Dict, Any, Optional
from sentence_transformers import CrossEncoder
from app.config import settings

logger = logging.getLogger(__name__)


class Reranker:
    """
    Cross-Encoder Reranker to accurately score and re-order (query, document) pairs.
    Filters top candidates (e.g. 20) down to the most relevant top-K (e.g. 5).
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Reranker, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_name: Optional[str] = None):
        if getattr(self, "_initialized", False):
            return

        self.model_name = model_name or getattr(
            settings, "RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        self._model = None
        self._initialized = True

    @property
    def model(self) -> CrossEncoder:
        if self._model is None:
            logger.info(f"Loading CrossEncoder model: {self.model_name}")
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Reranks a list of candidate chunk dicts against a query.
        Each candidate dict must contain a 'text' field.
        Returns the top_k candidate dicts sorted by rerank relevance score.
        """
        if not candidates:
            return []

        if len(candidates) <= top_k:
            return candidates

        try:
            pairs = [[query, c.get("text", "")] for c in candidates]
            scores = self.model.predict(pairs)

            ranked_results = []
            for score, candidate in zip(scores, candidates):
                cand_copy = dict(candidate)
                cand_copy["rerank_score"] = float(score)
                ranked_results.append(cand_copy)

            ranked_results.sort(key=lambda x: x["rerank_score"], reverse=True)
            return ranked_results[:top_k]
        except Exception as e:
            logger.warning(f"Reranker failed with error: {e}. Falling back to original candidate order.")
            return candidates[:top_k]
