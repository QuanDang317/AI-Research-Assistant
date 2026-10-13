import logging
from typing import List, Dict, Any, Optional
from fastembed import TextEmbedding, SparseTextEmbedding
from app.config import settings

logger = logging.getLogger(__name__)


class Embedder:
    """
    Production-ready Hybrid Embedder generating Dense and Sparse vectors.
    Uses ONNX-accelerated BGE and BM25 models for high-performance retrieval.
    """

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(Embedder, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        dense_model_name: Optional[str] = None,
        sparse_model_name: Optional[str] = None
    ):
        if getattr(self, "_initialized", False):
            return

        self.dense_model_name = dense_model_name or settings.DENSE_EMBEDDING_MODEL
        self.sparse_model_name = sparse_model_name or settings.SPARSE_EMBEDDING_MODEL
        self._dense_model = None
        self._sparse_model = None
        self._initialized = True

    @property
    def dense_model(self) -> TextEmbedding:
        if self._dense_model is None:
            logger.info(f"Loading dense embedding model: {self.dense_model_name}")
            self._dense_model = TextEmbedding(model_name=self.dense_model_name)
        return self._dense_model

    @property
    def sparse_model(self) -> SparseTextEmbedding:
        if self._sparse_model is None:
            logger.info(f"Loading sparse embedding model: {self.sparse_model_name}")
            self._sparse_model = SparseTextEmbedding(model_name=self.sparse_model_name)
        return self._sparse_model

    def embed_texts(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Embeds a list of texts into both dense and sparse representations.
        Returns:
            [
                {
                    "dense": list of floats,
                    "sparse": {"indices": list of ints, "values": list of floats}
                },
                ...
            ]
        """
        if not texts:
            return []

        # Generate dense embeddings
        dense_raw = list(self.dense_model.embed(texts))
        dense_vectors = [v.tolist() for v in dense_raw]

        # Generate sparse embeddings
        sparse_raw = list(self.sparse_model.embed(texts))
        sparse_vectors = [
            {
                "indices": s.indices.tolist(),
                "values": s.values.tolist()
            }
            for s in sparse_raw
        ]

        results = []
        for d, s in zip(dense_vectors, sparse_vectors):
            results.append({
                "dense": d,
                "sparse": s
            })

        return results

    def embed_query(self, query: str) -> Dict[str, Any]:
        """
        Embeds a single query string for hybrid retrieval.
        """
        results = self.embed_texts([query])
        if results:
            return results[0]
        return {
            "dense": [],
            "sparse": {"indices": [], "values": []}
        }
