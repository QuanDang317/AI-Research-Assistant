import logging
from typing import List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    Modifier,
    PointStruct,
    SparseVector,
    Filter,
    FieldCondition,
    MatchValue,
    Range
)
from app.config import settings

logger = logging.getLogger(__name__)

_qdrant_client_instance: Optional[QdrantClient] = None


def get_qdrant_client() -> QdrantClient:
    """
    Returns a shared QdrantClient instance.
    Attempts connection to remote Qdrant service first;
    falls back to in-memory Qdrant client if remote service is unavailable.
    """
    global _qdrant_client_instance
    if _qdrant_client_instance is not None:
        return _qdrant_client_instance

    try:
        # Try connecting to remote Qdrant (e.g. running via Docker)
        client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT, timeout=2.0)
        # Ping to check if alive
        client.get_collections()
        logger.info(f"Connected to Qdrant server at {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
        _qdrant_client_instance = client
        return _qdrant_client_instance
    except Exception as e:
        logger.warning(
            f"Could not connect to Qdrant at {settings.QDRANT_HOST}:{settings.QDRANT_PORT}: {e}. "
            "Falling back to local in-memory Qdrant client for local operations."
        )
        _qdrant_client_instance = QdrantClient(location=":memory:")
        return _qdrant_client_instance


def init_qdrant_collection(client: Optional[QdrantClient] = None):
    """
    Ensures the 'papers' collection exists with both Dense and Sparse vector configurations.
    """
    if client is None:
        client = get_qdrant_client()

    collection_name = settings.QDRANT_COLLECTION_NAME

    try:
        collections = client.get_collections().collections
        exists = any(c.name == collection_name for c in collections)

        if not exists:
            client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "dense": VectorParams(
                        size=settings.DENSE_EMBEDDING_DIM,
                        distance=Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        modifier=Modifier.IDF
                    )
                }
            )
            logger.info(
                f"Initialized Qdrant collection '{collection_name}' with Dense ({settings.DENSE_EMBEDDING_DIM} dim) "
                "and Sparse vector configurations."
            )
        else:
            logger.info(f"Qdrant collection '{collection_name}' already exists.")
    except Exception as e:
        logger.error(f"Error initializing Qdrant collection: {e}")


def upsert_chunks_to_qdrant(
    chunks: List[Dict[str, Any]],
    embeddings: List[Dict[str, Any]],
    client: Optional[QdrantClient] = None
) -> List[str]:
    """
    Upserts chunks along with their dense and sparse vector representations into Qdrant.
    Returns list of point IDs upserted.
    """
    if client is None:
        client = get_qdrant_client()

    init_qdrant_collection(client)
    collection_name = settings.QDRANT_COLLECTION_NAME

    points = []
    point_ids = []

    for chunk, emb in zip(chunks, embeddings):
        point_id = chunk.get("qdrant_id") or chunk.get("id")
        point_ids.append(point_id)

        sparse_data = emb.get("sparse", {})
        sparse_vec = SparseVector(
            indices=sparse_data.get("indices", []),
            values=sparse_data.get("values", [])
        )

        payload = {
            "chunk_id": point_id,
            "paper_id": chunk.get("paper_id"),
            "paper_title": chunk.get("paper_title"),
            "authors": chunk.get("authors", []),
            "year": chunk.get("year"),
            "section": chunk.get("section"),
            "page_start": chunk.get("page_start"),
            "page_end": chunk.get("page_end"),
            "text": chunk.get("text"),
            "token_count": chunk.get("token_count", 0),
        }

        point = PointStruct(
            id=point_id,
            vector={
                "dense": emb["dense"],
                "sparse": sparse_vec
            },
            payload=payload
        )
        points.append(point)

    if points:
        client.upsert(collection_name=collection_name, points=points)
        logger.info(f"Upserted {len(points)} chunks into Qdrant collection '{collection_name}'.")

    return point_ids
