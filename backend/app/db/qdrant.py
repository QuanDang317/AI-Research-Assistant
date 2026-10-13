from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, SparseVectorParams, Modifier
from app.config import settings
import logging

logger = logging.getLogger(__name__)

def get_qdrant_client() -> QdrantClient:
    """
    Returns an instance of QdrantClient configured from settings.
    """
    return QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)

def init_qdrant_collection(client: QdrantClient = None):
    """
    Initializes the 'papers' collection in Qdrant with dense and sparse vector configurations.
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
                        size=1024,
                        distance=Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        modifier=Modifier.IDF
                    )
                }
            )
            logger.info(f"Created Qdrant collection '{collection_name}' successfully.")
        else:
            logger.info(f"Qdrant collection '{collection_name}' already exists.")
    except Exception as e:
        logger.warning(f"Could not initialize Qdrant collection: {e}. Ensure Qdrant service is running.")
