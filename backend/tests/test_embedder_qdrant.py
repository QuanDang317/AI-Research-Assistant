import pytest
from app.services.embedder import Embedder
from app.db.qdrant import get_qdrant_client, init_qdrant_collection, upsert_chunks_to_qdrant
from app.config import settings

def test_embedder_dense_and_sparse():
    embedder = Embedder()
    sample_texts = [
        "Convolutional Neural Networks are widely used for image classification.",
        "Vision Transformers apply self-attention mechanisms to image patches."
    ]

    embeddings = embedder.embed_texts(sample_texts)
    assert len(embeddings) == 2

    # Check dense embedding
    assert len(embeddings[0]["dense"]) == settings.DENSE_EMBEDDING_DIM
    assert isinstance(embeddings[0]["dense"][0], float)

    # Check sparse embedding
    sparse_data = embeddings[0]["sparse"]
    assert "indices" in sparse_data
    assert "values" in sparse_data
    assert len(sparse_data["indices"]) > 0
    assert len(sparse_data["values"]) == len(sparse_data["indices"])

def test_qdrant_upsert_and_retrieve():
    client = get_qdrant_client()
    init_qdrant_collection(client)

    embedder = Embedder()
    test_chunk = [{
        "id": "c1111111-1111-1111-1111-111111111111",
        "paper_id": "p0000000-0000-0000-0000-000000000000",
        "paper_title": "Test Paper on AI",
        "section": "Methodology",
        "page_start": 3,
        "page_end": 4,
        "text": "We propose a novel residual learning framework to ease training of networks.",
        "token_count": 15
    }]

    embeddings = embedder.embed_texts([test_chunk[0]["text"]])
    point_ids = upsert_chunks_to_qdrant(test_chunk, embeddings, client=client)

    assert len(point_ids) == 1
    assert point_ids[0] == test_chunk[0]["id"]

    # Verify point exists in Qdrant
    points = client.retrieve(
        collection_name=settings.QDRANT_COLLECTION_NAME,
        ids=[point_ids[0]]
    )
    assert len(points) == 1
    assert points[0].payload["paper_title"] == "Test Paper on AI"
    assert points[0].payload["section"] == "Methodology"
