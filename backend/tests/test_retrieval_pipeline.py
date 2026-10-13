import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.db.qdrant import get_qdrant_client, init_qdrant_collection, upsert_chunks_to_qdrant
from app.services.embedder import Embedder
from app.services.retriever import HybridRetriever
from app.services.query_processor import QueryProcessor

@pytest.fixture(scope="module")
def setup_search_index():
    client = get_qdrant_client()
    init_qdrant_collection(client)
    embedder = Embedder()

    chunks = [
        {
            "id": "e1111111-1111-1111-1111-111111111111",
            "paper_id": "p1111111-1111-1111-1111-111111111111",
            "paper_title": "Deep Residual Learning",
            "year": 2016,
            "section": "Architecture",
            "page_start": 3,
            "page_end": 4,
            "text": "ResNet utilizes identity shortcut connections to solve vanishing gradients in very deep networks.",
            "token_count": 20
        },
        {
            "id": "e2222222-2222-2222-2222-222222222222",
            "paper_id": "p2222222-2222-2222-2222-222222222222",
            "paper_title": "An Image is Worth 16x16 Words",
            "year": 2020,
            "section": "Method",
            "page_start": 2,
            "page_end": 3,
            "text": "Vision Transformers split an image into fixed-size patches and linearly embed each of them.",
            "token_count": 20
        },
        {
            "id": "e3333333-3333-3333-3333-333333333333",
            "paper_id": "p3333333-3333-3333-3333-333333333333",
            "paper_title": "Support Vector Machines Overview",
            "year": 1995,
            "section": "Theory",
            "page_start": 1,
            "page_end": 2,
            "text": "SVM computes maximum-margin hyperplanes to separate classes in feature space.",
            "token_count": 18
        }
    ]

    embeddings = embedder.embed_texts([c["text"] for c in chunks])
    upsert_chunks_to_qdrant(chunks, embeddings, client=client)
    return client

def test_hybrid_retrieval_and_rerank(setup_search_index):
    retriever = HybridRetriever(client=setup_search_index)

    query = "How does ResNet solve vanishing gradients?"
    results = retriever.retrieve(query=query, limit=2, apply_rerank=True)

    assert len(results) > 0
    # Top result should be a ResNet chunk
    assert "ResNet" in results[0]["text"] or "Residual" in results[0]["payload"].get("paper_title", "")

def test_metadata_filtering(setup_search_index):
    retriever = HybridRetriever(client=setup_search_index)

    # Search with year filter (year >= 2019)
    query = "deep learning methods"
    results = retriever.retrieve(
        query=query,
        filters={"year_min": 2019},
        limit=5,
        apply_rerank=False
    )

    assert len(results) > 0
    # Only Vision Transformers (year 2020) should pass the filter
    for r in results:
        assert r["year"] >= 2019

def test_search_api_endpoint(setup_search_index):
    with TestClient(app) as client:
        payload = {
            "query": "Vision transformer image patches",
            "limit": 2,
            "use_multi_query": False
        }
        response = client.post("/api/v1/search", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["total_results"] > 0
        assert "Vision Transformers" in data["results"][0]["text"]
