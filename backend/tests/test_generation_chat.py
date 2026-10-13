import io
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.db.session import init_db, SessionLocal
from app.models.paper import Paper
from app.utils.citation_parser import CitationParser
from app.services.generator import Generator
from tests.test_parser_chunker import create_sample_academic_pdf

@pytest.fixture(scope="module", autouse=True)
def prepare_environment():
    init_db()

def test_citation_parser():
    retrieved_chunks = [
        {
            "paper_id": "p-001",
            "paper_title": "Deep Residual Learning for Image Recognition",
            "section": "Architecture",
            "page_start": 3,
            "text": "Residual networks utilize shortcut connections.",
            "rerank_score": 0.95
        }
    ]
    sample_answer = "ResNet incorporates skip connections [Deep Residual Learning for Image Recognition, Architecture, Page 3]."
    citations = CitationParser.extract_citations(sample_answer, retrieved_chunks)

    assert len(citations) == 1
    assert citations[0]["paper_id"] == "p-001"
    assert citations[0]["page"] == 3
    assert citations[0]["section"] == "Architecture"

def test_generator_service():
    generator = Generator()
    chunks = [
        {
            "paper_id": "p-001",
            "paper_title": "Deep Residual Learning",
            "section": "Methodology",
            "page_start": 5,
            "text": "We employ residual blocks with batch normalization."
        }
    ]
    result = generator.generate_answer(
        query="What blocks does ResNet use?",
        retrieved_chunks=chunks
    )

    assert "answer" in result
    assert len(result["answer"]) > 10
    assert len(result["citations"]) > 0

def test_chat_api_and_multi_turn(tmp_path):
    pdf_path = str(tmp_path / "academic_test_paper.pdf")
    create_sample_academic_pdf(pdf_path)

    with TestClient(app) as client:
        # 1. Upload a paper first to have real data
        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()
        res_upload = client.post(
            "/api/v1/papers/upload",
            files={"file": ("academic_test_paper.pdf", io.BytesIO(pdf_bytes), "application/pdf")}
        )
        assert res_upload.status_code == 200
        paper_id = res_upload.json()["id"]

        # 2. First Chat Turn
        chat_req = {
            "query": "What is the accuracy of the proposed model?",
            "paper_ids": [paper_id],
            "top_k": 3
        }
        res_chat = client.post("/api/v1/chat", json=chat_req)
        assert res_chat.status_code == 200
        chat_data = res_chat.json()
        conv_id = chat_data["conversation_id"]
        assert len(chat_data["answer"]) > 0
        assert "citations" in chat_data

        # 3. Second Chat Turn (Multi-turn using same conversation_id)
        chat_req_2 = {
            "conversation_id": conv_id,
            "query": "What dataset was used in the experiments?",
            "paper_ids": [paper_id],
            "top_k": 3
        }
        res_chat_2 = client.post("/api/v1/chat", json=chat_req_2)
        assert res_chat_2.status_code == 200
        chat_data_2 = res_chat_2.json()
        assert chat_data_2["conversation_id"] == conv_id

        # 4. Check conversation history
        res_conv = client.get(f"/api/v1/conversations/{conv_id}")
        assert res_conv.status_code == 200
        conv_detail = res_conv.json()
        assert len(conv_detail["messages"]) >= 4  # 2 user msgs + 2 assistant msgs

def test_compare_api(tmp_path):
    with TestClient(app) as client:
        # Upload two papers
        p1 = str(tmp_path / "p1.pdf")
        p2 = str(tmp_path / "p2.pdf")
        create_sample_academic_pdf(p1)
        create_sample_academic_pdf(p2)

        with open(p1, "rb") as f:
            r1 = client.post("/api/v1/papers/upload", files={"file": ("p1.pdf", io.BytesIO(f.read()), "application/pdf")})
        with open(p2, "rb") as f:
            r2 = client.post("/api/v1/papers/upload", files={"file": ("p2.pdf", io.BytesIO(f.read()), "application/pdf")})

        paper_id1 = r1.json()["id"]
        paper_id2 = r2.json()["id"]

        compare_req = {
            "paper_ids": [paper_id1, paper_id2],
            "focus_aspects": ["method", "dataset", "metric", "result"]
        }
        res_compare = client.post("/api/v1/compare", json=compare_req)
        assert res_compare.status_code == 200
        comp_data = res_compare.json()
        assert "summary" in comp_data
        assert len(comp_data["table"]) >= 2
        assert comp_data["table"][0]["paper_id"] in [paper_id1, paper_id2]
