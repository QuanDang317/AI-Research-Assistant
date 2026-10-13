import io
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.db.session import init_db
from tests.test_parser_chunker import create_sample_academic_pdf

@pytest.fixture(autouse=True)
def setup_db():
    init_db()

def test_health_check():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

def test_upload_and_list_paper(tmp_path):
    pdf_path = str(tmp_path / "test_paper.pdf")
    create_sample_academic_pdf(pdf_path)

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    with TestClient(app) as client:
        # Upload
        files = {"file": ("test_paper.pdf", io.BytesIO(pdf_bytes), "application/pdf")}
        res_upload = client.post("/api/v1/papers/upload", files=files)
        assert res_upload.status_code == 200, res_upload.text
        paper_data = res_upload.json()
        paper_id = paper_data["id"]
        assert "Plant Disease" in paper_data["title"]

        # List
        res_list = client.get("/api/v1/papers")
        assert res_list.status_code == 200
        papers = res_list.json()
        assert any(p["id"] == paper_id for p in papers)

        # Chunks
        res_chunks = client.get(f"/api/v1/papers/{paper_id}/chunks")
        assert res_chunks.status_code == 200
        chunks = res_chunks.json()
        assert len(chunks) > 0
        assert chunks[0]["paper_id"] == paper_id

        # Delete
        res_del = client.delete(f"/api/v1/papers/{paper_id}")
        assert res_del.status_code == 200
