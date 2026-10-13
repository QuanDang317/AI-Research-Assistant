from fastapi.testclient import TestClient
from app.main import app

def test_favicon_endpoint():
    client = TestClient(app)
    response = client.get("/favicon.ico")
    assert response.status_code == 200
    assert response.headers["content-type"] in ["image/x-icon", "image/vnd.microsoft.icon"]
    assert len(response.content) > 1000

def test_swagger_docs_favicon():
    client = TestClient(app)
    response = client.get("/docs")
    assert response.status_code == 200
    assert "/favicon.ico" in response.text

def test_redoc_favicon():
    client = TestClient(app)
    response = client.get("/redoc")
    assert response.status_code == 200
    assert "/favicon.ico" in response.text
