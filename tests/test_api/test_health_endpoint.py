from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_get_health():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["message"] == "RAG API endpoint Active"