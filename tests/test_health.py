from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict():
    # Test with a sample text
    response = client.post("/predict", json={"text": "Apple announces new iPhone"})
    assert response.status_code == 200
    data = response.json()
    assert "label" in data
    assert "probabilities" in data
    assert "text" in data
    assert data["text"] == "Apple announces new iPhone"
    # Check probabilities sum to ~1
    probs = data["probabilities"]
    assert len(probs) == 4  # AG News has 4 classes
    assert abs(sum(probs.values()) - 1.0) < 0.1  # Allow some floating point tolerance