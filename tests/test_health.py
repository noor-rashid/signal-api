from fastapi.testclient import TestClient

from signal_api.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_ready():
    response = client.get("/ready")
    assert response.status_code == 200
    # Ready depends on whether model is loaded
    data = response.json()
    assert "ready" in data


def test_signals_invalid_symbol():
    response = client.get("/signals/INVALID")
    assert response.status_code == 400


def test_predict_no_model():
    """If no model loaded, /predict should return 503 or work if model exists."""
    response = client.post("/predict", json={"symbol": "BTCUSDT"})
    # Either 503 (no model) or 200 (model loaded) — both are valid
    assert response.status_code in [200, 503]
