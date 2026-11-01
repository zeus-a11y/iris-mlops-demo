import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.train import train


@pytest.fixture(scope="module", autouse=True)
def ensure_model():
    train()


client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_predict():
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }
    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["predicted_class"] in {"setosa", "versicolor", "virginica"}
    assert abs(sum(body["probabilities"].values()) - 1.0) < 1e-6


def test_predict_rejects_invalid_input():
    resp = client.post("/predict", json={"sepal_length": -1, "sepal_width": 3.5,
                                          "petal_length": 1.4, "petal_width": 0.2})
    assert resp.status_code == 422


def test_metrics_endpoint_exposed():
    resp = client.get("/metrics")
    assert resp.status_code == 200
