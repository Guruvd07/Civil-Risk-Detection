
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200


def test_status_endpoint():
    response = client.get("/status")
    assert response.status_code == 200


def test_clause_endpoint_returns_clause():
    response = client.get(
        "/clauses/FIDIC%20Red%20Book%202017/4.12"
    )

    assert response.status_code == 200
    assert response.json()


def test_analyze_endpoint_returns_valid_risk_analysis():
    payload = {
        "risk_description": (
            "The contractor discovered unforeseen underground "
            "rock formations during excavation, causing additional "
            "costs and project delays."
        ),
        "probability": 4,
        "impact": 4,
        "contract": "FIDIC Red Book 2017",
        "top_k": 5,
    }

    response = client.post("/analyze", json=payload)

    assert response.status_code == 200, response.text

    data = response.json()

    # Verify the main response structure.
    assert "risk" in data
    assert "classification" in data
    assert "claim" in data
    assert "actions" in data

    # Verify risk scoring.
    assert data["risk"]["probability"] == 4
    assert data["risk"]["impact"] == 4
    assert data["risk"]["score"] == 16

    # Verify that a primary clause was identified.
    assert data["primary_clause"] is not None
    assert data["primary_clause"]["clause_number"] == "4.12"

    # Verify evidence is returned.
    assert isinstance(data["evidence"], list)
    assert len(data["evidence"]) > 0
