"""Tests for FastAPI backend REST API in api/main.py and api/schemas.py.

Verifies endpoints /health and /analyze, validation behavior, error responses,
lifespan model loading, CORS configuration, order preservation, and latency.
"""

from __future__ import annotations

import time
from typing import Generator
import pytest
from fastapi import status
from fastapi.testclient import TestClient

from api.main import app
from ml.config import MODEL_VERSION


@pytest.fixture(scope="module")
def client() -> Generator[TestClient, None, None]:
    """Provide a FastAPI TestClient with the application lifespan entered."""
    with TestClient(app) as test_client:
        yield test_client


def test_health_ok(client: TestClient) -> None:
    """Verify GET /health returns 200 OK with status 'ok' and model_version."""
    response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_version"] == MODEL_VERSION


def test_valid_analyze_response_schema(client: TestClient) -> None:
    """Verify POST /analyze returns 200 OK and conforms to PRD Section 9 schema."""
    payload = {
        "page_url": "https://example.com/product/test-item",
        "reviews": [
            {
                "id": "r1",
                "text": "The battery life is decent, lasts about 14 hours on a single charge. Sound clarity is good.",
                "rating": 4,
                "date": "2026-09-01",
            },
            {
                "id": "r2",
                "text": "AMAZING PRODUCT!! BEST PURCHASE EVER!! 100% RECOMMEND TO EVERYONE DO NOT HESITATE BUY IT NOW WOW WOW WOW!!!",
                "rating": 5,
                "date": "2026-10-01",
            },
        ],
    }

    response = client.post("/analyze", json=payload)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()

    # Top-level keys
    assert "results" in data
    assert "summary" in data
    assert "processing_ms" in data
    assert isinstance(data["processing_ms"], (int, float))
    assert data["processing_ms"] >= 0.0

    # Results validation
    results = data["results"]
    assert len(results) == 2

    for r in results:
        assert isinstance(r["id"], str)
        assert r["label"] in {"suspicious", "genuine", "uncertain"}
        assert isinstance(r["confidence"], float)
        assert 0.0 <= r["confidence"] <= 1.0
        assert isinstance(r["reasons"], list)
        for reason in r["reasons"]:
            assert "type" in reason
            assert "text" in reason
            assert isinstance(reason["type"], str)
            assert isinstance(reason["text"], str)

    # Summary validation
    summary = data["summary"]
    assert summary["total"] == 2
    assert isinstance(summary["suspicious_count"], int)
    assert 0 <= summary["suspicious_count"] <= 2
    assert isinstance(summary["suspicious_pct"], float)
    assert 0.0 <= summary["suspicious_pct"] <= 100.0
    assert isinstance(summary["reliability_score"], int)
    assert 0 <= summary["reliability_score"] <= 100
    assert isinstance(summary["reliability_label"], str)
    assert isinstance(summary["insights"], list)


def test_empty_list_rejected(client: TestClient) -> None:
    """Verify POST /analyze with an empty reviews list returns 422 Unprocessable Entity."""
    response = client.post("/analyze", json={"reviews": []})
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_text_too_long_handled(client: TestClient) -> None:
    """Verify review text exceeding 5,000 characters is rejected with 422."""
    long_text = "x" * 5001
    payload = {
        "reviews": [
            {"id": "long_rev", "text": long_text}
        ]
    }
    response = client.post("/analyze", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data


def test_empty_text_rejected(client: TestClient) -> None:
    """Verify empty or purely whitespace review text returns 422."""
    # Empty string
    response = client.post("/analyze", json={"reviews": [{"text": ""}]})
    assert response.status_code == 422

    # Whitespace only
    response = client.post("/analyze", json={"reviews": [{"text": "   \n\t "}]})
    assert response.status_code == 422


def test_missing_optional_fields_accepted(client: TestClient) -> None:
    """Verify omitting optional fields (page_url, id, rating, date) is accepted and generates id."""
    payload = {
        "reviews": [
            {"text": "Simple authentic review with sufficient detail and good build."}
        ]
    }
    response = client.post("/analyze", json=payload)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()

    assert len(data["results"]) == 1
    # Auto-generated ID must be present
    assert data["results"][0]["id"] == "r1"
    assert data["results"][0]["label"] in {"suspicious", "genuine", "uncertain"}


def test_order_preserved(client: TestClient) -> None:
    """Verify order of reviews in results strictly matches the input sequence."""
    payload = {
        "reviews": [
            {"id": "order_3", "text": "Third review text with concrete details."},
            {"id": "order_1", "text": "First review text with concrete details."},
            {"id": "order_2", "text": "Second review text with concrete details."},
        ]
    }
    response = client.post("/analyze", json=payload)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    output_ids = [r["id"] for r in data["results"]]
    assert output_ids == ["order_3", "order_1", "order_2"]


@pytest.mark.slow
def test_latency_50_reviews_under_2_seconds(client: TestClient) -> None:
    """Verify analyzing a batch of 50 reviews finishes within 2 seconds."""
    reviews = [
        {
            "id": f"r{i}",
            "text": f"Review number {i} notes that the device provides {i % 12 + 2} hours of operation with durable casing.",
            "rating": (i % 5) + 1,
            "date": "2026-09-15",
        }
        for i in range(50)
    ]
    payload = {"reviews": reviews}

    start_time = time.perf_counter()
    response = client.post("/analyze", json=payload)
    elapsed_wall_sec = time.perf_counter() - start_time

    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert len(data["results"]) == 50
    assert data["summary"]["total"] == 50
    # Both wall clock and server-reported processing time must be under 2 seconds
    assert elapsed_wall_sec < 2.0
    assert data["processing_ms"] < 2000.0


def test_max_reviews_limit_rejected(client: TestClient) -> None:
    """Verify batches exceeding 200 reviews are rejected with 422."""
    too_many = [
        {"id": f"r{i}", "text": f"Valid review text number {i}."}
        for i in range(201)
    ]
    response = client.post("/analyze", json={"reviews": too_many})
    assert response.status_code == 422


def test_invalid_rating_rejected(client: TestClient) -> None:
    """Verify ratings outside 1-5 range return 422."""
    response_high = client.post("/analyze", json={"reviews": [{"text": "Good", "rating": 6}]})
    assert response_high.status_code == 422

    response_low = client.post("/analyze", json={"reviews": [{"text": "Good", "rating": 0}]})
    assert response_low.status_code == 422


def test_invalid_date_rejected(client: TestClient) -> None:
    """Verify non-ISO date string returns 422."""
    payload = {
        "reviews": [{"text": "Good", "date": "yesterday"}]
    }
    response = client.post("/analyze", json=payload)
    assert response.status_code == 422


def test_cors_headers_chrome_extension(client: TestClient) -> None:
    """Verify CORS headers allow chrome-extension:// origins."""
    headers = {
        "Origin": "chrome-extension://abcdefghijklmnopqrstuvwxyzabcdef",
    }
    response = client.get("/health", headers=headers)
    assert response.status_code == status.HTTP_200_OK
    assert response.headers.get("access-control-allow-origin") == "chrome-extension://abcdefghijklmnopqrstuvwxyzabcdef"


def test_cors_headers_localhost(client: TestClient) -> None:
    """Verify CORS headers allow localhost origin."""
    headers = {
        "Origin": "http://localhost",
    }
    response = client.get("/health", headers=headers)
    assert response.status_code == status.HTTP_200_OK
    assert response.headers.get("access-control-allow-origin") == "http://localhost"


def test_missing_model_returns_503(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """Verify 503 Service Unavailable with helpful 'make train' message if model is missing."""
    import api.main as main_module
    from pathlib import Path

    # Monkeypatch MODEL_PATH to a nonexistent file and clear state
    fake_path = Path("non_existent_model_file.joblib")
    monkeypatch.setattr(main_module, "MODEL_PATH", fake_path)
    old_service = getattr(app.state, "model_service", None)
    app.state.model_service = None

    try:
        response = client.post("/analyze", json={"reviews": [{"text": "Sample review text."}]})
        assert response.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
        data = response.json()
        assert "make train" in data["detail"].lower()
    finally:
        app.state.model_service = old_service
