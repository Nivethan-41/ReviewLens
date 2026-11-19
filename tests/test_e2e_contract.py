"""End-to-end API contract tests verifying all data consumed by the Chrome extension popup.

Tests the full pipeline from sample_reviews.html extraction to /analyze and /health endpoints,
validating all response field names, types, value constraints, sorting, and explainability payloads.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List
from bs4 import BeautifulSoup
from fastapi import status
from fastapi.testclient import TestClient
import pytest

from api.main import app
from ml.config import MODEL_VERSION, PROJECT_ROOT


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Provide a FastAPI TestClient configured with application lifespan."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def sample_reviews_html_path() -> Path:
    """Return path to sample_reviews.html."""
    sample_path = PROJECT_ROOT / "extension" / "test" / "sample_reviews.html"
    assert sample_path.exists(), f"Missing sample reviews file at {sample_path}"
    return sample_path


@pytest.fixture(scope="module")
def extracted_sample_reviews(sample_reviews_html_path: Path) -> List[Dict[str, Any]]:
    """Simulate extension content script extraction on sample_reviews.html."""
    html_content = sample_reviews_html_path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html_content, "html.parser")

    reviews: List[Dict[str, Any]] = []
    seen = set()

    for idx, el in enumerate(soup.find_all(attrs={"itemprop": "reviewBody"}), start=1):
        text = " ".join(el.get_text().split())
        container = el.find_parent(attrs={"itemscope": True}) or el.parent

        rating_val = None
        rating_el = container.find(attrs={"itemprop": "ratingValue"}) if container else None
        if rating_el:
            try:
                raw_r = rating_el.get("content") or rating_el.get_text()
                rating_val = float(raw_r.strip())
            except (ValueError, AttributeError):
                pass

        date_str = None
        date_el = container.find(attrs={"itemprop": "datePublished"}) if container else None
        if date_el:
            date_str = date_el.get("datetime") or date_el.get_text().strip()

        key = text.lower()
        if key not in seen:
            seen.add(key)
            reviews.append({
                "id": f"rev-{idx}",
                "text": text,
                "rating": rating_val,
                "date": date_str,
            })

    return reviews


def test_extracted_sample_reviews_count(extracted_sample_reviews: List[Dict[str, Any]]) -> None:
    """Verify that sample_reviews.html yields exactly 12 structured reviews."""
    assert len(extracted_sample_reviews) == 12
    for r in extracted_sample_reviews:
        assert isinstance(r["id"], str)
        assert isinstance(r["text"], str) and len(r["text"]) >= 10
        assert r["rating"] is not None and 1.0 <= r["rating"] <= 5.0
        assert r["date"] is not None and len(r["date"]) == 10


def test_popup_e2e_contract_analyze_response(
    client: TestClient,
    extracted_sample_reviews: List[Dict[str, Any]],
) -> None:
    """Verify POST /analyze returns a payload satisfying all popup UI requirements.

    Validates:
    1. Summary fields: reliability_score (0-100), reliability_label, total,
       suspicious_count, suspicious_pct, and insights list.
    2. Results fields: id, label ('suspicious'|'genuine'|'uncertain'),
       confidence (0.0-1.0), reasons with type and text.
    """
    payload = {
        "page_url": "file:///extension/test/sample_reviews.html",
        "reviews": extracted_sample_reviews,
    }

    response = client.post("/analyze", json=payload)
    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert "summary" in data
    assert "results" in data

    # 1. Summary Card Contract Checks
    summary = data["summary"]
    assert "reliability_score" in summary
    assert isinstance(summary["reliability_score"], int)
    assert 0 <= summary["reliability_score"] <= 100

    assert "reliability_label" in summary
    assert isinstance(summary["reliability_label"], str)
    assert summary["reliability_label"] in {"High", "Moderate", "Low", "Not enough data"}

    assert "total" in summary
    assert isinstance(summary["total"], int)
    assert summary["total"] == 12

    assert "suspicious_count" in summary
    assert isinstance(summary["suspicious_count"], int)
    assert 0 <= summary["suspicious_count"] <= 12

    assert "suspicious_pct" in summary
    assert isinstance(summary["suspicious_pct"], (int, float))
    assert 0.0 <= summary["suspicious_pct"] <= 100.0

    assert "insights" in summary
    assert isinstance(summary["insights"], list)
    for insight in summary["insights"]:
        assert isinstance(insight, str)
        assert len(insight.strip()) > 0

    # 2. Results List Contract Checks
    results = data["results"]
    assert isinstance(results, list)
    assert len(results) == 12

    for item in results:
        assert "id" in item
        assert isinstance(item["id"], str)

        assert "label" in item
        assert item["label"] in {"suspicious", "genuine", "uncertain"}

        assert "confidence" in item
        assert isinstance(item["confidence"], float)
        assert 0.0 <= item["confidence"] <= 1.0
        # Check popup confidence percent calculation
        conf_pct = round(item["confidence"] * 100)
        assert 0 <= conf_pct <= 100

        assert "reasons" in item
        assert isinstance(item["reasons"], list)
        for rsn in item["reasons"]:
            assert "type" in rsn
            assert rsn["type"] in {"language", "style", "pattern"}
            assert "text" in rsn
            assert isinstance(rsn["text"], str)
            assert len(rsn["text"].strip()) > 0


def test_popup_sorting_contract(
    client: TestClient,
    extracted_sample_reviews: List[Dict[str, Any]],
) -> None:
    """Verify popup sorting contract: suspicious reviews first, sorted by confidence descending."""
    payload = {
        "page_url": "file:///extension/test/sample_reviews.html",
        "reviews": extracted_sample_reviews,
    }
    response = client.post("/analyze", json=payload)
    assert response.status_code == status.HTTP_200_OK

    results = response.json()["results"]

    # Replicate popup sorting logic
    sorted_results = sorted(
        results,
        key=lambda r: (1 if r["label"] == "suspicious" else 0, r["confidence"]),
        reverse=True,
    )

    # All suspicious reviews should appear before genuine/uncertain
    found_non_suspicious = False
    for r in sorted_results:
        if r["label"] != "suspicious":
            found_non_suspicious = True
        elif found_non_suspicious:
            pytest.fail("Found suspicious review after non-suspicious review in sorted list")

    # Within suspicious group, confidences must be non-increasing
    suspicious_confs = [r["confidence"] for r in sorted_results if r["label"] == "suspicious"]
    for i in range(len(suspicious_confs) - 1):
        assert suspicious_confs[i] >= suspicious_confs[i + 1]


def test_popup_filter_contract(
    client: TestClient,
    extracted_sample_reviews: List[Dict[str, Any]],
) -> None:
    """Verify popup filter chips contract: All, Suspicious, Genuine-looking."""
    payload = {
        "page_url": "file:///extension/test/sample_reviews.html",
        "reviews": extracted_sample_reviews,
    }
    response = client.post("/analyze", json=payload)
    assert response.status_code == status.HTTP_200_OK

    results = response.json()["results"]

    all_filter = [r for r in results]
    suspicious_filter = [r for r in results if r["label"] == "suspicious"]
    genuine_filter = [r for r in results if r["label"] == "genuine"]

    assert len(all_filter) == 12
    assert len(suspicious_filter) + len(genuine_filter) <= 12
    for r in suspicious_filter:
        assert r["label"] == "suspicious"
    for r in genuine_filter:
        assert r["label"] == "genuine"


def test_popup_footer_health_contract(client: TestClient) -> None:
    """Verify popup footer contract: GET /health returns 'ok' and model_version."""
    response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert data.get("status") == "ok"
    assert "model_version" in data
    assert data["model_version"] == MODEL_VERSION
    assert len(data["model_version"].strip()) > 0


def test_empty_reviews_contract_error_state(client: TestClient) -> None:
    """Verify POST /analyze rejects empty reviews list with 422 Unprocessable Entity."""
    response = client.post("/analyze", json={"reviews": []})
    assert response.status_code == 422
