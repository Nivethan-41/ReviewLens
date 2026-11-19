"""Tests for Chrome Manifest V3 extension configuration and sample review extraction.

Verifies manifest.json compliance with Manifest V3, permissions, icon integrity,
config constants, and verifies extraction of all 12 reviews on sample_reviews.html.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from bs4 import BeautifulSoup
from fastapi import status
from fastapi.testclient import TestClient
from PIL import Image
import pytest

from api.main import app
from ml.config import PROJECT_ROOT, RANDOM_SEED


@pytest.fixture(scope="module")
def extension_dir() -> Path:
    """Return the absolute path to the extension/ directory."""
    return PROJECT_ROOT / "extension"


@pytest.fixture(scope="module")
def manifest_data(extension_dir: Path) -> Dict[str, Any]:
    """Load and parse extension/manifest.json."""
    manifest_path = extension_dir / "manifest.json"
    assert manifest_path.exists(), f"manifest.json missing at {manifest_path}"
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def test_client() -> TestClient:
    """Provide a FastAPI TestClient with the application lifespan entered."""
    with TestClient(app) as client:
        yield client


def test_manifest_structure_and_permissions(manifest_data: Dict[str, Any]) -> None:
    """Verify manifest.json matches Chrome Manifest V3 requirements and prompt constraints.

    Must specify Manifest V3, ReviewLens name, background service worker,
    action popup, required permissions, host_permissions, and NO content_scripts
    (as injection is on demand).
    """
    assert manifest_data.get("manifest_version") == 3
    assert manifest_data.get("name") == "ReviewLens"
    assert manifest_data.get("version") == "0.1.0"

    # Required permissions
    permissions = manifest_data.get("permissions", [])
    assert "activeTab" in permissions
    assert "scripting" in permissions
    assert "storage" in permissions

    # Required host permissions
    host_permissions = manifest_data.get("host_permissions", [])
    assert "http://localhost:8000/*" in host_permissions
    assert "https://*/*" in host_permissions

    # Service worker background configuration
    background = manifest_data.get("background", {})
    assert background.get("service_worker") == "background.js"

    # Action popup configuration
    action = manifest_data.get("action", {})
    assert action.get("default_popup") == "popup.html"

    # On-demand injection rule: content_scripts should NOT be statically registered
    assert "content_scripts" not in manifest_data, (
        "Manifest must not declare static content_scripts; injection must be on demand."
    )


def test_extension_icons_exist_and_valid(extension_dir: Path, manifest_data: Dict[str, Any]) -> None:
    """Verify generated icons exist for sizes 16, 48, and 128 and are valid PNG images."""
    icons_dict = manifest_data.get("icons", {})
    assert "16" in icons_dict
    assert "48" in icons_dict
    assert "128" in icons_dict

    for size_str in ("16", "48", "128"):
        rel_path = icons_dict[size_str]
        icon_path = extension_dir / rel_path
        assert icon_path.exists(), f"Icon file missing at {icon_path}"
        assert icon_path.stat().st_size > 0

        # Verify image properties using PIL
        with Image.open(icon_path) as img:
            assert img.format == "PNG"
            expected_dim = int(size_str)
            assert img.size == (expected_dim, expected_dim)


def test_config_js_contains_api_base_url(extension_dir: Path) -> None:
    """Verify config.js defines API_BASE_URL constant as localhost:8000."""
    config_path = extension_dir / "config.js"
    assert config_path.exists(), f"config.js missing at {config_path}"
    content = config_path.read_text(encoding="utf-8")
    assert "API_BASE_URL" in content
    assert "http://localhost:8000" in content


def test_sample_reviews_html_contains_12_reviews(extension_dir: Path) -> None:
    """Verify extension/test/sample_reviews.html contains exactly 12 schema.org reviews."""
    sample_path = extension_dir / "test" / "sample_reviews.html"
    assert sample_path.exists(), f"sample_reviews.html missing at {sample_path}"

    html_content = sample_path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html_content, "html.parser")

    # Verify review bodies via itemprop="reviewBody"
    review_bodies = soup.find_all(attrs={"itemprop": "reviewBody"})
    assert len(review_bodies) == 12, f"Expected 12 review bodies, found {len(review_bodies)}"

    # Verify review containers
    review_cards = soup.find_all(attrs={"itemscope": True, "itemtype": lambda val: val and "Review" in val})
    assert len(review_cards) == 12, f"Expected 12 Review schema containers, found {len(review_cards)}"


def test_sample_reviews_extraction_logic(extension_dir: Path) -> None:
    """Simulate generic extraction strategy on sample_reviews.html and verify 12 structured reviews."""
    sample_path = extension_dir / "test" / "sample_reviews.html"
    html_content = sample_path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html_content, "html.parser")

    extracted_reviews: List[Dict[str, Any]] = []
    seen_texts = set()

    # Step b / a simulation matching content.js strategy
    review_elements = soup.find_all(attrs={"itemprop": "reviewBody"})
    for idx, el in enumerate(review_elements, start=1):
        text = " ".join(el.get_text().split())
        assert len(text) >= 10, f"Review text unexpectedly short: '{text}'"

        # Locate container
        container = el.find_parent(attrs={"itemscope": True}) or el.parent

        # Locate rating
        rating_val: Optional[float] = None
        rating_el = container.find(attrs={"itemprop": "ratingValue"}) if container else None
        if rating_el:
            raw_rating = rating_el.get("content") or rating_el.get_text()
            try:
                rating_val = float(raw_rating.strip())
            except ValueError:
                pass

        # Locate date
        date_str: Optional[str] = None
        date_el = container.find(attrs={"itemprop": "datePublished"}) if container else None
        if date_el:
            date_str = date_el.get("datetime") or date_el.get_text().strip()

        # Deduplication
        norm_key = text.lower()
        if norm_key not in seen_texts:
            seen_texts.add(norm_key)
            extracted_reviews.append(
                {
                    "id": f"rev-{idx}",
                    "text": text,
                    "rating": rating_val,
                    "date": date_str,
                }
            )

    # Assert exactly 12 reviews extracted
    assert len(extracted_reviews) == 12

    # Verify all ratings are between 1.0 and 5.0
    for rev in extracted_reviews:
        assert rev["rating"] is not None
        assert 1.0 <= rev["rating"] <= 5.0

    # Verify all dates are valid ISO format YYYY-MM-DD
    for rev in extracted_reviews:
        assert rev["date"] is not None
        assert len(rev["date"]) == 10
        assert rev["date"].startswith("2026-10-")


def test_sample_reviews_backend_analyze_integration(
    extension_dir: Path,
    test_client: TestClient,
) -> None:
    """Verify that extracted reviews from sample_reviews.html can be sent to POST /analyze."""
    sample_path = extension_dir / "test" / "sample_reviews.html"
    html_content = sample_path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html_content, "html.parser")

    reviews_payload = []
    for idx, el in enumerate(soup.find_all(attrs={"itemprop": "reviewBody"}), start=1):
        container = el.find_parent(attrs={"itemscope": True}) or el.parent
        rating_el = container.find(attrs={"itemprop": "ratingValue"}) if container else None
        date_el = container.find(attrs={"itemprop": "datePublished"}) if container else None

        rating = float(rating_el.get_text().strip()) if rating_el else None
        date = (date_el.get("datetime") or date_el.get_text().strip()) if date_el else None

        reviews_payload.append(
            {
                "id": f"r{idx}",
                "text": " ".join(el.get_text().split()),
                "rating": rating,
                "date": date,
            }
        )

    assert len(reviews_payload) == 12

    response = test_client.post(
        "/analyze",
        json={
            "page_url": "file:///extension/test/sample_reviews.html",
            "reviews": reviews_payload,
        },
    )

    assert response.status_code == status.HTTP_200_OK
    data = response.json()

    assert "results" in data
    assert len(data["results"]) == 12
    assert "summary" in data
    summary = data["summary"]
    assert summary["total"] == 12
    assert "reliability_score" in summary
    assert "reliability_label" in summary
    assert 0 <= summary["reliability_score"] <= 100
