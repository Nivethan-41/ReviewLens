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


def test_sample_indian_ecom_html_structure(extension_dir: Path) -> None:
    """Verify sample_indian_ecom.html contains simulated cards and seller info for Indian sites."""
    indian_fixture = extension_dir / "test" / "sample_indian_ecom.html"
    assert indian_fixture.exists(), f"sample_indian_ecom.html missing at {indian_fixture}"

    html_content = indian_fixture.read_text(encoding="utf-8")
    soup = BeautifulSoup(html_content, "html.parser")

    # Amazon India checks
    amz_cards = soup.select("[data-hook='review']")
    assert len(amz_cards) == 3, f"Expected 3 Amazon review cards, got {len(amz_cards)}"
    assert soup.select_one("#merchant-info") is not None
    assert soup.select_one("#sellerProfileTriggerId") is not None
    assert len(soup.select(".review-image-tile, [data-hook='review-image-tile']")) >= 1
    assert len(soup.select("video, [data-hook='review-video']")) >= 1
    assert len(soup.select("[data-hook='avp-badge']")) >= 1

    # Flipkart checks
    fk_cards = soup.select("div._27M-vq")
    assert len(fk_cards) == 3, f"Expected 3 Flipkart review cards, got {len(fk_cards)}"
    assert soup.select_one("#sellerName") is not None
    assert soup.select_one("._1RLviY") is not None
    assert len(soup.select("div._2tNq8e, img._3kidjx")) >= 1
    assert len(soup.select("._2mcSum")) >= 1

    # Meesho checks
    msh_cards = soup.select(".ReviewCard")
    assert len(msh_cards) == 1, f"Expected 1 Meesho review card, got {len(msh_cards)}"
    assert soup.select_one(".SupplierCard") is not None
    assert soup.select_one(".supplier-name") is not None
    assert len(soup.select(".review-image")) >= 1


def test_sample_indian_ecom_extraction_simulation(extension_dir: Path) -> None:
    """Verify extraction logic detects customer media, verified purchase, and seller info."""
    import re

    indian_fixture = extension_dir / "test" / "sample_indian_ecom.html"
    html_content = indian_fixture.read_text(encoding="utf-8")
    soup = BeautifulSoup(html_content, "html.parser")

    # 1. Test Amazon Review Extraction
    amz_extracted = []
    for card in soup.select("[data-hook='review']"):
        body_el = card.select_one("[data-hook='review-body']") or card
        text = " ".join(body_el.get_text().split())

        has_images = bool(card.select(".review-image-tile, [data-hook='review-image-tile']"))
        has_video = bool(card.select("video, [data-hook='review-video']"))
        verified = bool(card.select("[data-hook='avp-badge']"))

        rating_el = card.select_one("[data-hook='review-star-rating']")
        rating = None
        if rating_el:
            match = re.search(r"([1-5](?:\.\d+)?)\s*(?:out of|\/)\s*5", rating_el.get_text())
            if match:
                rating = float(match.group(1))

        amz_extracted.append({
            "id": card.get("id"),
            "text": text,
            "rating": rating,
            "has_images": has_images,
            "has_video": has_video,
            "verified_purchase": verified,
        })

    assert len(amz_extracted) == 3
    # Review 1: with photos, verified
    assert amz_extracted[0]["has_images"] is True
    assert amz_extracted[0]["has_video"] is False
    assert amz_extracted[0]["verified_purchase"] is True
    assert amz_extracted[0]["rating"] == 5.0

    # Review 2: with video, verified
    assert amz_extracted[1]["has_images"] is False
    assert amz_extracted[1]["has_video"] is True
    assert amz_extracted[1]["verified_purchase"] is True
    assert amz_extracted[1]["rating"] == 4.0

    # Review 3: text only, unverified
    assert amz_extracted[2]["has_images"] is False
    assert amz_extracted[2]["has_video"] is False
    assert amz_extracted[2]["verified_purchase"] is False
    assert amz_extracted[2]["rating"] == 2.0

    # 2. Test Flipkart Review Extraction
    fk_extracted = []
    for card in soup.select("div._27M-vq"):
        body_el = card.select_one("div.t-ZTKy") or card
        text = " ".join(body_el.get_text().split())

        has_images = bool(card.select("div._2tNq8e img, img._3kidjx"))
        has_video = bool(card.select("div._30kQv0, video"))
        verified = bool(card.select("._2mcSum"))

        rating_el = card.select_one("div._3LWZlK")
        rating = None
        if rating_el:
            match = re.search(r"([1-5](?:\.\d+)?)", rating_el.get_text())
            if match:
                rating = float(match.group(1))

        fk_extracted.append({
            "id": card.get("id"),
            "text": text,
            "rating": rating,
            "has_images": has_images,
            "has_video": has_video,
            "verified_purchase": verified,
        })

    assert len(fk_extracted) == 3
    # FK Review 1: with photo, verified
    assert fk_extracted[0]["has_images"] is True
    assert fk_extracted[0]["has_video"] is False
    assert fk_extracted[0]["verified_purchase"] is True
    assert fk_extracted[0]["rating"] == 5.0

    # FK Review 2: with video, verified
    assert fk_extracted[1]["has_images"] is False
    assert fk_extracted[1]["has_video"] is True
    assert fk_extracted[1]["verified_purchase"] is True
    assert fk_extracted[1]["rating"] == 4.0

    # FK Review 3: text only, verified
    assert fk_extracted[2]["has_images"] is False
    assert fk_extracted[2]["has_video"] is False
    assert fk_extracted[2]["verified_purchase"] is True
    assert fk_extracted[2]["rating"] == 1.0

    # 3. Test Seller Info Extraction
    # Amazon seller
    seller_profile = soup.select_one("#sellerProfileTriggerId")
    assert seller_profile is not None
    assert "Appario Retail" in seller_profile.get_text()

    # Flipkart seller
    seller_name_el = soup.select_one("#sellerName ._1RLviY span")
    assert seller_name_el is not None
    assert "RetailNet" in seller_name_el.get_text()

    # Meesho supplier
    supplier_el = soup.select_one(".SupplierCard .supplier-name")
    assert supplier_el is not None
    assert "Surat Fashion Hub" in supplier_el.get_text()


def test_sample_indian_ecom_backend_integration(
    extension_dir: Path,
    test_client: TestClient,
) -> None:
    """Verify that simulated Indian e-commerce payload with media flags and seller_info is analyzed successfully."""
    import re

    indian_fixture = extension_dir / "test" / "sample_indian_ecom.html"
    html_content = indian_fixture.read_text(encoding="utf-8")
    soup = BeautifulSoup(html_content, "html.parser")

    reviews = []
    # Collect from Amazon and Flipkart sections
    for idx, card in enumerate(soup.select("[data-hook='review'], div._27M-vq"), start=1):
        body_el = card.select_one("[data-hook='review-body'], div.t-ZTKy") or card
        text = " ".join(body_el.get_text().split())
        if len(text) < 10:
            continue

        has_images = bool(card.select(".review-image-tile, [data-hook='review-image-tile'], div._2tNq8e, img._3kidjx"))
        has_video = bool(card.select("video, [data-hook='review-video'], div._30kQv0"))
        verified = bool(card.select("[data-hook='avp-badge'], ._2mcSum"))

        rating_el = card.select_one("[data-hook='review-star-rating'], div._3LWZlK")
        rating = None
        if rating_el:
            match = re.search(r"([1-5](?:\.\d+)?)", rating_el.get_text())
            if match:
                rating = float(match.group(1))

        reviews.append({
            "id": f"ind-rev-{idx}",
            "text": text,
            "rating": rating,
            "date": "2026-10-15",
            "has_images": has_images,
            "has_video": has_video,
            "verified_purchase": verified,
        })

    assert len(reviews) == 6

    # Test POST /analyze with reviews and seller_info
    payload = {
        "page_url": "https://www.amazon.in/dp/B07X2T5Y8H",
        "reviews": reviews,
        "seller_info": {
            "name": "Appario Retail Private Ltd",
            "rating": 4.7,
        },
    }

    response = test_client.post("/analyze", json=payload)
    assert response.status_code == status.HTTP_200_OK

    data = response.json()
    assert "results" in data
    assert len(data["results"]) == 6
    assert "summary" in data
    summary = data["summary"]
    assert summary["total"] == 6
    assert "reliability_score" in summary
    assert "reliability_label" in summary


def test_content_js_selectors_and_functions_via_node(extension_dir: Path) -> None:
    """Run Node.js script to verify SELECTORS, parseRating, parseDate, and deduplicateAndFormat."""
    import subprocess

    content_js_path = extension_dir / "content.js"
    assert content_js_path.exists()

    node_script = f"""
    const c = require({json.dumps(str(content_js_path))});

    // 1. Verify SELECTORS keys and specific Indian e-com hooks
    if (!c.SELECTORS.amazon || !c.SELECTORS.flipkart || !c.SELECTORS.meesho) {{
        console.error('Missing platform selectors');
        process.exit(1);
    }}
    if (!c.SELECTORS.amazon.reviewCard.includes('[data-hook=\\'review\\']')) {{
        console.error('Amazon reviewCard missing');
        process.exit(1);
    }}
    if (!c.SELECTORS.flipkart.reviewCard.includes('_27M-vq')) {{
        console.error('Flipkart reviewCard missing');
        process.exit(1);
    }}
    if (!c.SELECTORS.amazon.verifiedPurchase.includes('avp-badge')) {{
        console.error('Amazon verifiedPurchase missing');
        process.exit(1);
    }}
    if (!c.SELECTORS.flipkart.verifiedPurchase.includes('_2mcSum')) {{
        console.error('Flipkart verifiedPurchase missing');
        process.exit(1);
    }}

    // 2. Verify parseRating
    if (c.parseRating('4.4 ★') !== 4.4) {{
        console.error('Failed parseRating 4.4 ★');
        process.exit(1);
    }}
    if (c.parseRating('5.0 out of 5 stars') !== 5.0) {{
        console.error('Failed parseRating 5.0 out of 5 stars');
        process.exit(1);
    }}

    // 3. Verify parseDate
    if (c.parseDate('Reviewed in India on 2026-10-15') !== '2026-10-15') {{
        console.error('Failed parseDate with India prefix');
        process.exit(1);
    }}

    // 4. Verify deduplicateAndFormat defaults
    const out = c.deduplicateAndFormat([
        {{ text: 'High quality product that meets all specifications.', rating: 4 }}
    ]);
    if (out.length !== 1 || out[0].has_images !== false || out[0].has_video !== false || out[0].verified_purchase !== false) {{
        console.error('Failed default values in deduplicateAndFormat');
        process.exit(1);
    }}

    console.log('ALL_NODE_TESTS_PASSED');
    """

    res = subprocess.run(["node", "-e", node_script], capture_output=True, text=True, check=True)
    assert "ALL_NODE_TESTS_PASSED" in res.stdout


def test_popup_html_elements_and_dashboard_structure(extension_dir: Path) -> None:
    """Verify popup.html contains the Product Genuineness Dashboard, volume banner, stats grid, and chips."""
    popup_path = extension_dir / "popup.html"
    assert popup_path.exists(), f"popup.html missing at {popup_path}"

    html_content = popup_path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html_content, "html.parser")

    # 1. Volume Risk Alert Banner
    vol_banner = soup.select_one("#volume-risk-banner")
    assert vol_banner is not None, "Missing #volume-risk-banner"
    assert soup.select_one("#volume-risk-text") is not None, "Missing #volume-risk-text"

    # 2. Top Product Genuineness Card & Gauges
    gen_card = soup.select_one(".genuineness-card")
    assert gen_card is not None, "Missing .genuineness-card container"
    assert soup.select_one("#genuineness-verdict") is not None, "Missing #genuineness-verdict label"
    assert soup.select_one("#genuineness-circle") is not None, "Missing circular gauge SVG #genuineness-circle"
    assert soup.select_one("#product-genuineness-score") is not None, "Missing #product-genuineness-score"
    assert soup.select_one("#gauge-fill") is not None, "Missing horizontal #gauge-fill"

    # 3. Decision-Support Stats Grid
    assert soup.select_one("#stat-media-pct") is not None, "Missing #stat-media-pct"
    assert soup.select_one("#stat-media-label") is not None, "Missing #stat-media-label"
    assert soup.select_one("#stat-suspicious-pct") is not None, "Missing #stat-suspicious-pct"
    assert soup.select_one("#stat-suspicious-label") is not None, "Missing #stat-suspicious-label"
    assert soup.select_one("#stat-seller-rating") is not None, "Missing #stat-seller-rating"
    assert soup.select_one("#stat-seller-label") is not None, "Missing #stat-seller-label"

    # 4. Filter chips: All, With Photos/Videos, Suspicious, Genuine
    chips = soup.select(".filter-chips .chip")
    chip_filters = [c.get("data-filter") for c in chips]
    assert "all" in chip_filters, "Missing 'all' filter chip"
    assert "media" in chip_filters, "Missing 'media' filter chip (With Photos/Videos)"
    assert "suspicious" in chip_filters, "Missing 'suspicious' filter chip"
    assert "genuine" in chip_filters, "Missing 'genuine' filter chip"

    # 5. Sort toggle control
    assert soup.select_one("#sort-toggle-btn") is not None, "Missing #sort-toggle-btn"
    assert soup.select_one("#sort-toggle-text") is not None, "Missing #sort-toggle-text"


def test_popup_css_dashboard_styles(extension_dir: Path) -> None:
    """Verify popup.css contains styling rules for the risk banner, gauges, media badge, and callout."""
    css_path = extension_dir / "popup.css"
    assert css_path.exists(), f"popup.css missing at {css_path}"

    css_content = css_path.read_text(encoding="utf-8")

    # Volume banner styles
    assert ".volume-risk-banner" in css_content
    assert ".volume-risk-moderate" in css_content or ".volume-risk-high" in css_content

    # Genuineness circular & horizontal gauge styles
    assert ".circular-gauge-box" in css_content
    assert ".circular-gauge-svg" in css_content
    assert ".gauge-circle-fill" in css_content
    assert ".stroke-green" in css_content
    assert ".stroke-amber" in css_content
    assert ".stroke-red" in css_content

    # Decision stats grid
    assert ".decision-stats-grid" in css_content
    assert ".decision-stat-card" in css_content

    # Media & verified purchase badges and highlight
    assert ".badge-media" in css_content
    assert ".badge-verified" in css_content
    assert ".media-trust-highlight" in css_content
    assert ".btn-sort" in css_content


def test_popup_js_integration_and_rendering(extension_dir: Path) -> None:
    """Verify popup.js renders genuineness score, volume risk, media stats, seller trust, and badges."""
    js_path = extension_dir / "popup.js"
    assert js_path.exists(), f"popup.js missing at {js_path}"

    content = js_path.read_text(encoding="utf-8")

    # Verify key tokens and references
    assert "product_genuineness_score" in content
    assert "product_genuineness_label" in content
    assert "volume_risk" in content
    assert "media_review_pct" in content
    assert "seller_confidence" in content
    assert "badge-media" in content
    assert "badge-verified" in content
    assert "media-trust-highlight" in content
    assert "sort-toggle-btn" in content
    assert "data-filter" in content


