"""Tests for ml/predict.py."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pytest

from ml.config import MODEL_PATH
from ml.predict import DEMO_REVIEWS, ModelService


@pytest.fixture(scope="module")
def model_service() -> ModelService:
    """Instantiate a ModelService once for the test module."""
    return ModelService(MODEL_PATH)


def test_missing_model_raises_file_not_found(tmp_path: Path):
    """Verify ModelService raises FileNotFoundError when model path does not exist."""
    fake_path = tmp_path / "non_existent_model.joblib"
    with pytest.raises(FileNotFoundError, match="Model artifact not found"):
        ModelService(model_path=fake_path)


def test_analyze_output_schema_matches_prd_contract(model_service: ModelService):
    """Verify analyze output structure and schema strictly match PRD.md Section 9."""
    reviews = [
        {
            "id": "rev_1",
            "text": "Solid product, arrived in 2 days and works as expected. Battery lasts about 24 hours.",
            "rating": 5,
            "date": "2026-09-01",
        },
        {
            "id": "rev_2",
            "text": "AMAZING BUY NOW BEST THING EVER INCREDIBLE LUXURY TOP RATED 100% PERFECT!!!",
            "rating": 5,
            "date": "2026-09-02",
        },
    ]

    response = model_service.analyze(reviews)

    # 1. Top-level keys
    assert isinstance(response, dict)
    assert set(response.keys()) == {"results", "summary"}

    # 2. Results list structure
    results = response["results"]
    assert isinstance(results, list)
    assert len(results) == 2

    for r in results:
        assert isinstance(r, dict)
        assert set(r.keys()) == {"id", "label", "confidence", "reasons"}
        assert isinstance(r["id"], str)
        assert r["label"] in {"genuine", "suspicious", "uncertain"}
        assert isinstance(r["confidence"], float)
        assert 0.5 <= r["confidence"] <= 1.0

        # Reasons list
        assert isinstance(r["reasons"], list)
        assert len(r["reasons"]) >= 1
        for reason in r["reasons"]:
            assert isinstance(reason, dict)
            assert "type" in reason and "text" in reason
            assert reason["type"] in {"language", "style", "pattern"}
            assert isinstance(reason["text"], str)

    # 3. Summary structure
    summary = response["summary"]
    assert isinstance(summary, dict)
    expected_summary_keys = {
        "total",
        "suspicious_count",
        "suspicious_pct",
        "reliability_score",
        "reliability_label",
        "insights",
        "product_genuineness_score",
        "product_genuineness_label",
        "media_review_count",
        "media_review_pct",
        "volume_risk",
        "seller_confidence",
    }
    assert set(summary.keys()) == expected_summary_keys

    assert summary["total"] == 2
    assert isinstance(summary["suspicious_count"], int)
    assert 0 <= summary["suspicious_count"] <= 2
    assert isinstance(summary["suspicious_pct"], float)
    assert 0.0 <= summary["suspicious_pct"] <= 100.0
    assert isinstance(summary["reliability_score"], int)
    assert 0 <= summary["reliability_score"] <= 100
    assert isinstance(summary["reliability_label"], str)
    assert isinstance(summary["insights"], list)


def test_order_preservation(model_service: ModelService):
    """Verify analyze maintains input reviews order in output results."""
    reviews = [
        {"id": "id_third", "text": "Third review with realistic text details for testing."},
        {"id": "id_first", "text": "First review with realistic text details for testing."},
        {"id": "id_second", "text": "Second review with realistic text details for testing."},
    ]
    response = model_service.analyze(reviews)
    output_ids = [r["id"] for r in response["results"]]
    assert output_ids == ["id_third", "id_first", "id_second"]


def test_generate_ids_when_missing(model_service: ModelService):
    """Verify analyze generates ids (e.g. r1, r2) when id is missing or None."""
    reviews = [
        {"text": "Review without explicit id field."},
        {"id": None, "text": "Review with None as id value."},
        {"id": "   ", "text": "Review with whitespace id value."},
        {"id": "custom_id", "text": "Review with explicitly given id."},
    ]
    response = model_service.analyze(reviews)
    results = response["results"]

    assert results[0]["id"] == "r1"
    assert results[1]["id"] == "r2"
    assert results[2]["id"] == "r3"
    assert results[3]["id"] == "custom_id"


def test_empty_text_validation_raises_error(model_service: ModelService):
    """Verify analyze raises ValueError on empty or whitespace review text."""
    with pytest.raises(ValueError, match="empty or missing text"):
        model_service.analyze([{"text": ""}])

    with pytest.raises(ValueError, match="empty or missing text"):
        model_service.analyze([{"id": "r1", "text": "   \n\t  "}])

    with pytest.raises(ValueError, match="empty or missing text"):
        model_service.analyze([{"id": "r2"}])


def test_long_text_truncated_to_5000_chars(model_service: ModelService):
    """Verify review text exceeding 5,000 characters is safely truncated."""
    long_text = "Good battery life. " * 350  # Over 6,000 characters
    assert len(long_text) > 5000

    response = model_service.analyze([{"id": "long_rev", "text": long_text}])
    assert len(response["results"]) == 1
    assert response["results"][0]["id"] == "long_rev"


def test_max_reviews_limit_enforced(model_service: ModelService):
    """Verify batch limit of 200 reviews is enforced."""
    too_many_reviews = [
        {"id": f"r{i}", "text": f"Review number {i} text."}
        for i in range(201)
    ]
    with pytest.raises(ValueError, match="exceeds the maximum limit of 200"):
        model_service.analyze(too_many_reviews)


def test_empty_reviews_list_handling(model_service: ModelService):
    """Verify passing an empty review list returns valid empty structure."""
    response = model_service.analyze([])
    assert response["results"] == []
    assert response["summary"]["total"] == 0
    assert response["summary"]["suspicious_count"] == 0
    assert response["summary"]["suspicious_pct"] == 0.0
    assert response["summary"]["reliability_label"] == "Not enough data"


def test_demo_reviews_execution(model_service: ModelService):
    """Verify DEMO_REVIEWS contains 10 items and analyzes properly."""
    assert len(DEMO_REVIEWS) == 10
    response = model_service.analyze(DEMO_REVIEWS)

    assert response["summary"]["total"] == 10
    assert len(response["results"]) == 10
    # Demo reviews include 3 identical reviews, so near-duplicate insight must appear
    dup_insights = [s for s in response["summary"]["insights"] if "near-duplicate" in s.lower()]
    assert len(dup_insights) == 1


def test_thread_safety(model_service: ModelService):
    """Verify analyze can be safely called from multiple concurrent threads."""
    test_batch = [
        {"id": f"r{i}", "text": f"Thread review {i}: dependable quality and good experience."}
        for i in range(5)
    ]

    def _worker():
        return model_service.analyze(test_batch)

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(_worker) for _ in range(8)]
        results = [f.result() for f in futures]

    first_result_labels = [r["label"] for r in results[0]["results"]]
    for res in results[1:]:
        assert [r["label"] for r in res["results"]] == first_result_labels


def test_analyze_with_seller_info_and_media_reason(model_service: ModelService):
    """Verify analyze accepts seller_info and appends media credibility reason to genuine reviews."""
    reviews = [
        {
            "id": "rev_media_1",
            "text": "I bought this kettle 2 months ago. It boils water quickly in 3 minutes and the stainless steel interior is easy to rinse.",
            "rating": 5,
            "has_images": True,
        },
        {
            "id": "rev_text_2",
            "text": "Average device that does the job as described. Nothing spectacular.",
            "rating": 3,
            "has_images": False,
        },
    ]

    response = model_service.analyze(reviews, seller_info={"rating": 4.5})

    # Verify seller_confidence in summary
    assert response["summary"]["seller_confidence"] == 90.0
    assert response["summary"]["media_review_count"] == 1
    assert response["summary"]["media_review_pct"] == 50.0

    # Verify media reason attached to rev_media_1
    res1 = next(r for r in response["results"] if r["id"] == "rev_media_1")
    if res1["label"] == "genuine":
        media_reasons = [rsn for rsn in res1["reasons"] if rsn["type"] == "media"]
        assert len(media_reasons) == 1
        assert "Customer photo/video attached" in media_reasons[0]["text"]


def test_prioritize_media_rich_reviews_in_sorting(model_service: ModelService):
    """Verify results ordering prioritizes media-rich reviews when sorting for user inspection."""
    reviews = [
        {"id": "text_only_1", "text": "Ordinary text-only review 1.", "has_images": False},
        {"id": "media_rich_1", "text": "Genuine review with attached customer photos.", "has_images": True},
        {"id": "text_only_2", "text": "Ordinary text-only review 2.", "has_images": False},
        {"id": "media_rich_2", "text": "Another review with attached video proof.", "has_video": True},
    ]

    response = model_service.analyze(reviews, sort_for_inspection=True)
    ordered_ids = [r["id"] for r in response["results"]]

    # Both media-rich reviews must appear before the text-only reviews
    assert ordered_ids[:2] == ["media_rich_1", "media_rich_2"] or set(ordered_ids[:2]) == {"media_rich_1", "media_rich_2"}
    assert set(ordered_ids[2:]) == {"text_only_1", "text_only_2"}
