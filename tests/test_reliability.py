"""Tests for ml/reliability.py."""

from __future__ import annotations

import pytest

from ml.reliability import compute_reliability


def test_clean_versus_duplicate_heavy_set():
    """Verify that an obviously clean set scores clearly higher than identical 5-star reviews."""
    clean_reviews = [
        {
            "id": f"c_{i}",
            "text": f"Review {i}: I have been using this device for {i + 1} weeks now. Battery lasts well, build quality is solid and it fits comfortably.",
            "rating": 4 if i % 2 == 0 else 5,
            "date": f"2026-0{1 + (i % 8)}-15",
        }
        for i in range(10)
    ]
    clean_results = [
        {
            "id": f"c_{i}",
            "label": "genuine",
            "confidence": 0.88,
            "reasons": [{"type": "language", "text": "Genuine phrasing"}],
        }
        for i in range(10)
    ]

    dup_text = "BEST PRODUCT EVER!! ABSOLUTELY AMAZING INCREDIBLE MUST BUY FIVE STARS WOW!!!"
    duplicate_reviews = [
        {
            "id": f"d_{i}",
            "text": dup_text,
            "rating": 5,
            "date": "2026-10-01",
        }
        for i in range(10)
    ]
    duplicate_results = [
        {
            "id": f"d_{i}",
            "label": "suspicious",
            "confidence": 0.95,
            "reasons": [{"type": "style", "text": "ALL CAPS"}],
        }
        for i in range(10)
    ]

    clean_summary = compute_reliability(clean_reviews, clean_results)
    dup_summary = compute_reliability(duplicate_reviews, duplicate_results)

    # Clean set must score clearly higher than the duplicate-heavy set
    assert clean_summary["reliability_score"] > dup_summary["reliability_score"]
    assert clean_summary["reliability_score"] >= 70
    assert clean_summary["reliability_label"] == "High"

    assert dup_summary["reliability_score"] < 50
    assert dup_summary["reliability_label"] in ("Low", "Moderate")

    # Near-duplicate insight must be present in the duplicate set
    dup_insights_str = " ".join(dup_summary["insights"]).lower()
    assert "near-duplicate" in dup_insights_str
    assert "burst" in dup_insights_str or "same day" in dup_insights_str


def test_fewer_than_five_reviews_edge_case():
    """Verify that fewer than 5 reviews return 'Not enough data' label."""
    # 0 reviews
    empty_summary = compute_reliability([], [])
    assert empty_summary["reliability_label"] == "Not enough data"
    assert empty_summary["reliability_score"] == 0

    # 1 to 4 reviews
    for count in range(1, 5):
        reviews = [
            {
                "id": f"r{i}",
                "text": f"Genuine honest review {i} with detailed feedback.",
                "rating": 4,
                "date": "2026-09-01",
            }
            for i in range(count)
        ]
        results = [
            {
                "id": f"r{i}",
                "label": "genuine",
                "confidence": 0.90,
                "reasons": [],
            }
            for i in range(count)
        ]
        summary = compute_reliability(reviews, results)
        assert summary["reliability_label"] == "Not enough data"
        assert any("fewer than 5" in s.lower() for s in summary["insights"])

    # 5 reviews should produce a standard label (High, Moderate, or Low)
    reviews_5 = [
        {
            "id": f"r{i}",
            "text": f"Genuine honest review {i} with detailed feedback and usage specs.",
            "rating": 4,
            "date": f"2026-09-0{i + 1}",
        }
        for i in range(5)
    ]
    results_5 = [
        {
            "id": f"r{i}",
            "label": "genuine",
            "confidence": 0.90,
            "reasons": [],
        }
        for i in range(5)
    ]
    summary_5 = compute_reliability(reviews_5, results_5)
    assert summary_5["reliability_label"] in ("High", "Moderate", "Low")
    assert summary_5["reliability_label"] != "Not enough data"


def test_missing_rating_field():
    """Verify missing rating field is skipped without penalizing reliability."""
    reviews_no_rating = [
        {
            "id": f"r{i}",
            "text": f"Review {i}: Excellent device that works reliably every day with good battery.",
            # rating omitted
            "date": f"2026-08-0{i + 1}",
        }
        for i in range(6)
    ]
    results = [
        {
            "id": f"r{i}",
            "label": "genuine",
            "confidence": 0.85,
            "reasons": [],
        }
        for i in range(6)
    ]

    summary = compute_reliability(reviews_no_rating, results)
    assert summary["reliability_score"] >= 70
    assert summary["reliability_label"] == "High"
    # No rating insights when ratings are absent
    assert not any("stars" in s for s in summary["insights"])


def test_missing_date_field():
    """Verify missing date field is skipped without penalizing reliability."""
    reviews_no_date = [
        {
            "id": f"r{i}",
            "text": f"Review {i}: Excellent device that works reliably every day with good battery.",
            "rating": 4,
            # date omitted
        }
        for i in range(6)
    ]
    results = [
        {
            "id": f"r{i}",
            "label": "genuine",
            "confidence": 0.85,
            "reasons": [],
        }
        for i in range(6)
    ]

    summary = compute_reliability(reviews_no_date, results)
    assert summary["reliability_score"] >= 70
    assert summary["reliability_label"] == "High"
    # No burst insights when dates are absent
    assert not any("burst" in s.lower() for s in summary["insights"])


def test_missing_both_rating_and_date_fields():
    """Verify missing both rating and date fields skips both optional penalties."""
    reviews = [
        {
            "id": f"r{i}",
            "text": f"Review {i}: High quality and durable item that I used for 3 months.",
        }
        for i in range(6)
    ]
    results = [
        {
            "id": f"r{i}",
            "label": "genuine",
            "confidence": 0.90,
            "reasons": [],
        }
        for i in range(6)
    ]

    summary = compute_reliability(reviews, results)
    assert summary["reliability_score"] >= 70
    assert summary["reliability_label"] == "High"


def test_rating_skew_all_5_stars_insight():
    """Verify 100% 5-star distribution triggers skew insight and reduces score."""
    reviews_5star = [
        {
            "id": f"r{i}",
            "text": f"Product review {i} with some specific details about size and fit.",
            "rating": 5,
            "date": f"2026-07-0{i + 1}",
        }
        for i in range(6)
    ]
    results = [
        {
            "id": f"r{i}",
            "label": "genuine",
            "confidence": 0.80,
            "reasons": [],
        }
        for i in range(6)
    ]

    summary = compute_reliability(reviews_5star, results)
    assert any("5 stars" in s for s in summary["insights"])
