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


def test_media_reviews_increase_product_genuineness_score():
    """Verify that adding customer media reviews increases the product genuineness score."""
    base_reviews = [
        {
            "id": f"r{i}",
            "text": f"Review {i}: I have been using this product for several weeks and it works well.",
            "rating": 4,
            "has_images": False,
            "has_video": False,
        }
        for i in range(6)
    ]
    base_results = [
        {
            "id": f"r{i}",
            "label": "genuine",
            "confidence": 0.80,
            "p_deceptive": 0.20,
            "reasons": [],
        }
        for i in range(6)
    ]

    base_summary = compute_reliability(base_reviews, base_results)
    assert base_summary["media_review_count"] == 0
    assert base_summary["media_review_pct"] == 0.0

    # Add media to two of the reviews
    media_reviews = [dict(r) for r in base_reviews]
    media_reviews[0]["has_images"] = True
    media_reviews[1]["has_video"] = True

    media_summary = compute_reliability(media_reviews, base_results)

    # Adding media reviews must increase product genuineness score
    assert media_summary["product_genuineness_score"] > base_summary["product_genuineness_score"]
    assert media_summary["media_review_count"] == 2
    assert media_summary["media_review_pct"] == 33.3
    assert media_summary["product_genuineness_label"] == "Likely Genuine"


def test_volume_risk_warnings_for_various_review_counts():
    """Verify that review sets with < 5 reviews trigger high risk and 5-9 trigger moderate risk."""
    high_risk_warning = "Very few reviews. High risk of fabricated ratings or insufficient track record."
    moderate_risk_warning = "Limited review sample size."

    # 1. 0 reviews (empty)
    empty_summary = compute_reliability([], [])
    assert empty_summary["volume_risk"]["is_risky"] is True
    assert empty_summary["volume_risk"]["level"] == "High"
    assert empty_summary["volume_risk"]["warning"] == high_risk_warning

    # 2. 3 reviews (< 5 reviews)
    revs_3 = [
        {"id": f"r{i}", "text": f"Review {i} text with details.", "rating": 5}
        for i in range(3)
    ]
    res_3 = [{"id": f"r{i}", "label": "genuine", "confidence": 0.85} for i in range(3)]
    summary_3 = compute_reliability(revs_3, res_3)
    assert summary_3["volume_risk"]["is_risky"] is True
    assert summary_3["volume_risk"]["level"] == "High"
    assert summary_3["volume_risk"]["warning"] == high_risk_warning
    assert high_risk_warning in summary_3["insights"]

    # 3. 7 reviews (5 <= n < 10)
    revs_7 = [
        {"id": f"r{i}", "text": f"Review {i} text with details.", "rating": 4}
        for i in range(7)
    ]
    res_7 = [{"id": f"r{i}", "label": "genuine", "confidence": 0.85} for i in range(7)]
    summary_7 = compute_reliability(revs_7, res_7)
    assert summary_7["volume_risk"]["is_risky"] is True
    assert summary_7["volume_risk"]["level"] == "Moderate"
    assert summary_7["volume_risk"]["warning"] == moderate_risk_warning
    assert moderate_risk_warning in summary_7["insights"]

    # 4. 10 reviews (n >= 10)
    revs_10 = [
        {"id": f"r{i}", "text": f"Review {i} text with details.", "rating": 4}
        for i in range(10)
    ]
    res_10 = [{"id": f"r{i}", "label": "genuine", "confidence": 0.85} for i in range(10)]
    summary_10 = compute_reliability(revs_10, res_10)
    assert summary_10["volume_risk"]["is_risky"] is False
    assert summary_10["volume_risk"]["level"] == "Low"
    assert summary_10["volume_risk"]["warning"] is None


def test_seller_confidence_calculation_and_dissonance():
    """Verify seller confidence score calculation and dissonance detection."""
    reviews = [
        {"id": f"r{i}", "text": f"Genuine review {i} with positive details.", "rating": 5}
        for i in range(6)
    ]
    results = [{"id": f"r{i}", "label": "genuine", "confidence": 0.90} for i in range(6)]

    # 1. Rating out of 5 passed as dict
    summary_high = compute_reliability(reviews, results, seller_info={"rating": 4.5})
    assert summary_high["seller_confidence"] == 90.0

    # 2. Rating passed as numeric
    summary_num = compute_reliability(reviews, results, seller_info=3.5)
    assert summary_num["seller_confidence"] == 70.0

    # 3. None seller_info
    summary_none = compute_reliability(reviews, results, seller_info=None)
    assert summary_none["seller_confidence"] is None

    # 4. Major dissonance: poor seller rating (1.5 / 5 = 30%) with 5-star glowing reviews
    summary_dissonance = compute_reliability(reviews, results, seller_info={"rating": 1.5})
    assert summary_dissonance["seller_confidence"] == 30.0
    insights_str = " ".join(summary_dissonance["insights"]).lower()
    assert "seller dissonance" in insights_str or "dissonance" in insights_str


def test_product_genuineness_labels():
    """Verify categorical product genuineness labels across quality levels."""
    # Likely Genuine
    genuine_revs = [
        {"id": f"r{i}", "text": f"Review {i} with high quality.", "rating": 4}
        for i in range(6)
    ]
    genuine_res = [
        {"id": f"r{i}", "label": "genuine", "confidence": 0.90, "p_deceptive": 0.10}
        for i in range(6)
    ]
    sum_gen = compute_reliability(genuine_revs, genuine_res)
    assert sum_gen["product_genuineness_label"] == "Likely Genuine"
    assert sum_gen["product_genuineness_score"] >= 70.0

    # Potentially Deceptive / Risky
    deceptive_revs = [
        {"id": f"r{i}", "text": f"Review {i} suspicious text.", "rating": 5}
        for i in range(6)
    ]
    deceptive_res = [
        {"id": f"r{i}", "label": "suspicious", "confidence": 0.95, "p_deceptive": 0.95}
        for i in range(6)
    ]
    sum_dec = compute_reliability(deceptive_revs, deceptive_res)
    assert sum_dec["product_genuineness_label"] == "Potentially Deceptive / Risky"
    assert sum_dec["product_genuineness_score"] < 40.0
