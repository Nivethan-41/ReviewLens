"""Reliability scoring module combining review-level and product-level signals.

Calculates an aggregate reliability score (0-100), categorical reliability label
(High, Moderate, Low, or Not enough data), and human-readable insights for a collection
of reviews based on confidence-weighted genuineness, near-duplicate text detection,
rating distribution skew, temporal review bursts, and length/specificity.
"""

from __future__ import annotations

from collections import Counter
from datetime import date
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ml.config import (
    BURST_MAX_DAY_RATIO,
    BURST_WINDOW_DAYS,
    MIN_REVIEWS_FOR_RELIABILITY,
    NEAR_DUPLICATE_SIMILARITY_THRESHOLD,
    RATING_J_SHAPED_THRESHOLD,
    RATING_SKEW_5STAR_THRESHOLD,
    RELIABILITY_HIGH_THRESHOLD,
    RELIABILITY_MODERATE_THRESHOLD,
    RELIABILITY_WEIGHT_LENGTH_SPECIFICITY,
    RELIABILITY_WEIGHT_NEAR_DUPLICATE,
    RELIABILITY_WEIGHT_NON_SUSPICIOUS,
    RELIABILITY_WEIGHT_RATING_SKEW,
    RELIABILITY_WEIGHT_REVIEW_BURST,
    SPECIFICITY_TARGET_WORD_COUNT,
)

logger = logging.getLogger(__name__)


def _compute_non_suspicious_score(results: list[dict[str, Any]]) -> float:
    """Compute confidence-weighted share of non-suspicious reviews.

    Args:
        results: List of prediction dictionaries containing 'label' and 'confidence'.

    Returns:
        Score between 0.0 and 1.0 representing the share of non-suspicious reviews.
    """
    if not results:
        return 0.5

    non_suspicious_mass = 0.0
    suspicious_mass = 0.0

    for res in results:
        label = str(res.get("label", "")).lower()
        conf = float(res.get("confidence", 0.5))
        # Ensure confidence is within [0.5, 1.0]
        conf = max(0.5, min(1.0, conf))

        if label in ("suspicious", "deceptive"):
            suspicious_mass += conf
        elif label == "uncertain":
            non_suspicious_mass += 0.5 * conf
            suspicious_mass += 0.5 * conf
        else:
            # Genuine or any non-suspicious label
            non_suspicious_mass += conf

    total_mass = non_suspicious_mass + suspicious_mass
    if total_mass == 0.0:
        return 0.5

    return float(np.clip(non_suspicious_mass / total_mass, 0.0, 1.0))


def _compute_near_duplicate_score(
    texts: list[str],
    threshold: float = NEAR_DUPLICATE_SIMILARITY_THRESHOLD,
) -> tuple[float, list[str]]:
    """Detect near-duplicate reviews using pairwise TF-IDF cosine similarity.

    Args:
        texts: List of review texts.
        threshold: Cosine similarity cutoff above which two reviews are near-duplicates.

    Returns:
        Tuple of (score in [0.0, 1.0], list of insight strings).
    """
    n = len(texts)
    if n < 2:
        return 1.0, []

    insights: list[str] = []
    duplicate_indices: set[int] = set()

    try:
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            token_pattern=r"(?u)\b\w+\b",
            min_df=1,
        )
        tfidf_matrix = vectorizer.fit_transform(texts)
        sim_matrix = cosine_similarity(tfidf_matrix)
    except ValueError:
        # Occurs if vocabulary is empty (e.g., all empty or non-word strings)
        sim_matrix = np.eye(n)

    # Check for identical strings or high cosine similarity
    for i in range(n):
        for j in range(i + 1, n):
            # Check exact text identity for short non-tokenizable strings
            is_exact = texts[i].strip() != "" and texts[i].strip() == texts[j].strip()
            if is_exact or sim_matrix[i, j] >= threshold:
                duplicate_indices.add(i)
                duplicate_indices.add(j)

    dup_count = len(duplicate_indices)
    if dup_count > 0:
        insights.append(f"{dup_count} near-duplicate reviews found")

    dup_share = dup_count / float(n)
    score = float(np.clip(1.0 - dup_share, 0.0, 1.0))
    return score, insights


def _compute_rating_skew_score(reviews: list[dict[str, Any]]) -> tuple[Optional[float], list[str]]:
    """Analyze rating distribution skew (all-5-star or polarized J-shaped).

    Args:
        reviews: List of review dictionaries containing optional 'rating' field.

    Returns:
        Tuple of (score in [0.0, 1.0] or None if no ratings present, list of insights).
    """
    valid_ratings: list[float] = []
    for r in reviews:
        val = r.get("rating")
        if val is not None:
            try:
                num = float(val)
                if 1.0 <= num <= 5.0:
                    valid_ratings.append(num)
            except (ValueError, TypeError):
                continue

    if not valid_ratings:
        return None, []

    insights: list[str] = []
    n_r = len(valid_ratings)
    count_5 = sum(1 for r in valid_ratings if round(r) == 5)
    count_1 = sum(1 for r in valid_ratings if round(r) == 1)
    count_mid = sum(1 for r in valid_ratings if round(r) in (2, 3, 4))

    share_5 = count_5 / float(n_r)
    share_1 = count_1 / float(n_r)
    share_mid = count_mid / float(n_r)

    # All 5-star or excessive 5-star skew
    if count_5 == n_r:
        penalty_5 = 1.0
        insights.append("100% of reviews are 5 stars, higher than typical")
    elif share_5 >= RATING_SKEW_5STAR_THRESHOLD:
        penalty_5 = min(1.0, (share_5 - 0.50) / 0.50)
        pct_5 = int(round(share_5 * 100))
        insights.append(f"{pct_5}% of reviews are 5 stars, higher than typical")
    else:
        penalty_5 = 0.0

    # Polarized J-shaped distribution (high 5-stars, some 1-stars, almost no middle)
    if (
        share_5 + share_1 >= RATING_J_SHAPED_THRESHOLD
        and share_5 >= 0.40
        and share_1 >= 0.10
        and share_mid <= 0.20
    ):
        penalty_j = min(1.0, ((share_5 + share_1) - 0.70) / 0.30)
        insights.append("Polarized J-shaped rating distribution detected with few intermediate ratings")
    else:
        penalty_j = 0.0

    total_penalty = min(1.0, max(penalty_5, penalty_j))
    score = float(np.clip(1.0 - total_penalty, 0.0, 1.0))
    return score, insights


def _compute_review_burst_score(reviews: list[dict[str, Any]]) -> tuple[Optional[float], list[str]]:
    """Analyze temporal clustering and burstiness of review submissions.

    Args:
        reviews: List of review dictionaries containing optional 'date' field.

    Returns:
        Tuple of (score in [0.0, 1.0] or None if no valid dates present, list of insights).
    """
    parsed_dates: list[date] = []
    for r in reviews:
        d_val = r.get("date")
        if d_val:
            try:
                dt = pd.to_datetime(d_val).date()
                parsed_dates.append(dt)
            except Exception:
                continue

    if not parsed_dates:
        return None, []

    insights: list[str] = []
    n_d = len(parsed_dates)
    date_counts = Counter(parsed_dates)
    most_common_date, max_single_day_count = date_counts.most_common(1)[0]
    single_day_ratio = max_single_day_count / float(n_d)

    sorted_dates = sorted(parsed_dates)
    date_span_days = (sorted_dates[-1] - sorted_dates[0]).days

    burst_penalty = 0.0

    # Check single-day burst
    if max_single_day_count >= 3 and single_day_ratio >= BURST_MAX_DAY_RATIO:
        burst_penalty = min(1.0, (single_day_ratio - 0.25) / 0.50)
        pct_burst = int(round(single_day_ratio * 100))
        insights.append(
            f"Review burst detected: {max_single_day_count} reviews ({pct_burst}%) posted on the same day ({most_common_date})"
        )
    elif n_d >= 5 and date_span_days <= BURST_WINDOW_DAYS:
        burst_penalty = 0.70
        insights.append(
            f"Review burst detected: all {n_d} reviews posted within a short {date_span_days + 1}-day window"
        )

    score = float(np.clip(1.0 - burst_penalty, 0.0, 1.0))
    return score, insights


def _compute_length_specificity_score(texts: list[str]) -> tuple[float, list[str]]:
    """Evaluate average review length and informational specificity.

    Args:
        texts: List of review text strings.

    Returns:
        Tuple of (score in [0.0, 1.0], list of insights).
    """
    if not texts:
        return 0.5, []

    insights: list[str] = []
    word_counts = [len(t.split()) for t in texts]
    avg_words = float(np.mean(word_counts))

    # Check for numbers / specific details in texts
    digit_ratio = sum(1 for t in texts if any(c.isdigit() for c in t)) / float(len(texts))

    # Score based on target word count and presence of numbers/details
    if avg_words >= SPECIFICITY_TARGET_WORD_COUNT:
        length_ratio = 1.0
    else:
        length_ratio = min(1.0, max(0.1, avg_words / float(SPECIFICITY_TARGET_WORD_COUNT)))

    combined = 0.7 * length_ratio + 0.3 * digit_ratio
    score = float(np.clip(combined, 0.0, 1.0))

    if avg_words < 15.0:
        insights.append(
            f"Reviews are unusually short on average ({int(round(avg_words))} words), lacking specific details"
        )

    return score, insights


def compute_reliability(
    reviews: list[dict[str, Any]],
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compute aggregate reliability score, label, and insights for a collection of reviews.

    Combines:
    - 45% share of non-suspicious reviews (confidence-weighted)
    - 20% near-duplicate penalty: pairwise cosine similarity on TF-IDF above 0.85
    - 15% rating-distribution skew: penalize J-shaped or all-5-star distributions (skipped if missing)
    - 10% review-burst penalty: many reviews on same day / short window (skipped if missing)
    - 10% average length/specificity of reviews

    Labels:
    - 70+ High
    - 40-69 Moderate
    - below 40 Low
    - Fewer than 5 reviews: 'Not enough data'

    Args:
        reviews: List of review dictionaries (with 'id', 'text', optional 'rating', optional 'date').
        results: List of per-review prediction dictionaries (with 'id', 'label', 'confidence').

    Returns:
        Dictionary with:
        {
            "reliability_score": int (0-100),
            "reliability_label": "High" | "Moderate" | "Low" | "Not enough data",
            "insights": list[str]
        }
    """
    total = len(reviews)
    if total == 0:
        return {
            "reliability_score": 0,
            "reliability_label": "Not enough data",
            "insights": ["No reviews provided to evaluate reliability."],
        }

    texts = [str(r.get("text", "")) for r in reviews]
    all_insights: list[str] = []

    # 1. Non-suspicious confidence-weighted share (45%)
    score_non_suspicious = _compute_non_suspicious_score(results)

    # 2. Near-duplicate penalty (20%)
    score_near_dup, dup_insights = _compute_near_duplicate_score(texts)
    all_insights.extend(dup_insights)

    # 3. Rating distribution skew (15%, skipped if ratings missing)
    score_rating, rating_insights = _compute_rating_skew_score(reviews)
    all_insights.extend(rating_insights)

    # 4. Review burst penalty (10%, skipped if dates missing)
    score_burst, burst_insights = _compute_review_burst_score(reviews)
    all_insights.extend(burst_insights)

    # 5. Length and specificity (10%)
    score_length, length_insights = _compute_length_specificity_score(texts)
    all_insights.extend(length_insights)

    # Assemble active weights and normalize
    active_weights: list[float] = [
        RELIABILITY_WEIGHT_NON_SUSPICIOUS,
        RELIABILITY_WEIGHT_NEAR_DUPLICATE,
        RELIABILITY_WEIGHT_LENGTH_SPECIFICITY,
    ]
    active_scores: list[float] = [
        score_non_suspicious,
        score_near_dup,
        score_length,
    ]

    if score_rating is not None:
        active_weights.append(RELIABILITY_WEIGHT_RATING_SKEW)
        active_scores.append(score_rating)

    if score_burst is not None:
        active_weights.append(RELIABILITY_WEIGHT_REVIEW_BURST)
        active_scores.append(score_burst)

    total_weight = sum(active_weights)
    if total_weight > 0:
        weighted_score = sum(s * w for s, w in zip(active_scores, active_weights)) / total_weight
    else:
        weighted_score = 0.5

    reliability_score = int(round(np.clip(weighted_score * 100.0, 0.0, 100.0)))

    # Determine label
    if total < MIN_REVIEWS_FOR_RELIABILITY:
        reliability_label = "Not enough data"
        all_insights.append(
            f"Fewer than {MIN_REVIEWS_FOR_RELIABILITY} reviews analyzed (minimum {MIN_REVIEWS_FOR_RELIABILITY} required for reliability label)."
        )
    elif reliability_score >= RELIABILITY_HIGH_THRESHOLD:
        reliability_label = "High"
    elif reliability_score >= RELIABILITY_MODERATE_THRESHOLD:
        reliability_label = "Moderate"
    else:
        reliability_label = "Low"

    return {
        "reliability_score": reliability_score,
        "reliability_label": reliability_label,
        "insights": all_insights,
    }
