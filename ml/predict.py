"""Inference module for running end-to-end review predictions and reliability scoring.

Provides the ModelService class for thread-safe batch and individual review analysis,
producing API-compliant results and summaries according to the PRD specification.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional

import joblib
from sklearn.pipeline import Pipeline

from ml.config import (
    MAX_REVIEW_TEXT_LENGTH,
    MAX_REVIEWS_PER_CALL,
    MODEL_PATH,
)
from ml.explain import classify_probability, explain_review
from ml.reliability import compute_reliability

logger = logging.getLogger(__name__)

# 10 diverse sample reviews for demonstration and sanity checking
DEMO_REVIEWS: list[dict[str, Any]] = [
    {
        "id": "r1",
        "text": "I bought this coffee maker 3 months ago. It brews quickly in under 4 minutes and the temperature is consistently around 195 degrees. Solid construction and easy to clean.",
        "rating": 5,
        "date": "2026-08-15",
    },
    {
        "id": "r2",
        "text": "AMAZING PRODUCT!! BEST PURCHASE EVER!! 100% RECOMMEND TO EVERYONE DO NOT HESITATE BUY IT NOW WOW WOW WOW!!!",
        "rating": 5,
        "date": "2026-10-01",
    },
    {
        "id": "r3",
        "text": "AMAZING PRODUCT!! BEST PURCHASE EVER!! 100% RECOMMEND TO EVERYONE DO NOT HESITATE BUY IT NOW WOW WOW WOW!!!",
        "rating": 5,
        "date": "2026-10-01",
    },
    {
        "id": "r4",
        "text": "Good headphones for the price. The noise cancellation works fine in an office setting, though bass could be a bit punchier. Battery lasts around 18 hours.",
        "rating": 4,
        "date": "2026-09-02",
    },
    {
        "id": "r5",
        "text": "Terrible customer support. The package arrived with a dented box and the left earbud stopped charging after 2 days. Had to return it.",
        "rating": 1,
        "date": "2026-09-10",
    },
    {
        "id": "r6",
        "text": "AMAZING PRODUCT!! BEST PURCHASE EVER!! 100% RECOMMEND TO EVERYONE DO NOT HESITATE BUY IT NOW WOW WOW WOW!!!",
        "rating": 5,
        "date": "2026-10-01",
    },
    {
        "id": "r7",
        "text": "Decent fit for my 15-inch laptop. Zipper feels a bit stiff initially but loosened up after a week. Water resistance handled light rain without issues.",
        "rating": 4,
        "date": "2026-08-20",
    },
    {
        "id": "r8",
        "text": "INCREDIBLE QUALITY TOP NOTCH FIVE STARS BEST THING EVER BUY BUY BUY!!!",
        "rating": 5,
        "date": "2026-10-01",
    },
    {
        "id": "r9",
        "text": "Works as described. Nothing spectacular, just does the job properly. Good value for $25.",
        "rating": 3,
        "date": "2026-08-28",
    },
    {
        "id": "r10",
        "text": "The display is crisp with 144Hz refresh rate, but colors need calibration out of the box. Used for 2 weeks for both gaming and video editing.",
        "rating": 4,
        "date": "2026-09-14",
    },
]


def sort_for_user_inspection(
    results: list[dict[str, Any]],
    reviews: Optional[list[dict[str, Any]]] = None,
) -> list[dict[str, Any]]:
    """Sort review results for user inspection, prioritizing media-rich reviews.

    Media-rich reviews (with customer images or video) appear first, followed by text-only
    reviews. Within each group, suspicious reviews appear first, ordered by confidence descending.

    Args:
        results: List of prediction result dictionaries.
        reviews: Optional list of review dictionaries to check for media presence.

    Returns:
        New sorted list of result dictionaries.
    """
    media_ids: set[str] = set()
    if reviews:
        for r in reviews:
            if bool(r.get("has_images") or r.get("has_video")):
                media_ids.add(str(r.get("id", "")))

    def _sort_key(item: dict[str, Any]) -> tuple[int, int, float]:
        item_id = str(item.get("id", ""))
        has_media = item_id in media_ids or any(
            isinstance(rsn, dict) and rsn.get("type") == "media"
            for rsn in item.get("reasons", [])
        )
        is_suspicious = 1 if item.get("label") == "suspicious" else 0
        conf = float(item.get("confidence", 0.0))
        return (1 if has_media else 0, is_suspicious, conf)

    return sorted(results, key=_sort_key, reverse=True)


class ModelService:
    """Thread-safe inference service for ReviewLens.

    Loads the trained model bundle artifact once upon instantiation and exposes
    the `analyze()` method for evaluating sets of reviews.
    """

    sort_for_inspection = staticmethod(sort_for_user_inspection)
    sort_for_user_inspection = staticmethod(sort_for_user_inspection)

    def __init__(self, model_path: Path | str = MODEL_PATH) -> None:
        """Initialize ModelService and load the trained pipeline artifact.

        Args:
            model_path: Path to serialized model artifact (.joblib). Defaults to MODEL_PATH.

        Raises:
            FileNotFoundError: If model artifact is missing from disk.
            RuntimeError: If model artifact fails to load.
        """
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at '{self.model_path}'. "
                "Please run model training first (e.g. `python -m ml.train` or `make train`)."
            )

        try:
            logger.info("Loading model artifact from %s", self.model_path)
            self.pipeline: Pipeline = joblib.load(self.model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load model artifact from '{self.model_path}': {e}") from e

        self._lock = threading.Lock()

    def analyze(
        self,
        reviews: list[dict[str, Any]],
        seller_info: Optional[Any] = None,
        sort_for_inspection: bool = False,
    ) -> dict[str, Any]:
        """Analyze a collection of reviews and produce prediction results with summary.

        Validates input reviews, generates missing IDs, truncates overly long texts,
        predicts deceptive probabilities, extracts explainability reasons,
        computes product-level reliability, and returns the response structure
        matching PRD.md Section 9 API contract.

        Args:
            reviews: List of review dictionaries containing 'text', and optional 'id',
                     'rating', 'date', 'has_images', and 'has_video'.
            seller_info: Optional seller information or rating out of 5 for seller confidence.
            sort_for_inspection: If True, prioritizes media-rich reviews in results ordering.

        Returns:
            Dictionary matching API contract:
            {
                "results": [
                    {
                        "id": str,
                        "label": "suspicious" | "genuine" | "uncertain",
                        "confidence": float,
                        "reasons": [{"type": str, "text": str}]
                    }
                ],
                "summary": {
                    "total": int,
                    "suspicious_count": int,
                    "suspicious_pct": float,
                    "reliability_score": int,
                    "reliability_label": str,
                    "insights": list[str],
                    "product_genuineness_score": float,
                    "product_genuineness_label": str,
                    "media_review_count": int,
                    "media_review_pct": float,
                    "volume_risk": dict,
                    "seller_confidence": Optional[float]
                }
            }

        Raises:
            TypeError: If input is not a list or an item is not a dictionary.
            ValueError: If reviews exceed max allowed limit or any review has empty text.
        """
        if not isinstance(reviews, list):
            raise TypeError(f"Input 'reviews' must be a list, got {type(reviews).__name__}")

        if len(reviews) > MAX_REVIEWS_PER_CALL:
            raise ValueError(
                f"Batch size of {len(reviews)} reviews exceeds the maximum limit of {MAX_REVIEWS_PER_CALL} reviews per call."
            )

        if not reviews:
            reliability = compute_reliability([], [], seller_info=seller_info)
            return {
                "results": [],
                "summary": {
                    "total": 0,
                    "suspicious_count": 0,
                    "suspicious_pct": 0.0,
                    "reliability_score": reliability["reliability_score"],
                    "reliability_label": reliability["reliability_label"],
                    "insights": reliability["insights"],
                    "product_genuineness_score": reliability["product_genuineness_score"],
                    "product_genuineness_label": reliability["product_genuineness_label"],
                    "media_review_count": reliability["media_review_count"],
                    "media_review_pct": reliability["media_review_pct"],
                    "volume_risk": reliability["volume_risk"],
                    "seller_confidence": reliability["seller_confidence"],
                },
            }

        cleaned_reviews: list[dict[str, Any]] = []
        for idx, rev in enumerate(reviews):
            if not isinstance(rev, dict):
                raise TypeError(f"Review at index {idx} must be a dictionary, got {type(rev).__name__}")

            raw_id = rev.get("id")
            if raw_id is not None and str(raw_id).strip():
                review_id = str(raw_id).strip()
            else:
                review_id = f"r{idx + 1}"

            raw_text = rev.get("text")
            if raw_text is None or not isinstance(raw_text, str) or not raw_text.strip():
                raise ValueError(f"Review '{review_id}' contains empty or missing text.")

            clean_text = raw_text.strip()
            if len(clean_text) > MAX_REVIEW_TEXT_LENGTH:
                clean_text = clean_text[:MAX_REVIEW_TEXT_LENGTH]

            cleaned_item = dict(rev)
            cleaned_item["id"] = review_id
            cleaned_item["text"] = clean_text
            cleaned_item["has_images"] = bool(rev.get("has_images", False))
            cleaned_item["has_video"] = bool(rev.get("has_video", False))
            cleaned_reviews.append(cleaned_item)

        # Batch prediction with thread safety
        texts = [r["text"] for r in cleaned_reviews]
        with self._lock:
            if hasattr(self.pipeline, "predict_proba"):
                probs = self.pipeline.predict_proba(texts)
            else:
                preds = self.pipeline.predict(texts)
                probs = np.column_stack([1.0 - preds, preds])

        results: list[dict[str, Any]] = []
        for idx, cleaned_item in enumerate(cleaned_reviews):
            p_deceptive = float(probs[idx][1]) if probs.shape[1] > 1 else float(probs[idx][0])

            # Indian consumer pattern adjustment:
            # Short and sweet reviews (2 to 25 words) that are natural (not repetitive spam or all-caps)
            # are authentic and common. If p_deceptive is borderline due to brevity, notch up the trust.
            raw_t = cleaned_item["text"].strip()
            w_cnt = len(raw_t.split())
            t_lower = raw_t.lower()
            has_hype_spam = any(
                spam in t_lower
                for spam in ["wow wow", "buy buy", "must buy must buy", "best ever best ever", "100% recommend"]
            )
            is_all_caps = len(raw_t) > 6 and raw_t.isupper()

            if 2 <= w_cnt <= 25 and not has_hype_spam and not is_all_caps:
                if p_deceptive < 0.65:
                    p_deceptive = max(0.0, p_deceptive * 0.82)

            label, confidence = classify_probability(
                p_deceptive,
                deceptive_label="suspicious",
                genuine_label="genuine",
                uncertain_label="uncertain",
            )

            with self._lock:
                raw_reasons = explain_review(
                    cleaned_item["text"],
                    self.pipeline,
                    top_k=3,
                    has_images=cleaned_item["has_images"],
                    has_video=cleaned_item["has_video"],
                )

            reasons = [{"type": reas["type"], "text": reas["text"]} for reas in raw_reasons]

            results.append({
                "id": cleaned_item["id"],
                "label": label,
                "confidence": confidence,
                "reasons": reasons,
            })

        total = len(cleaned_reviews)
        suspicious_count = sum(1 for res in results if res["label"] == "suspicious")
        suspicious_pct = round((suspicious_count / float(total)) * 100.0, 1) if total > 0 else 0.0

        reliability = compute_reliability(cleaned_reviews, results, seller_info=seller_info)

        summary = {
            "total": total,
            "suspicious_count": suspicious_count,
            "suspicious_pct": suspicious_pct,
            "reliability_score": reliability["reliability_score"],
            "reliability_label": reliability["reliability_label"],
            "insights": reliability["insights"],
            "product_genuineness_score": reliability["product_genuineness_score"],
            "product_genuineness_label": reliability["product_genuineness_label"],
            "media_review_count": reliability["media_review_count"],
            "media_review_pct": reliability["media_review_pct"],
            "volume_risk": reliability["volume_risk"],
            "seller_confidence": reliability["seller_confidence"],
        }

        if sort_for_inspection:
            results = sort_for_user_inspection(results, cleaned_reviews)

        return {
            "results": results,
            "summary": summary,
        }


def main() -> None:
    """CLI entrypoint for running demo inference."""
    parser = argparse.ArgumentParser(description="ReviewLens Inference Service")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Analyze 10 hard-coded sample reviews and print the resulting JSON.",
    )
    args = parser.parse_args()

    if args.demo:
        service = ModelService()
        output = service.analyze(DEMO_REVIEWS)
        print(json.dumps(output, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
