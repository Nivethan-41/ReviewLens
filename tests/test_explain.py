"""Unit and integration tests for ReviewLens explanation and calibration modules."""

from __future__ import annotations

import json
from pathlib import Path
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ml.config import (
    CALIBRATION_CURVE_PATH,
    MODEL_PATH,
    UNCERTAIN_LOWER,
    UNCERTAIN_UPPER,
)
from ml.explain import (
    classify_probability,
    compute_stylometric_stats,
    explain_prediction,
    explain_review,
    get_linear_coefficients,
)
from ml.features import build_feature_pipeline
from ml.train import calibrate_classifier, plot_calibration_curve


@pytest.fixture
def sample_10_reviews() -> list[str]:
    """Provide 10 diverse sample reviews representing different classes and styles."""
    return [
        # 1. Detailed genuine review with numbers and balanced experience
        "I bought this blender 3 months ago. The 700-watt motor blends frozen fruits smoothly, although it is rather loud on high speed.",
        # 2. Extreme deceptive hype with superlatives and punctuation
        "AMAZING UNBELIEVABLE BEST EVER INCREDIBLE 100% PERFECT MUST BUY NOW WOW!!!",
        # 3. Concise genuine review with specifics
        "Ordered on Tuesday, arrived Thursday. The blue color matches the picture and the fit is true to size for my 10.5 shoes.",
        # 4. Deceptive generic praise
        "Highly recommend this miracle product to everyone! Best ever experience, changed my life completely!",
        # 5. Short and generic text
        "Good product. Nice.",
        # 6. Deceptive all-caps urgency
        "DO NOT WAIT BUY THIS RIGHT NOW!! BEST PURCHASE OF MY ENTIRE LIFE GUARANTEED!!",
        # 7. Genuine review with specific numeric measurements
        "The battery lasts about 6 hours instead of the advertised 8 hours. Sound quality is decent for $35 earbuds.",
        # 8. Borderline review
        "A pretty decent item that works as described. Would probably recommend to friends.",
        # 9. Superlatives without factual details
        "Top quality best of the best! Outstanding perfection in every way imaginable!",
        # 10. Genuine personal experience in the rain
        "My wife and I used this tent on a 2-day camping trip in the rain. Kept us completely dry and packed up easily.",
    ]


@pytest.fixture
def trained_bundle() -> Pipeline:
    """Load or build a fitted model bundle with stylometric stats and calibration."""
    if MODEL_PATH.exists():
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            pass

    # Fallback to creating a synthetic bundle if model.joblib does not exist
    texts = [
        "Normal honest review with realistic details and 5 stars for durability.",
        "Good quality product, arrived in 2 days, fits well.",
        "Average item, nothing special but works fine for the low price.",
        "Decent purchase, 4 out of 5 stars based on 3 weeks of daily use.",
        "AMAZING BEST EVER INCREDIBLE 100% PERFECT MUST BUY NOW WOW!!!",
        "UNBELIEVABLE MIRACLE PRODUCT BEST EXPERIENCE EVER BUY IMMEDIATELY!",
        "BEST ITEM IN THE ENTIRE WORLD AMAZING INCREDIBLE TOP RATED!",
        "BUY NOW DO NOT HESITATE BEST PURCHASE EVER FANTASTIC LUXURY!",
    ]
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    features = build_feature_pipeline()
    X_trans = features.fit_transform(texts)

    clf = LogisticRegression(random_state=42, solver="liblinear")
    cal, method, metrics, _ = calibrate_classifier(
        chosen_classifier=clf,
        feature_pipeline=features,
        X_train=texts,
        y_train=labels,
        X_val=texts,
        y_val=labels,
        random_seed=42,
    )

    pipe = Pipeline([
        ("features", features),
        ("classifier", cal),
    ])
    pipe.stylometric_stats_ = compute_stylometric_stats(texts, labels)
    pipe.surrogate_model_ = clf.fit(X_trans, labels)
    pipe.bundle_ = {
        "pipeline": pipe,
        "stylometric_stats": pipe.stylometric_stats_,
        "surrogate_model": pipe.surrogate_model_,
        "calibration_method": method,
        "chosen_model": "LogisticRegression",
    }
    return pipe


def test_calibrated_probabilities_in_range_0_1(
    trained_bundle: Pipeline, sample_10_reviews: list[str]
):
    """Verify that calibrated predict_proba always outputs valid probabilities in [0.0, 1.0]."""
    probs = trained_bundle.predict_proba(sample_10_reviews)

    assert probs.shape == (len(sample_10_reviews), 2)
    assert np.all(probs >= 0.0), "Calibrated probabilities must be >= 0.0"
    assert np.all(probs <= 1.0), "Calibrated probabilities must be <= 1.0"
    # Row probabilities must sum to 1.0
    row_sums = np.sum(probs, axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)


def test_uncertain_band_boundaries():
    """Verify that the uncertainty band behaves exactly as defined at the boundaries.

    Uncertainty band: between 0.45 and 0.55 inclusive -> label 'uncertain'.
    Confidence: probability of predicted class in [0.5, 1.0].
    """
    # Exact lower boundary: 0.45
    label_045, conf_045 = classify_probability(0.45)
    assert label_045 == "uncertain"
    assert conf_045 == 0.55

    # Center: 0.50
    label_050, conf_050 = classify_probability(0.50)
    assert label_050 == "uncertain"
    assert conf_050 == 0.50

    # Exact upper boundary: 0.55
    label_055, conf_055 = classify_probability(0.55)
    assert label_055 == "uncertain"
    assert conf_055 == 0.55

    # Just below lower boundary: 0.449 -> genuine
    label_0449, conf_0449 = classify_probability(0.449)
    assert label_0449 == "genuine"
    assert round(conf_0449, 4) == 0.551

    # Just above upper boundary: 0.551 -> deceptive
    label_0551, conf_0551 = classify_probability(0.551)
    assert label_0551 == "deceptive"
    assert round(conf_0551, 4) == 0.551

    # Extreme boundaries
    label_00, conf_00 = classify_probability(0.0)
    assert label_00 == "genuine"
    assert conf_00 == 1.0

    label_10, conf_10 = classify_probability(1.0)
    assert label_10 == "deceptive"
    assert conf_10 == 1.0


def test_reasons_non_empty_for_10_sample_reviews(
    trained_bundle: Pipeline, sample_10_reviews: list[str]
):
    """Verify that explain_review returns at least one non-empty reason for 10 sample reviews."""
    assert len(sample_10_reviews) == 10

    for review in sample_10_reviews:
        reasons = explain_review(review, trained_bundle, top_k=3)
        assert isinstance(reasons, list)
        assert len(reasons) >= 1, f"Reasons list was empty for review: '{review}'"
        assert len(reasons) <= 3, f"Reasons exceeded top_k=3 for review: '{review}'"

        for r in reasons:
            assert "type" in r
            assert "text" in r
            assert "weight" in r
            assert r["type"] in {"language", "style", "pattern"}
            assert isinstance(r["text"], str)
            assert len(r["text"].strip()) > 0
            assert isinstance(r["weight"], float)


def test_reasons_never_contain_raw_feature_names(
    trained_bundle: Pipeline, sample_10_reviews: list[str]
):
    """Verify that reason text never leaks internal pipeline feature names like 'tfidf__'."""
    forbidden_substrings = [
        "tfidf__",
        "word_tfidf__",
        "char_tfidf__",
        "stylometric__",
        "__",
    ]

    for review in sample_10_reviews:
        reasons = explain_review(review, trained_bundle, top_k=3)
        for r in reasons:
            text = r["text"]
            for forbidden in forbidden_substrings:
                assert forbidden not in text, (
                    f"Reason text '{text}' leaked internal feature name '{forbidden}'"
                )


def test_genuine_clean_review_fallback_reason(trained_bundle: Pipeline):
    """Verify genuine reviews return genuine phrases or the fallback reason."""
    # 1. Genuine review with common genuine vocabulary
    genuine_review = (
        "I bought this blender 3 months ago. The 700-watt motor blends frozen fruits smoothly, "
        "although it is rather loud on high speed."
    )
    reasons = explain_review(genuine_review, trained_bundle, top_k=3)
    assert len(reasons) >= 1
    texts = [r["text"] for r in reasons]
    assert any("genuine" in t.lower() for t in texts)

    # 2. Genuine-looking review where no specific features trigger
    class MockGenuinePipeline:
        def predict_proba(self, X):
            return np.array([[0.95, 0.05]])

        def predict(self, X):
            return np.array([0])

        @property
        def named_steps(self):
            return {}

    mock_bundle = {"pipeline": MockGenuinePipeline()}
    fallback_reasons = explain_review("Ordinary neutral text.", mock_bundle, top_k=3)
    assert len(fallback_reasons) == 1
    assert fallback_reasons[0]["type"] == "pattern"
    assert fallback_reasons[0]["text"] == "No strong deceptive signals found"
    assert fallback_reasons[0]["weight"] == 0.0


def test_stylometric_stats_stored_in_bundle_needs_no_training_data(
    trained_bundle: Pipeline,
):
    """Verify explain_review operates strictly using stats in the bundle with no training data."""
    # Ensure stylometric_stats_ is attached and structured
    stats = getattr(trained_bundle, "stylometric_stats_", None)
    if stats is None and hasattr(trained_bundle, "bundle_"):
        stats = trained_bundle.bundle_.get("stylometric_stats")

    assert stats is not None, "stylometric_stats was not attached to bundle"
    assert "genuine" in stats
    assert "word_count" in stats["genuine"]
    assert "mean" in stats["genuine"]["word_count"]
    assert "std" in stats["genuine"]["word_count"]

    # Passing solely the bundle must succeed without any DataFrame or training set
    result = explain_prediction("Short test text.", trained_bundle)
    assert "label" in result
    assert "confidence" in result
    assert "reasons" in result
    assert len(result["reasons"]) >= 1


def test_surrogate_fallback_for_non_linear_model():
    """Verify non-linear classifier falls back to surrogate LogisticRegression coefficients."""
    texts = [
        "Normal genuine review with common everyday vocabulary.",
        "Another genuine item purchased last week with standard shipping.",
        "AMAZING BEST EVER INCREDIBLE 100% PERFECT BUY NOW WOW!",
        "UNBELIEVABLE MIRACLE BEST PRODUCT EVER WOW INCREDIBLE!",
    ]
    labels = np.array([0, 0, 1, 1])

    features = build_feature_pipeline()
    X_trans = features.fit_transform(texts)

    # Train a non-linear RandomForestClassifier (has no coef_ attribute)
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X_trans, labels)

    pipe = Pipeline([
        ("features", features),
        ("classifier", rf),
    ])

    # Surrogate LogisticRegression
    surrogate = LogisticRegression(random_state=42, solver="liblinear")
    surrogate.fit(X_trans, labels)

    # With surrogate attached
    pipe.surrogate_model_ = surrogate
    pipe.stylometric_stats_ = compute_stylometric_stats(texts, labels)

    coef, is_surrogate = get_linear_coefficients(pipe)
    assert is_surrogate is True, "Should have fallen back to surrogate model"
    assert isinstance(coef, np.ndarray)
    assert len(coef) == X_trans.shape[1]

    # Test that explain_review succeeds using the surrogate
    reasons = explain_review("AMAZING BEST EVER INCREDIBLE!", pipe)
    assert len(reasons) >= 1
    for r in reasons:
        assert "__" not in r["text"]


def test_calibration_curve_plot_helper(tmp_path: Path):
    """Verify plot_calibration_curve generates a valid, non-empty PNG file."""
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    plot_target = tmp_path / "test_calibration.png"

    saved_path = plot_calibration_curve(
        y_true=y_true,
        y_prob=y_prob,
        method_name="isotonic",
        brier_score=0.035,
        save_path=plot_target,
    )

    assert saved_path.exists(), "Calibration plot was not created"
    assert saved_path.stat().st_size > 1000, "Calibration plot file is too small"
