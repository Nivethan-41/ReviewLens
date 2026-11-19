"""Model explanation module providing human-interpretable reasons for predictions.

Provides methods for computing stylometric baseline statistics from training data,
extracting language-based and style-based explanations from linear or surrogate models,
calibrated probability classification with an uncertainty band, and generating
plain-English reasons for review predictions.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ml.config import (
    MODEL_PATH,
    UNCERTAIN_LOWER,
    UNCERTAIN_UPPER,
)
from ml.features import (
    STYLOMETRIC_FEATURE_NAMES,
    StylometricFeatures,
    _to_text_list,
)

logger = logging.getLogger(__name__)


def compute_stylometric_stats(
    X_train: pd.Series | list[str],
    y_train: pd.Series | np.ndarray,
) -> dict[str, dict[str, dict[str, float]]]:
    """Compute mean and std of stylometric features on training set.

    Computes statistics separately for genuine reviews (class 0) and all reviews
    so explain_review requires no raw training data at inference time.

    Args:
        X_train: Training text series or list of strings.
        y_train: Training labels (0 = genuine, 1 = deceptive).

    Returns:
        Dictionary mapping class identifier ('genuine', 'all') to feature statistics:
        {class_name: {feature_name: {'mean': float, 'std': float}}}.
    """
    texts = _to_text_list(X_train)
    y_arr = np.array(y_train, dtype=int)

    stylo = StylometricFeatures()
    features_all = stylo.transform(texts)

    genuine_mask = y_arr == 0
    if np.any(genuine_mask):
        features_genuine = features_all[genuine_mask]
    else:
        features_genuine = features_all

    feat_names = stylo.feature_names_
    genuine_stats: dict[str, dict[str, float]] = {}
    all_stats: dict[str, dict[str, float]] = {}

    count_features = {
        "word_count",
        "char_count",
        "exclamation_count",
        "superlative_count",
        "sentence_count",
    }
    avg_features = {"avg_word_length", "avg_sentence_length"}

    for idx, name in enumerate(feat_names):
        if name in count_features:
            min_std = 1.0
        elif name in avg_features:
            min_std = 0.5
        else:
            min_std = 0.02

        gen_vals = features_genuine[:, idx]
        gen_mean = float(np.mean(gen_vals))
        gen_std = float(np.std(gen_vals))
        if gen_std < min_std:
            gen_std = min_std
        genuine_stats[name] = {
            "mean": round(gen_mean, 6),
            "std": round(gen_std, 6),
        }

        all_vals = features_all[:, idx]
        all_mean = float(np.mean(all_vals))
        all_std = float(np.std(all_vals))
        if all_std < min_std:
            all_std = min_std
        all_stats[name] = {
            "mean": round(all_mean, 6),
            "std": round(all_std, 6),
        }

    return {
        "genuine": genuine_stats,
        "all": all_stats,
    }


def classify_probability(
    p_deceptive: float,
    threshold_lower: float = UNCERTAIN_LOWER,
    threshold_upper: float = UNCERTAIN_UPPER,
    deceptive_label: str = "deceptive",
    genuine_label: str = "genuine",
    uncertain_label: str = "uncertain",
) -> tuple[str, float]:
    """Determine class label and confidence score from calibrated deceptive probability.

    Defines confidence as the probability of the predicted class in [0.5, 1.0].
    Reviews with probability between threshold_lower (0.45) and threshold_upper (0.55)
    inclusive receive the label 'uncertain'.

    Args:
        p_deceptive: Calibrated probability of being deceptive (class 1), in [0.0, 1.0].
        threshold_lower: Lower bound of uncertainty band. Defaults to 0.45.
        threshold_upper: Upper bound of uncertainty band. Defaults to 0.55.
        deceptive_label: Label for deceptive predictions. Defaults to 'deceptive'.
        genuine_label: Label for genuine predictions. Defaults to 'genuine'.
        uncertain_label: Label for uncertain predictions. Defaults to 'uncertain'.

    Returns:
        Tuple of (label, confidence), where confidence is in [0.5, 1.0].
    """
    p_clean = float(np.clip(p_deceptive, 0.0, 1.0))
    confidence = float(max(p_clean, 1.0 - p_clean))

    if threshold_lower <= p_clean <= threshold_upper:
        label = uncertain_label
    elif p_clean > threshold_upper:
        label = deceptive_label
    else:
        label = genuine_label

    return label, round(confidence, 4)


def get_linear_coefficients(
    pipeline: Pipeline,
    surrogate_model: Optional[Any] = None,
) -> tuple[np.ndarray, bool]:
    """Extract linear model coefficients, falling back to surrogate LogisticRegression if needed.

    If the chosen model is not linear (such as RandomForestClassifier or HistGradientBoosting),
    this function falls back to the LogisticRegression coefficients from a surrogate model
    trained on the same features, ensuring that feature importance explanations remain available.

    Args:
        pipeline: Fitted scikit-learn Pipeline.
        surrogate_model: Optional pre-fitted surrogate LogisticRegression estimator.

    Returns:
        Tuple of (coef_1d_array, is_surrogate_flag).

    Raises:
        ValueError: If neither a linear classifier nor a valid surrogate model is found.
    """
    classifier = pipeline.named_steps.get("classifier")

    # 1. CalibratedClassifierCV wrapping a linear base estimator
    if hasattr(classifier, "calibrated_classifiers_"):
        sub_clfs = classifier.calibrated_classifiers_
        if sub_clfs and hasattr(sub_clfs[0], "estimator"):
            base_estimator = sub_clfs[0].estimator
            if hasattr(base_estimator, "coef_"):
                coef_list = [c.estimator.coef_ for c in sub_clfs if hasattr(c.estimator, "coef_")]
                if coef_list:
                    avg_coef = np.mean(coef_list, axis=0)
                    if avg_coef.ndim > 1:
                        avg_coef = avg_coef[0]
                    return avg_coef, False

    # 2. Direct linear classifier with coef_ attribute
    if hasattr(classifier, "coef_"):
        coef = classifier.coef_
        if coef.ndim > 1:
            coef = coef[0]
        return coef, False

    # 3. Fallback to surrogate LogisticRegression model
    surrogate = surrogate_model or getattr(pipeline, "surrogate_model_", None)
    if surrogate is not None and hasattr(surrogate, "coef_"):
        coef = surrogate.coef_
        if coef.ndim > 1:
            coef = coef[0]
        return coef, True

    # 4. If surrogate is inside pipeline.bundle_
    bundle = getattr(pipeline, "bundle_", None)
    if isinstance(bundle, dict):
        bundle_surrogate = bundle.get("surrogate_model")
        if bundle_surrogate is not None and hasattr(bundle_surrogate, "coef_"):
            coef = bundle_surrogate.coef_
            if coef.ndim > 1:
                coef = coef[0]
            return coef, True

    raise ValueError(
        "Classifier is non-linear and no trained LogisticRegression surrogate model was provided."
    )


def _extract_language_reason(
    text: str,
    pipeline: Pipeline,
    coef: np.ndarray,
    predicted_class: int,
    max_phrases: int = 3,
) -> Optional[dict[str, Any]]:
    """Generate a human-readable language reason from word TF-IDF contributions.

    Args:
        text: Review text.
        pipeline: Trained Pipeline containing feature extraction step.
        coef: Linear model coefficients corresponding to feature union.
        predicted_class: 1 for deceptive, 0 for genuine.
        max_phrases: Maximum number of n-grams to mention in sentence.

    Returns:
        Reason dictionary or None if no contributing phrases were found.
    """
    feature_pipeline = pipeline.named_steps.get("features")
    if feature_pipeline is None:
        return None

    # Transform single review to get feature vector
    x_feat = feature_pipeline.transform([text])
    if hasattr(x_feat, "toarray"):
        x_vec = x_feat.toarray()[0]
    else:
        x_vec = np.asarray(x_feat)[0]

    feature_names = feature_pipeline.get_feature_names_out()
    if len(feature_names) != len(coef):
        return None

    # Filter for word TF-IDF features present in the review (value > 0)
    word_prefix = "word_tfidf__"
    word_contributions: list[tuple[str, float]] = []

    for idx, (name, val) in enumerate(zip(feature_names, x_vec)):
        if val <= 0.0 or not name.startswith(word_prefix):
            continue

        raw_phrase = name[len(word_prefix):].strip()
        weight = float(coef[idx])

        # For deceptive (class 1), positive coefficient pushes towards deceptive
        # For genuine (class 0), negative coefficient pushes towards genuine
        if predicted_class == 1:
            contribution = weight * float(val)
        else:
            contribution = -weight * float(val)

        if contribution > 0.0001:
            word_contributions.append((raw_phrase, contribution))

    if not word_contributions:
        return None

    # Sort descending by contribution
    word_contributions.sort(key=lambda item: item[1], reverse=True)
    top_items = word_contributions[:max_phrases]
    top_phrases = [phrase for phrase, _ in top_items]
    total_weight = sum(contrib for _, contrib in top_items)

    formatted_phrases = ", ".join(f"'{p}'" for p in top_phrases)

    if predicted_class == 1:
        sentence = f"Contains phrases common in deceptive reviews: {formatted_phrases}"
    else:
        sentence = f"Contains phrases typical of genuine reviews: {formatted_phrases}"

    return {
        "type": "language",
        "text": sentence,
        "weight": round(float(total_weight), 4),
    }


def _extract_style_reasons(
    text: str,
    genuine_stats: dict[str, dict[str, float]],
    predicted_class: int,
) -> list[dict[str, Any]]:
    """Generate human-readable stylometric reasons by comparing with genuine-class stats.

    Compares the review's stylometric values with the genuine-class mean and std
    from training (z-score above 1.5), then formats plain templates.

    Args:
        text: Review text.
        genuine_stats: Mapping of feature names to {'mean': float, 'std': float}.
        predicted_class: 1 for deceptive, 0 for genuine.

    Returns:
        List of style-based reason dictionaries.
    """
    reasons: list[dict[str, Any]] = []
    if not genuine_stats:
        return reasons

    # Compute raw stylometrics for the review
    extractor = StylometricFeatures()
    values = extractor._compute_single(text)
    feature_names = extractor.feature_names_
    val_map = dict(zip(feature_names, values))

    def get_z(name: str) -> float:
        if name not in genuine_stats or name not in val_map:
            return 0.0
        m = genuine_stats[name]["mean"]
        s = genuine_stats[name]["std"]
        if s < 1e-6:
            return 0.0
        return (val_map[name] - m) / s

    # 1. Unusually many exclamation marks
    z_excl_cnt = get_z("exclamation_count")
    z_excl_rat = get_z("exclamation_ratio")
    if val_map.get("exclamation_count", 0.0) >= 2.0 and (z_excl_cnt > 1.5 or z_excl_rat > 1.5):
        weight = min(20.0, max(z_excl_cnt, z_excl_rat))
        reasons.append({
            "type": "style",
            "text": "Unusually many exclamation marks",
            "weight": round(float(weight), 4),
        })

    # Style checks 2 & 3 specifically indicate deceptive review patterns
    if predicted_class == 1:
        # 2. Very short and generic
        z_words = get_z("word_count")
        z_chars = get_z("char_count")
        if (z_words < -1.5 or z_chars < -1.5) and val_map.get("word_count", 0.0) < 20.0:
            weight = min(20.0, abs(z_words))
            reasons.append({
                "type": "style",
                "text": "Very short and generic",
                "weight": round(float(weight), 4),
            })

        # 3. Almost no specific details (numbers or first-person experience)
        val_digits = val_map.get("digit_ratio", 0.0)
        val_fp = val_map.get("first_person_ratio", 0.0)
        z_digits = get_z("digit_ratio")
        z_fp = get_z("first_person_ratio")
        if (val_digits == 0.0 or z_digits < -1.0) and (val_fp == 0.0 or z_fp < -1.0):
            if z_digits < -1.5 or z_fp < -1.5 or (val_digits == 0.0 and val_fp == 0.0):
                weight = min(20.0, abs(min(z_digits, z_fp)))
                reasons.append({
                    "type": "style",
                    "text": "Almost no specific details (numbers or first-person experience)",
                    "weight": round(float(weight), 4),
                })

    # 4. Superlatives and extreme praise words
    z_super = get_z("superlative_count")
    if val_map.get("superlative_count", 0.0) >= 2.0 and z_super > 1.5:
        reasons.append({
            "type": "style",
            "text": "Unusually high number of superlatives and extreme praise words",
            "weight": round(float(min(20.0, z_super)), 4),
        })

    # 5. Uppercase / all-caps words
    z_upper = get_z("uppercase_word_ratio")
    if val_map.get("uppercase_word_ratio", 0.0) > 0.15 and z_upper > 1.5:
        reasons.append({
            "type": "style",
            "text": "Unusually high proportion of ALL-CAPS words",
            "weight": round(float(min(20.0, z_upper)), 4),
        })

    # 6. Repeated words
    z_rep = get_z("repeated_word_ratio")
    if val_map.get("repeated_word_ratio", 0.0) > 0.25 and z_rep > 1.5:
        reasons.append({
            "type": "style",
            "text": "Unusually high proportion of repeated words",
            "weight": round(float(min(20.0, z_rep)), 4),
        })

    # 7. Second-person pronoun addressing
    z_sp = get_z("second_person_ratio")
    if val_map.get("second_person_ratio", 0.0) > 0.05 and z_sp > 1.5:
        reasons.append({
            "type": "style",
            "text": "Unusually frequent direct address to reader ('you', 'your')",
            "weight": round(float(z_sp), 4),
        })

    # 8. Question marks
    z_qm = get_z("question_mark_ratio")
    if val_map.get("question_mark_ratio", 0.0) > 0.02 and z_qm > 1.5:
        reasons.append({
            "type": "style",
            "text": "Unusually high number of question marks",
            "weight": round(float(z_qm), 4),
        })

    return reasons


def explain_review(
    text: str,
    model_bundle: Any,
    top_k: int = 3,
    has_images: bool = False,
    has_video: bool = False,
) -> list[dict[str, Any]]:
    """Explain prediction for a review text by identifying language and style signals.

    Extracts:
    1. Language reasons: from linear model's coefficients times the review's TF-IDF values,
       takes the top contributing word n-grams toward the predicted class and turns them
       into a human-readable sentence.
    2. Style reasons: compares the review's stylometric values with the genuine-class
       mean and std from training (z-score above 1.5), then applies plain English templates.
    3. Media reasons: if images or video are attached to a genuine review, appends a positive
       credibility rationale.

    Always returns at least one reason; if nothing stands out, returns
    'No strong deceptive signals found' for genuine-looking reviews.

    If the chosen model is not linear (e.g., RandomForestClassifier), falls back to the
    LogisticRegression coefficients from a surrogate model trained on the same features.

    Args:
        text: Review text to analyze and explain.
        model_bundle: Trained Pipeline or dictionary bundle containing pipeline,
            stylometric_stats, and optional surrogate_model.
        top_k: Maximum number of reasons to return. Defaults to 3.
        has_images: Whether review includes customer photos. Defaults to False.
        has_video: Whether review includes customer video. Defaults to False.

    Returns:
        List of reason dictionaries:
        [{'type': 'language' | 'style' | 'pattern' | 'media', 'text': str, 'weight': float}]
    """
    # 1. Unpack bundle
    pipeline: Pipeline
    stylometric_stats: Optional[dict[str, Any]] = None
    surrogate_model: Optional[Any] = None

    if isinstance(model_bundle, dict):
        pipeline = model_bundle.get("pipeline", model_bundle.get("model"))
        stylometric_stats = model_bundle.get("stylometric_stats")
        surrogate_model = model_bundle.get("surrogate_model")
    elif isinstance(model_bundle, Pipeline):
        pipeline = model_bundle
        stylometric_stats = getattr(model_bundle, "stylometric_stats_", None)
        surrogate_model = getattr(model_bundle, "surrogate_model_", None)
    else:
        raise TypeError(
            f"Unsupported model_bundle type: {type(model_bundle)}. "
            "Expected Pipeline or dict bundle."
        )

    # Fallback for stylometric stats if stored in bundle_ attribute
    if stylometric_stats is None and hasattr(pipeline, "bundle_") and isinstance(pipeline.bundle_, dict):
        stylometric_stats = pipeline.bundle_.get("stylometric_stats")

    genuine_stats = (
        stylometric_stats.get("genuine", stylometric_stats)
        if isinstance(stylometric_stats, dict)
        else {}
    )

    # 2. Compute prediction probability and class
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba([text])[0]
        p_deceptive = float(probs[1]) if len(probs) > 1 else float(probs[0])
    else:
        preds = pipeline.predict([text])
        p_deceptive = 1.0 if preds[0] == 1 else 0.0

    label, confidence = classify_probability(p_deceptive)
    predicted_class = 1 if p_deceptive >= 0.5 else 0

    reasons: list[dict[str, Any]] = []

    # 3. Extract language reasons
    try:
        coef, _ = get_linear_coefficients(pipeline, surrogate_model=surrogate_model)
        lang_reason = _extract_language_reason(
            text=text,
            pipeline=pipeline,
            coef=coef,
            predicted_class=predicted_class,
        )
        if lang_reason is not None:
            reasons.append(lang_reason)
    except Exception as e:
        logger.debug("Could not extract language reason: %s", e)

    # 4. Extract style reasons
    try:
        style_reasons = _extract_style_reasons(
            text=text,
            genuine_stats=genuine_stats,
            predicted_class=predicted_class,
        )
        reasons.extend(style_reasons)
    except Exception as e:
        logger.debug("Could not extract style reasons: %s", e)

    # 5. Sanitize reasons to ensure raw feature names never leak
    sanitized_reasons: list[dict[str, Any]] = []
    for r in reasons:
        text_val = r["text"]
        # Remove any lingering internal feature prefixes
        text_val = text_val.replace("word_tfidf__", "").replace("char_tfidf__", "").replace("stylometric__", "")
        text_val = text_val.replace("tfidf__", "")
        sanitized_reasons.append({
            "type": r["type"],
            "text": text_val,
            "weight": round(float(r["weight"]), 4),
        })

    # 6. Fallback if no specific reasons triggered
    if not sanitized_reasons:
        if predicted_class == 0 or label == "genuine":
            sanitized_reasons.append({
                "type": "pattern",
                "text": "No strong deceptive signals found",
                "weight": 0.0,
            })
        elif label == "uncertain":
            sanitized_reasons.append({
                "type": "pattern",
                "text": "Review displays mixed signals with borderline deceptive probability",
                "weight": round(float(confidence), 4),
            })
        else:
            sanitized_reasons.append({
                "type": "pattern",
                "text": "Overall text patterns align with deceptive review characteristics",
                "weight": round(float(confidence), 4),
            })

    # Sort descending by weight
    sanitized_reasons.sort(key=lambda r: abs(r["weight"]), reverse=True)
    final_reasons = sanitized_reasons[:top_k]

    # If has_images or has_video is True on a genuine review, append a positive reason
    if (has_images or has_video) and (label == "genuine" or predicted_class == 0):
        media_reason = {
            "type": "media",
            "text": "Customer photo/video attached (higher credibility signal)",
            "weight": round(float(confidence), 4),
        }
        if not any(r.get("type") == "media" for r in final_reasons):
            final_reasons.append(media_reason)

    return final_reasons


def explain_prediction(
    text: str,
    model_bundle: Any,
    top_k: int = 3,
    has_images: bool = False,
    has_video: bool = False,
) -> dict[str, Any]:
    """Analyze a single review and return label, confidence, and readable reasons.

    Args:
        text: Review text.
        model_bundle: Trained Pipeline or dictionary bundle.
        top_k: Maximum reasons to return.
        has_images: Whether review includes customer photos. Defaults to False.
        has_video: Whether review includes customer video. Defaults to False.

    Returns:
        Dictionary with keys: 'label', 'confidence', 'p_deceptive', 'reasons'.
    """
    pipeline: Pipeline
    if isinstance(model_bundle, dict):
        pipeline = model_bundle.get("pipeline", model_bundle.get("model"))
    elif isinstance(model_bundle, Pipeline):
        pipeline = model_bundle
    else:
        raise TypeError(f"Unsupported model_bundle type: {type(model_bundle)}")

    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba([text])[0]
        p_deceptive = float(probs[1]) if len(probs) > 1 else float(probs[0])
    else:
        preds = pipeline.predict([text])
        p_deceptive = 1.0 if preds[0] == 1 else 0.0

    label, confidence = classify_probability(p_deceptive)
    reasons = explain_review(
        text,
        model_bundle,
        top_k=top_k,
        has_images=has_images,
        has_video=has_video,
    )

    return {
        "label": label,
        "confidence": confidence,
        "p_deceptive": round(p_deceptive, 4),
        "reasons": reasons,
    }


def demo_sample_reviews(model_path: Path | str = MODEL_PATH) -> None:
    """Run sample explanations on 10 diverse reviews and print formatted output."""
    path = Path(model_path)
    if not path.exists():
        print(f"Model file not found at {path}. Run 'python -m ml.train' first.")
        return

    bundle = joblib.load(path)

    sample_reviews = [
        # 1. Clear genuine with detailed usage
        "I bought this blender 3 months ago. The 700-watt motor blends frozen fruits smoothly, although it is rather loud on high speed.",
        # 2. Deceptive extreme praise
        "AMAZING UNBELIEVABLE BEST EVER INCREDIBLE 100% PERFECT MUST BUY NOW WOW!!!",
        # 3. Genuine concise
        "Ordered on Tuesday, arrived Thursday. The blue color matches the picture and the fit is true to size for my 10.5 shoes.",
        # 4. Deceptive hype phrases
        "Highly recommend this miracle product to everyone! Best ever experience, changed my life completely!",
        # 5. Short and generic
        "Good product. Nice.",
        # 6. Deceptive all caps with many exclamation marks
        "DO NOT WAIT BUY THIS RIGHT NOW!! BEST PURCHASE OF MY ENTIRE LIFE GUARANTEED!!",
        # 7. Genuine balanced criticism
        "The battery lasts about 6 hours instead of the advertised 8 hours. Sound quality is decent for $35 earbuds.",
        # 8. Borderline review
        "A pretty decent item that works as described. Would probably recommend to friends.",
        # 9. Deceptive superlatives without specifics
        "Top quality best of the best! Outstanding perfection in every way imaginable!",
        # 10. Genuine personal experience
        "My wife and I used this tent on a 2-day camping trip in the rain. Kept us completely dry and packed up easily.",
    ]

    print("\n" + "=" * 80)
    print("                    REVIEWLENS EXPLANATION DEMO (10 REVIEWS)")
    print("=" * 80)

    for i, review in enumerate(sample_reviews, 1):
        result = explain_prediction(review, bundle, top_k=3)
        print(f"\nReview #{i}: \"{review[:75]}{'...' if len(review) > 75 else ''}\"")
        print(f"  Label:       {result['label'].upper()} (p_deceptive={result['p_deceptive']:.4f})")
        print(f"  Confidence:  {result['confidence']:.4f}")
        print("  Reasons:")
        for r in result["reasons"]:
            print(f"    - [{r['type']}] {r['text']} (weight: {r['weight']:.4f})")

    print("\n" + "=" * 80 + "\n")


def main() -> None:
    """CLI entrypoint for ml/explain.py."""
    parser = argparse.ArgumentParser(
        description="Explain ReviewLens fake review detection predictions."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=MODEL_PATH,
        help="Path to trained model artifact (default: models/model.joblib)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        default=True,
        help="Run explanation demo on 10 sample reviews",
    )
    args = parser.parse_args()

    if args.demo:
        demo_sample_reviews(args.model_path)


if __name__ == "__main__":
    main()
